import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.signal import convolve2d
import skimage.filters.rank as sfr
from skimage.morphology import disk

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Depth Anything V2 depth estimation + haze generation (compatible with old logic)."
    )
    parser.add_argument("--img-path", type=str, required=True, help="Single image path or folder path")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Use vits (smallest) by default",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path. If empty, use checkpoints/depth_anything_v2_<encoder>.pth",
    )
    parser.add_argument(
        "--add-fog-method",
        type=str,
        default="mean",
        choices=["mean", "mean+min+max", "mean+min+max+th"],
    )
    parser.add_argument("--ext", type=str, default="jpg")
    parser.add_argument("--seed", type=int, default=None, help="Fix split randomness")
    parser.add_argument("--mean-window", type=int, default=96)
    parser.add_argument("--w-max", type=int, default=128)
    parser.add_argument("--w-min", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=70.0, help="Atmospheric light percentile")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0, 0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69],
        help="Haze density coefficients",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save t, J*t, A*(1-t), and I for each beta for presentation",
    )
    return parser.parse_args()


def resolve_checkpoint(args):
    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt = os.path.join(script_dir, "checkpoints", f"depth_anything_v2_{args.encoder}.pth")

    if not os.path.exists(ckpt) or os.path.getsize(ckpt) == 0:
        raise FileNotFoundError(
            f"Checkpoint not found or empty: {ckpt}. Please download the weight file first."
        )
    return ckpt


def collect_images(img_path, ext):
    if os.path.isfile(img_path):
        return [img_path]

    if not os.path.isdir(img_path):
        raise FileNotFoundError(f"Can not find --img-path: {img_path}")

    pattern = os.path.join(img_path, "**", f"*.{ext}")
    paths = glob.glob(pattern, recursive=True)

    if not paths:
        raise RuntimeError(f"No images found by pattern: {pattern}")

    return sorted(paths)


def normalize_to_01(arr):
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def apply_old_style_smoothing(depth_map, method, w_max, w_min, w_mean):
    depth_proc = depth_map.astype(np.float32)

    if method in ("mean+min+max", "mean+min+max+th"):
        depth_u8 = (normalize_to_01(depth_proc) * 255.0).astype(np.uint8)

        depth_max = sfr.maximum(depth_u8, disk(w_max)).astype(np.float32)

        if method == "mean+min+max+th":
            th = np.percentile(depth_u8, 15)
            keep_mask = depth_u8 < th
            depth_max[keep_mask] = depth_u8[keep_mask]

        depth_proc = sfr.minimum(depth_max.astype(np.uint8), disk(w_min)).astype(np.float32)

    window = np.ones((w_mean, w_mean), dtype=np.float32)
    window /= np.sum(window)
    depth_smooth = convolve2d(depth_proc, window, mode="same", boundary="symm")

    return depth_smooth.astype(np.float32)


def generate_fog_components(rgb_image, depth_smooth, beta, alpha):
    # DAV2 relative depth is typically larger for nearer pixels; invert so farther regions get denser haze.
    depth_norm = 1.0 - normalize_to_01(depth_smooth)
    depth_scaled = depth_norm * (10.0 - 0.1) + 0.1

    transmission = np.exp(-beta * depth_scaled)
    atmosphere = np.percentile(rgb_image, alpha)

    j_weighted = rgb_image.astype(np.float32) * transmission[..., None]
    a_weighted = np.full_like(rgb_image, atmosphere, dtype=np.float32) * (1.0 - transmission[..., None])
    fog = j_weighted + a_weighted

    return (
        transmission.astype(np.float32),
        np.clip(j_weighted, 0, 255).astype(np.uint8),
        np.clip(a_weighted, 0, 255).astype(np.uint8),
        np.clip(fog, 0, 255).astype(np.uint8),
    )


def save_intermediate_outputs(out_dir, stem, beta_index, beta, transmission, j_weighted, a_weighted, fog):
    sample_dir = os.path.join(out_dir, "intermediates", stem)
    os.makedirs(sample_dir, exist_ok=True)

    prefix = f"beta{beta_index:02d}_{beta:.2f}"

    t_u8 = np.clip(transmission * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(t_u8).save(os.path.join(sample_dir, f"{prefix}_t.png"))

    Image.fromarray(j_weighted).save(os.path.join(sample_dir, f"{prefix}_Jt.png"))
    Image.fromarray(a_weighted).save(os.path.join(sample_dir, f"{prefix}_A1mt.png"))
    Image.fromarray(fog).save(os.path.join(sample_dir, f"{prefix}_I.png"))


def save_depth_outputs(depth, out_depth_dir, stem):
    os.makedirs(out_depth_dir, exist_ok=True)

    depth_npy_path = os.path.join(out_depth_dir, f"{stem}_depth.npy")
    np.save(depth_npy_path, depth)

    depth_u8 = (normalize_to_01(depth) * 255.0).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
    depth_png_path = os.path.join(out_depth_dir, f"{stem}_depth.png")
    cv2.imwrite(depth_png_path, depth_vis)

    return depth_npy_path, depth_png_path


def main():
    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available() and not args.no_cuda
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint = resolve_checkpoint(args)
    image_paths = collect_images(args.img_path, args.ext)

    print(f"-> Loading DAV2 model: {args.encoder}")
    print(f"-> Checkpoint: {checkpoint}")
    print(f"-> Device: {device}")

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    split_dirs = {name: os.path.join(outdir, name) for name in ["train", "val", "test"]}
    for split_dir in split_dirs.values():
        os.makedirs(split_dir, exist_ok=True)

    depth_dir = os.path.join(outdir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    total_variants = len(image_paths) * len(args.betas)
    rng = np.random.default_rng(args.seed)
    sampled = rng.choice(total_variants, int(total_variants * 0.7), replace=False)
    val_arr = set(sampled[int(total_variants * 0.6):].tolist())
    train_arr = set(sampled[:int(total_variants * 0.6)].tolist())

    counters = {"train": 0, "val": 0, "test": 0}
    img_path_map = {"train": [], "val": [], "test": []}
    label_map = {"train": [], "val": [], "test": []}

    print(f"-> Predicting depth and generating haze on {len(image_paths)} images")

    for idx, image_path in enumerate(image_paths):
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"   Skipped unreadable image: {image_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = model.infer_image(bgr, args.input_size)

        stem = os.path.splitext(os.path.basename(image_path))[0]
        save_depth_outputs(depth, depth_dir, stem)

        depth_smooth = apply_old_style_smoothing(
            depth,
            args.add_fog_method,
            w_max=args.w_max,
            w_min=args.w_min,
            w_mean=args.mean_window,
        )

        for b_idx, beta in enumerate(args.betas):
            transmission, j_weighted, a_weighted, fog_rgb = generate_fog_components(
                rgb,
                depth_smooth,
                beta=float(beta),
                alpha=args.alpha,
            )
            fog_pil = Image.fromarray(fog_rgb)

            if args.save_intermediates:
                save_intermediate_outputs(
                    outdir,
                    stem,
                    b_idx,
                    float(beta),
                    transmission,
                    j_weighted,
                    a_weighted,
                    fog_rgb,
                )

            flat_idx = idx * len(args.betas) + b_idx
            if flat_idx in train_arr:
                split = "train"
            elif flat_idx in val_arr:
                split = "val"
            else:
                split = "test"

            save_name = f"{counters[split]}.jpg"
            save_path = os.path.join(split_dirs[split], save_name)
            fog_pil.save(save_path)

            img_path_map[split].append(image_path)
            label_map[split].append(int(b_idx))
            counters[split] += 1

        print(f"   Processed {idx + 1}/{len(image_paths)}: {image_path}")

    for split in ["train", "val", "test"]:
        np.save(os.path.join(split_dirs[split], f"{split}_label.npy"), np.array(label_map[split]))
        np.save(os.path.join(split_dirs[split], f"{split}_path.npy"), np.array(img_path_map[split]))

    print("-> Done!")


if __name__ == "__main__":
    main()

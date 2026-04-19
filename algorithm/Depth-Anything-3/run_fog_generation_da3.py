import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.signal import convolve2d
import skimage.filters.rank as sfr
from skimage.morphology import disk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from depth_anything_3.api import DepthAnything3  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="DepthAnything3 (small) + atmospheric scattering outputs for PPT"
    )
    parser.add_argument("--img-path", type=str, required=True, help="Single image path or folder path")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3-SMALL",
        help="HF repo id or local model directory",
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
    )
    parser.add_argument(
        "--add-fog-method",
        type=str,
        default="mean",
        choices=["mean", "mean+min+max", "mean+min+max+th"],
    )
    parser.add_argument("--ext", type=str, default="jpg")
    parser.add_argument("--mean-window", type=int, default=96)
    parser.add_argument("--w-max", type=int, default=128)
    parser.add_argument("--w-min", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=70.0)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0, 0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69],
    )
    parser.add_argument(
        "--depth-order",
        type=str,
        default="far-large",
        choices=["far-large", "near-large"],
        help="How model depth values correlate with actual distance",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save t, J*t, A*(1-t), and I for each beta",
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HF endpoint for model download",
    )
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


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


def generate_fog_components(rgb_image, depth_smooth, beta, alpha, depth_order):
    depth_norm = normalize_to_01(depth_smooth)
    if depth_order == "near-large":
        depth_norm = 1.0 - depth_norm

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


def save_depth_outputs(depth, out_depth_dir, stem):
    os.makedirs(out_depth_dir, exist_ok=True)

    depth_npy_path = os.path.join(out_depth_dir, f"{stem}_depth.npy")
    np.save(depth_npy_path, depth)

    depth_u8 = (normalize_to_01(depth) * 255.0).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
    depth_png_path = os.path.join(out_depth_dir, f"{stem}_depth.png")
    cv2.imwrite(depth_png_path, depth_vis)


def save_intermediate_outputs(out_dir, stem, beta_index, beta, transmission, j_weighted, a_weighted, fog):
    sample_dir = os.path.join(out_dir, "intermediates", stem)
    os.makedirs(sample_dir, exist_ok=True)

    prefix = f"beta{beta_index:02d}_{beta:.2f}"

    t_u8 = np.clip(transmission * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(t_u8).save(os.path.join(sample_dir, f"{prefix}_t.png"))
    np.save(os.path.join(sample_dir, f"{prefix}_t.npy"), transmission)

    Image.fromarray(j_weighted).save(os.path.join(sample_dir, f"{prefix}_Jt.png"))
    Image.fromarray(a_weighted).save(os.path.join(sample_dir, f"{prefix}_A1mt.png"))
    Image.fromarray(fog).save(os.path.join(sample_dir, f"{prefix}_I.png"))


def main():
    args = parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    device = (
        "cuda"
        if torch.cuda.is_available() and not args.no_cuda
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    image_paths = collect_images(args.img_path, args.ext)

    print(f"-> Loading DA3 model: {args.model_dir}")
    print(f"-> Device: {device}")

    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device=device)
    model.eval()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    depth_dir = os.path.join(outdir, "depth")
    fog_dir = os.path.join(outdir, "fog")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(fog_dir, exist_ok=True)

    print(f"-> Predicting depth and generating haze on {len(image_paths)} images")

    for idx, image_path in enumerate(image_paths):
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"   Skipped unreadable image: {image_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        prediction = model.inference(
            [image_path],
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            export_dir=None,
            export_format="mini_npz",
        )
        depth = prediction.depth[0].astype(np.float32)

        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

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
                depth_order=args.depth_order,
            )

            fog_name = f"{stem}_beta{b_idx:02d}_{beta:.2f}.png"
            Image.fromarray(fog_rgb).save(os.path.join(fog_dir, fog_name))

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

        print(f"   Processed {idx + 1}/{len(image_paths)}: {image_path}")

    print("-> Done!")


if __name__ == "__main__":
    main()

import argparse
import copy
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from denoiser import Denoiser
from main_jit import get_args_parser
from util.paired_dataset import PairedImageDataset


def parse_args():
    parser = argparse.ArgumentParser("JiT paired inference", parents=[get_args_parser()])
    parser.add_argument("--split", default="test", type=str, help="Dataset split for inference")
    parser.add_argument("--num_samples", default=8, type=int, help="How many samples to save")
    parser.add_argument("--start_index", default=0, type=int, help="Dataset start index")
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
        help="Directory to save inference outputs; defaults to <resume>/paired_infer_<split>",
    )
    parser.add_argument(
        "--save_triptych",
        action="store_true",
        help="Save concatenated input/pred/gt images in addition to separate files",
    )
    return parser.parse_args()


def to_model_range(x):
    x = x.to(torch.float32).div_(255.0)
    return x * 2.0 - 1.0


def to_uint8_image(x):
    x = x.detach().cpu().clamp_(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()[:, :, ::-1]


def swap_to_ema1(model):
    current_state = copy.deepcopy(model.state_dict())
    ema_state = copy.deepcopy(model.state_dict())
    for i, (name, _value) in enumerate(model.named_parameters()):
        ema_state[name] = model.ema_params1[i]
    model.load_state_dict(ema_state)
    return current_state


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = PairedImageDataset(
        root=args.data_path,
        split=args.split,
        lq_dirname=args.lq_dirname,
        gt_dirname=args.gt_dirname,
        img_size=args.img_size,
        pairing_mode=args.pairing_mode,
        pair_meta=args.pair_meta if args.pair_meta else None,
        random_flip=False,
    )

    start = max(args.start_index, 0)
    end = min(start + max(args.num_samples, 0), len(dataset))
    if start >= end:
        raise ValueError(f"Invalid sample range: start={start}, end={end}, dataset_size={len(dataset)}")

    subset = Subset(dataset, list(range(start, end)))
    loader = DataLoader(
        subset,
        batch_size=args.gen_bsz,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = Denoiser(args).to(device)

    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.ema_params1 = [checkpoint["model_ema1"][name].to(device) for name, _ in model.named_parameters()]
    model.ema_params2 = [checkpoint["model_ema2"][name].to(device) for name, _ in model.named_parameters()]
    del checkpoint

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.resume) / f"paired_infer_{args.split}"
    save_dir.mkdir(parents=True, exist_ok=True)

    original_state = swap_to_ema1(model)
    model.eval()

    sample_index = start
    amp_enabled = torch.cuda.is_available()
    with torch.no_grad():
        for lq, gt, labels in loader:
            y_cond = to_model_range(lq.to(device, non_blocking=True))
            gt = to_model_range(gt.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                pred = model.generate_paired(
                    y_cond,
                    labels if args.use_paired_label_condition else None,
                )

            for b in range(pred.size(0)):
                stem = f"{sample_index:05d}"
                lq_img = to_uint8_image(y_cond[b])
                pred_img = to_uint8_image(pred[b])
                gt_img = to_uint8_image(gt[b])

                cv2.imwrite(str(save_dir / f"{stem}_input.png"), lq_img)
                cv2.imwrite(str(save_dir / f"{stem}_pred.png"), pred_img)
                cv2.imwrite(str(save_dir / f"{stem}_gt.png"), gt_img)

                if args.save_triptych:
                    triptych = np.concatenate([lq_img, pred_img, gt_img], axis=1)
                    cv2.imwrite(str(save_dir / f"{stem}_triptych.png"), triptych)

                mse = torch.mean((pred[b] - gt[b]) ** 2).item()
                psnr = -10.0 * math.log10(max(mse, 1e-12))
                print(f"[{stem}] psnr={psnr:.3f}dB")
                sample_index += 1

    model.load_state_dict(original_state)
    print(f"Saved paired inference results to: {save_dir}")


if __name__ == "__main__":
    main()

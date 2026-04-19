#!/usr/bin/env python3
"""Runtime benchmark for C2PNet on image-pair datasets (haze_images + original_images)."""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from models.C2PNet import C2PNet


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HAZE_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C2PNet runtime benchmark on current dataset")
    parser.add_argument("--train-root", type=str, required=True, help="Path to train split root, e.g. datasets/train")
    parser.add_argument("--val-root", type=str, required=True, help="Path to valid split root, e.g. datasets/valid")
    parser.add_argument("--batch-size", type=int, default=2, help="Train batch size")
    parser.add_argument("--val-batch-size", type=int, default=2, help="Valid batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--image-size", type=int, default=240, help="Square crop/resize size")
    parser.add_argument("--gps", type=int, default=3, help="C2PNet residual groups")
    parser.add_argument("--blocks", type=int, default=19, help="C2PNet residual blocks per group")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Benchmark epochs")
    parser.add_argument("--max-train-batches", type=int, default=100, help="Cap train batches for quick benchmark")
    parser.add_argument("--max-val-batches", type=int, default=20, help="Cap val batches for quick benchmark")
    parser.add_argument("--print-every", type=int, default=20, help="Print loss interval")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    files.sort()
    return files


def haze_to_clean_stem(stem: str) -> str:
    return HAZE_SUFFIX_RE.sub("", stem)


class HazeCleanPairDataset(Dataset):
    def __init__(self, split_root: Path, image_size: int) -> None:
        haze_dir = split_root / "haze_images"
        clean_dir = split_root / "original_images"
        if not haze_dir.exists() or not clean_dir.exists():
            raise FileNotFoundError(f"Expected haze_images and original_images under: {split_root}")

        clean_files = list_images(clean_dir)
        clean_by_stem: Dict[str, Path] = {p.stem: p for p in clean_files}

        pairs: List[Tuple[Path, Path]] = []
        missing = 0
        for haze_path in list_images(haze_dir):
            clean_stem = haze_to_clean_stem(haze_path.stem)
            clean_path = clean_by_stem.get(clean_stem)
            if clean_path is None:
                missing += 1
                continue
            pairs.append((haze_path, clean_path))

        if not pairs:
            raise RuntimeError(f"No haze-clean pairs found under {split_root}")
        if missing > 0:
            print(f"[warn] {split_root.name}: {missing} haze images skipped (no matching clean stem)")

        self.samples = pairs
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        haze_path, clean_path = self.samples[idx]
        haze = Image.open(haze_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")
        return self.tf(haze), self.tf(clean)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
    print_every: int,
    max_batches: int,
) -> Tuple[float, float, int]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_count = 0
    num_batches = 0
    running = 0.0
    start = time.perf_counter()

    for batch_idx, (haze, clean) in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        num_batches += 1

        haze = haze.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            pred = model(haze)
            loss = criterion(pred, clean)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = haze.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs
        running += float(loss.item())

        if train_mode and print_every > 0 and (batch_idx % print_every == 0):
            print(f"{desc} [{batch_idx}/{len(loader)}] avg_loss={running/print_every:.6f}")
            running = 0.0

    elapsed = time.perf_counter() - start
    mean_loss = 0.0 if total_count == 0 else (total_loss / total_count)
    return mean_loss, elapsed, num_batches


def main() -> None:
    args = parse_args()
    print(args)

    set_seed(args.seed)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    train_ds = HazeCleanPairDataset(Path(args.train_root), image_size=args.image_size)
    val_ds = HazeCleanPairDataset(Path(args.val_root), image_size=args.image_size)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = C2PNet(gps=args.gps, blocks=args.blocks).to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    total_time = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_sec, tr_batches = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            desc=f"train-e{epoch}",
            print_every=args.print_every,
            max_batches=args.max_train_batches,
        )
        va_loss, va_sec, va_batches = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            desc=f"valid-e{epoch}",
            print_every=0,
            max_batches=args.max_val_batches,
        )

        epoch_time = tr_sec + va_sec
        total_time += epoch_time

        full_train_batches = len(train_loader)
        full_val_batches = len(val_loader)
        est_full_train_sec = tr_sec * (full_train_batches / max(tr_batches, 1))
        est_full_val_sec = va_sec * (full_val_batches / max(va_batches, 1))
        est_full_epoch_sec = est_full_train_sec + est_full_val_sec

        print(
            f"[epoch {epoch:03d}] train_loss={tr_loss:.6f} val_loss={va_loss:.6f} "
            f"bench_train_sec={tr_sec:.1f} bench_val_sec={va_sec:.1f} bench_epoch_sec={epoch_time:.1f}"
        )
        print(
            f"[estimate] full_train_batches={full_train_batches} full_val_batches={full_val_batches} "
            f"full_epoch_sec~{est_full_epoch_sec:.1f} ({est_full_epoch_sec/60:.2f} min)"
        )

    print(f"Benchmark done. measured_time={total_time:.1f}s for {args.epochs} epoch(s)")


if __name__ == "__main__":
    main()

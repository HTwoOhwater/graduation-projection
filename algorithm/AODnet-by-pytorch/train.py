import argparse
import os
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

from model import AODnet


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HAZE_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AOD-Net from scratch on image pairs")
    parser.add_argument("--dataroot", required=True, help="Train split root, e.g. datasets/train")
    parser.add_argument("--valDataroot", required=True, help="Valid split root, e.g. datasets/valid")
    parser.add_argument("--batchSize", type=int, default=16, help="Train batch size")
    parser.add_argument("--valBatchSize", type=int, default=16, help="Valid batch size")
    parser.add_argument("--nEpochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--threads", type=int, default=4, help="Data loader workers")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize for both haze and gt")
    parser.add_argument("--printEvery", type=int, default=50, help="Print interval (batches)")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug/benchmark cap per train epoch")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Debug/benchmark cap per val epoch")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--exp", default="model_pretrained", help="Checkpoint directory")
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
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

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
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_count = 0
    start = time.perf_counter()
    running = 0.0

    iterator = tqdm(loader, desc=desc, leave=False)
    for batch_idx, (haze, clean) in enumerate(iterator, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break

        haze = haze.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

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

        if print_every > 0 and (batch_idx % print_every == 0):
            print(f"{desc} [{batch_idx}/{len(loader)}] avg_loss={running/print_every:.6f}")
            running = 0.0

    if total_count == 0:
        return 0.0, 0.0

    elapsed = time.perf_counter() - start
    return total_loss / total_count, elapsed


def save_checkpoint(model: nn.Module, out_dir: Path, epoch: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"AOD_net_epoch_{epoch}.pth"
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")


def main() -> None:
    args = parse_args()
    print(args)

    set_seed(args.seed)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    train_ds = HazeCleanPairDataset(Path(args.dataroot), image_size=args.image_size)
    val_ds = HazeCleanPairDataset(Path(args.valDataroot), image_size=args.image_size)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.threads,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.threads > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.valBatchSize,
        shuffle=False,
        num_workers=args.threads,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.threads > 0),
    )

    model = AODnet().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_wall = 0.0
    for epoch in range(1, args.nEpochs + 1):
        tr_loss, tr_sec = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            desc=f"train-e{epoch}",
            print_every=args.printEvery,
            max_batches=args.max_train_batches,
        )
        va_loss, va_sec = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            desc=f"valid-e{epoch}",
            print_every=0,
            max_batches=args.max_val_batches,
        )

        epoch_sec = tr_sec + va_sec
        total_wall += epoch_sec
        eta = (total_wall / epoch) * (args.nEpochs - epoch)
        print(
            f"[epoch {epoch:03d}] train_loss={tr_loss:.6f} val_loss={va_loss:.6f} "
            f"train_sec={tr_sec:.1f} val_sec={va_sec:.1f} epoch_sec={epoch_sec:.1f} ETA={eta/60:.1f} min"
        )

        save_checkpoint(model, Path(args.exp), epoch)

    print(f"Done. total_time={total_wall/60:.2f} min for {args.nEpochs} epochs")


if __name__ == "__main__":
    main()

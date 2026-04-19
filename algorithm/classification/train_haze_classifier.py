#!/usr/bin/env python3
"""Train a Swin-based haze concentration classifier (levels 0-9).

Label rule:
- filename suffix uses ..._{A_index}_{beta_index}.ext
- beta_index in [0, 9] is used as class label
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Swin_T_Weights, swin_t


LEVEL_RE = re.compile(r"_(\d+)_(\d+)$")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_beta_idx_from_name(path: Path) -> int:
    m = LEVEL_RE.search(path.stem)
    if m is None:
        raise ValueError(f"Cannot parse label from filename: {path.name}")
    beta_idx = int(m.group(2))
    if not (0 <= beta_idx <= 9):
        raise ValueError(f"beta_index out of expected range [0,9] in {path.name}")
    return beta_idx


class HazeLevelDataset(Dataset):
    """Dataset for haze level classification from generated haze images."""

    def __init__(self, data_root: Path, split: str, transform: transforms.Compose) -> None:
        self.transform = transform
        haze_dir = data_root / split / "haze_images"
        if not haze_dir.exists():
            raise FileNotFoundError(f"Split haze directory not found: {haze_dir}")

        self.samples: List[Tuple[Path, int]] = []
        for p in sorted(haze_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in VALID_EXTS:
                continue
            label = extract_beta_idx_from_name(p)
            self.samples.append((p, label))

        if not self.samples:
            raise RuntimeError(f"No haze images found in {haze_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label


def make_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    # Keep augmentation light to avoid changing haze level semantics too much.
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def make_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 0,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda" and train_mode))

    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
        steps += 1

    if steps == 0:
        return {"loss": 0.0, "acc": 0.0}
    return {"loss": total_loss / steps, "acc": total_acc / steps}


def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    correct = [0 for _ in range(10)]
    total = [0 for _ in range(10)]

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = torch.argmax(model(images), dim=1)
            for cls in range(10):
                mask = labels == cls
                cls_total = int(mask.sum().item())
                if cls_total == 0:
                    continue
                total[cls] += cls_total
                correct[cls] += int((preds[mask] == labels[mask]).sum().item())

    out: Dict[str, float] = {}
    for cls in range(10):
        if total[cls] == 0:
            out[f"class_{cls}"] = -1.0
        else:
            out[f"class_{cls}"] = correct[cls] / total[cls]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin haze concentration classifier")
    parser.add_argument("--data-root", type=str, default="datasets", help="Root containing train/valid/test")
    parser.add_argument("--output-dir", type=str, default="Swin_Transformer/outputs_haze_cls", help="Checkpoint/log output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained Swin-T weights")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug: limit train batches per epoch")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Debug: limit valid/test batches")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate checkpoint on valid and test")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint for eval-only or resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = make_transforms(args.image_size)
    train_ds = HazeLevelDataset(data_root, "train", train_tf)
    valid_ds = HazeLevelDataset(data_root, "valid", eval_tf)
    test_ds = HazeLevelDataset(data_root, "test", eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = make_model(num_classes=10, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_valid_acc = -1.0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and not args.eval_only:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_valid_acc = float(ckpt.get("best_valid_acc", -1.0))

    # Save dataset stats for reproducibility.
    meta = {
        "train_size": len(train_ds),
        "valid_size": len(valid_ds),
        "test_size": len(test_ds),
        "args": vars(args),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.eval_only:
        valid_metrics = run_epoch(
            model, valid_loader, criterion, optimizer=None, device=device, use_amp=args.amp, max_batches=args.max_eval_batches
        )
        test_metrics = run_epoch(
            model, test_loader, criterion, optimizer=None, device=device, use_amp=args.amp, max_batches=args.max_eval_batches
        )
        per_class = evaluate_per_class(model, test_loader, device, max_batches=args.max_eval_batches)
        print(f"[eval-only] valid loss={valid_metrics['loss']:.4f}, acc={valid_metrics['acc']:.4f}")
        print(f"[eval-only] test  loss={test_metrics['loss']:.4f}, acc={test_metrics['acc']:.4f}")
        print("[eval-only] test per-class acc:")
        for k, v in per_class.items():
            print(f"  {k}: {v:.4f}")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            use_amp=args.amp,
            max_batches=args.max_train_batches,
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            criterion,
            optimizer=None,
            device=device,
            use_amp=args.amp,
            max_batches=args.max_eval_batches,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f} | "
            f"valid loss={valid_metrics['loss']:.4f}, acc={valid_metrics['acc']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_valid_acc": best_valid_acc,
            "args": vars(args),
        }
        torch.save(ckpt, output_dir / "last.pt")

        if valid_metrics["acc"] > best_valid_acc:
            best_valid_acc = valid_metrics["acc"]
            ckpt["best_valid_acc"] = best_valid_acc
            torch.save(ckpt, output_dir / "best.pt")

    best_ckpt = torch.load(output_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model"])
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        device=device,
        use_amp=args.amp,
        max_batches=args.max_eval_batches,
    )
    per_class = evaluate_per_class(model, test_loader, device, max_batches=args.max_eval_batches)

    print(f"[final] best valid acc={best_valid_acc:.4f}")
    print(f"[final] test loss={test_metrics['loss']:.4f}, acc={test_metrics['acc']:.4f}")
    print("[final] test per-class acc:")
    for k, v in per_class.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

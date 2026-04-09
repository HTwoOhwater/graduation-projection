#!/usr/bin/env python3
"""Train haze concentration classifiers (levels 0-9) with torchvision backbones.

Label rule:
- filename suffix uses ..._{A_index}_{beta_index}.ext
- beta_index in [0, 9] is used as class label
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision import models


LEVEL_RE = re.compile(r"_(\d+)_(\d+)$")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_MODELS = (
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "swin_t",
    "vit_b_16",
)


def set_seed(seed: int) -> None:
    # 固定随机种子，保证多模型对比时结果可复现。
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int]:
    """初始化分布式环境，返回 (is_distributed, rank, world_size, local_rank)。"""
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_distributed = args.distributed or env_world_size > 1
    if not use_distributed:
        return False, 0, 1, 0

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.set_device(local_rank)
        # 新版 PyTorch 支持在 init_process_group 指定 device_id，可避免后续 barrier 设备告警。
        try:
            dist.init_process_group(
                backend=args.dist_backend,
                rank=rank,
                world_size=world_size,
                device_id=local_rank,
            )
        except TypeError:
            # 兼容旧版 PyTorch（无 device_id 参数）。
            dist.init_process_group(backend=args.dist_backend, rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    return True, rank, world_size, local_rank


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def distributed_barrier(is_distributed: bool, device: torch.device) -> None:
    """带设备上下文的 barrier，减少 NCCL 下的设备选择告警。"""
    if not is_distributed:
        return
    if dist.get_backend() == "nccl" and device.type == "cuda" and device.index is not None:
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


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
        # 遍历当前 split 下的所有图像，并从文件名解析 beta 等级作为标签。
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
    # 统一将图像缩放到固定分辨率，确保 DataLoader 可正常 stack。
    # 注意：Resize(单个整数)只会固定短边，长边仍可变，容易引发 batch 尺寸不一致报错。
    fixed_resize = transforms.Resize((image_size, image_size))

    # 无增强训练版本：用于先验证模型泛化问题，避免随机变换带来的额外扰动。
    train_tf = transforms.Compose(
        [
            fixed_resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            fixed_resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def make_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    # 使用 torchvision 官方权重，便于统一比较不同结构。
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if arch == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if arch == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {arch}")


def parse_models(spec: str, fallback_model: str) -> List[str]:
    # 支持通过 --models 一次性指定多个模型进行对比。
    if not spec.strip():
        return [fallback_model]
    names = [x.strip() for x in spec.split(",") if x.strip()]
    invalid = [n for n in names if n not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(f"Unsupported models in --models: {invalid}")
    return names


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
    is_distributed: bool,
    rank: int,
    desc: str,
    max_batches: int = 0,
) -> Dict[str, float]:
    # optimizer 为 None 时表示验证/测试阶段。
    train_mode = optimizer is not None
    model.train(train_mode)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda" and train_mode))

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    iterator = loader
    if is_main_process(rank):
        iterator = tqdm(loader, desc=desc, leave=False)

    for batch_idx, (images, labels) in enumerate(iterator):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        if train_mode:
            # 训练阶段执行反向传播与参数更新。
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += float((torch.argmax(logits, dim=1) == labels).sum().item())
        total_count += bs

    if is_distributed:
        stat = torch.tensor([total_loss, total_correct, float(total_count)], device=device)
        dist.all_reduce(stat, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_count = stat.tolist()

    if total_count == 0:
        return {"loss": 0.0, "acc": 0.0}
    return {"loss": total_loss / total_count, "acc": total_correct / total_count}


def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_distributed: bool,
    max_batches: int = 0,
) -> Dict[str, float]:
    # 计算每个浓度等级（0-9）的独立准确率，便于分析模型偏置。
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

    if is_distributed:
        c = torch.tensor(correct, device=device, dtype=torch.float64)
        t = torch.tensor(total, device=device, dtype=torch.float64)
        dist.all_reduce(c, op=dist.ReduceOp.SUM)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        correct = [int(x) for x in c.tolist()]
        total = [int(x) for x in t.tolist()]

    out: Dict[str, float] = {}
    for cls in range(10):
        if total[cls] == 0:
            out[f"class_{cls}"] = -1.0
        else:
            out[f"class_{cls}"] = correct[cls] / total[cls]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train haze concentration classifier")
    parser.add_argument("--model", type=str, default="swin_t", choices=list(SUPPORTED_MODELS), help="Single model architecture")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated models for multi-run, e.g. resnet18,resnet50,swin_t",
    )
    parser.add_argument("--data-root", type=str, default="datasets", help="Root containing train/valid/test")
    parser.add_argument("--output-dir", type=str, default="result/classification", help="Root output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug: limit train batches per epoch")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Debug: limit valid/test batches")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Early stopping patience on valid acc (0 disables)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate checkpoint on valid and test")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint for eval-only or resume")
    parser.add_argument("--distributed", action="store_true", help="Enable DDP training (recommended with torchrun)")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="DDP backend, e.g. nccl/gloo")
    return parser.parse_args()


def run_for_model(
    args: argparse.Namespace,
    model_name: str,
    data_root: Path,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    train_size: int,
    valid_size: int,
    test_size: int,
    device: torch.device,
    distributed: bool,
    rank: int,
) -> Dict[str, object]:
    # 每个模型单独写入 result/classification/<model_name> 目录，避免互相覆盖。
    output_dir = Path(args.output_dir) / model_name
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(arch=model_name, num_classes=10, pretrained=args.pretrained).to(device)
    if distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[device.index], output_device=device.index)
        else:
            model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_valid_acc = -1.0
    no_improve_epochs = 0
    checkpoint_path = args.checkpoint
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        target = model.module if isinstance(model, DDP) else model
        target.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and not args.eval_only:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_valid_acc = float(ckpt.get("best_valid_acc", -1.0))
        no_improve_epochs = int(ckpt.get("no_improve_epochs", 0))

    # 记录当前模型运行参数，便于论文复现实验。
    meta = {
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
        "model": model_name,
        "args": vars(args),
    }
    if is_main_process(rank):
        (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.eval_only:
        # 仅评估模式：直接读取权重并输出 valid/test 指标。
        valid_metrics = run_epoch(
            model,
            valid_loader,
            criterion,
            optimizer=None,
            device=device,
            use_amp=args.amp,
            is_distributed=distributed,
            rank=rank,
            desc=f"{model_name}-valid",
            max_batches=args.max_eval_batches,
        )
        test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer=None,
            device=device,
            use_amp=args.amp,
            is_distributed=distributed,
            rank=rank,
            desc=f"{model_name}-test",
            max_batches=args.max_eval_batches,
        )
        per_class = evaluate_per_class(
            model, test_loader, device, is_distributed=distributed, max_batches=args.max_eval_batches
        )
        if is_main_process(rank):
            print(f"[{model_name}][eval-only] valid loss={valid_metrics['loss']:.4f}, acc={valid_metrics['acc']:.4f}")
            print(f"[{model_name}][eval-only] test  loss={test_metrics['loss']:.4f}, acc={test_metrics['acc']:.4f}")
        return {
            "model": model_name,
            "best_valid_acc": valid_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "per_class": per_class,
            "output_dir": str(output_dir),
        }

    for epoch in range(start_epoch, args.epochs + 1):
        # 标准训练循环：train -> valid -> 保存 last/best checkpoint。
        if distributed:
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            use_amp=args.amp,
            is_distributed=distributed,
            rank=rank,
            desc=f"{model_name}-train e{epoch}",
            max_batches=args.max_train_batches,
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            criterion,
            optimizer=None,
            device=device,
            use_amp=args.amp,
            is_distributed=distributed,
            rank=rank,
            desc=f"{model_name}-valid e{epoch}",
            max_batches=args.max_eval_batches,
        )

        if is_main_process(rank):
            print(
                f"[{model_name}] epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f} | "
                f"valid loss={valid_metrics['loss']:.4f}, acc={valid_metrics['acc']:.4f}"
            )

        ckpt = {
            "epoch": epoch,
            "model": (model.module if isinstance(model, DDP) else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_valid_acc": best_valid_acc,
            "no_improve_epochs": no_improve_epochs,
            "args": vars(args),
            "model_name": model_name,
        }
        if is_main_process(rank):
            torch.save(ckpt, output_dir / "last.pt")

        # 简化规则：验证集精度只要高于历史最好值，就视为有提升。
        improved = valid_metrics["acc"] > best_valid_acc
        if improved:
            best_valid_acc = valid_metrics["acc"]
            ckpt["best_valid_acc"] = best_valid_acc
            no_improve_epochs = 0
            ckpt["no_improve_epochs"] = no_improve_epochs
            if is_main_process(rank):
                torch.save(ckpt, output_dir / "best.pt")
        else:
            no_improve_epochs += 1

        distributed_barrier(distributed, device)

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            if is_main_process(rank):
                print(
                    f"[{model_name}] early stopping at epoch {epoch:03d} | "
                    f"best valid acc={best_valid_acc:.4f}, "
                    f"no_improve_epochs={no_improve_epochs}"
                )
            distributed_barrier(distributed, device)
            break

    best_ckpt = torch.load(output_dir / "best.pt", map_location="cpu")
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(best_ckpt["model"])
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        device=device,
        use_amp=args.amp,
        is_distributed=distributed,
        rank=rank,
        desc=f"{model_name}-test",
        max_batches=args.max_eval_batches,
    )
    per_class = evaluate_per_class(model, test_loader, device, is_distributed=distributed, max_batches=args.max_eval_batches)

    if is_main_process(rank):
        print(f"[{model_name}][final] best valid acc={best_valid_acc:.4f}")
        print(f"[{model_name}][final] test loss={test_metrics['loss']:.4f}, acc={test_metrics['acc']:.4f}")
    return {
        "model": model_name,
        "best_valid_acc": best_valid_acc,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "per_class": per_class,
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    distributed, rank, world_size, local_rank = init_distributed(args)

    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    if is_main_process(rank):
        output_root.mkdir(parents=True, exist_ok=True)
    model_list = parse_models(args.models, args.model)

    if len(model_list) > 1 and args.checkpoint:
        raise ValueError("When using multiple models, please do not pass --checkpoint.")

    train_tf, eval_tf = make_transforms(args.image_size)
    # 三个 split 保持固定划分，避免数据泄漏。
    train_ds = HazeLevelDataset(data_root, "train", train_tf)
    valid_ds = HazeLevelDataset(data_root, "valid", eval_tf)
    test_ds = HazeLevelDataset(data_root, "test", eval_tf)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    valid_sampler = DistributedSampler(valid_ds, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    if distributed and args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    # 依次运行多个模型并汇总结果。
    summary: List[Dict[str, object]] = []
    for model_name in model_list:
        result = run_for_model(
            args=args,
            model_name=model_name,
            data_root=data_root,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            train_size=len(train_ds),
            valid_size=len(valid_ds),
            test_size=len(test_ds),
            device=device,
            distributed=distributed,
            rank=rank,
        )
        if is_main_process(rank):
            summary.append(result)

    # 汇总结果：
    # - 单模型：summary.json 放到对应模型目录（与 best/last/run_meta 同目录）
    # - 多模型：summary.json 放到输出根目录，便于统一比较
    if is_main_process(rank):
        if len(model_list) == 1:
            summary_dir = output_root / model_list[0]
        else:
            summary_dir = output_root
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "summary.json"

        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[summary] saved to {summary_path}")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""YAML-driven haze generation for single-image validation and online dataloading."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    DataLoader = None
    Dataset = object


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def load_depth_image(depth_path: str) -> np.ndarray:
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Depth image not found or unreadable: {depth_path}")

    if d.ndim == 3:
        # Color depth map mode: hue-based conversion, larger means farther.
        hsv = cv2.cvtColor(d, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0].astype(np.float32)
        d = h / 179.0
        return normalize_to_01(d)

    d = d.astype(np.float32)
    if d.max() > 1.5:
        d = d / 255.0
    return normalize_to_01(d)


def parse_A(value: Any) -> np.ndarray:
    if isinstance(value, (int, float)):
        vals = [float(value)]
    elif isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        vals = [float(p) for p in parts]
    elif isinstance(value, Sequence):
        vals = [float(v) for v in value]
    else:
        raise ValueError(f"Unsupported A format: {type(value)}")

    if len(vals) == 0:
        raise ValueError("A is empty")
    if len(vals) == 1:
        vals = [vals[0], vals[0], vals[0]]
    if len(vals) != 3:
        raise ValueError("A must be one value or three values")

    arr = np.array(vals, dtype=np.float32)
    arr = np.where(arr <= 1.0, arr * 255.0, arr)
    return np.clip(arr, 0.0, 255.0)


def apply_tint(rgb_image: np.ndarray, A: np.ndarray, tint: float) -> np.ndarray:
    tint_strength = float(np.clip(tint, 0.0, 1.0))
    if tint_strength <= 0.0:
        return rgb_image
    a_reshaped = A.reshape((1, 1, 3)).astype(np.float32)
    out = rgb_image.astype(np.float32) * (1.0 - tint_strength) + a_reshaped * tint_strength
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def generate_haze(rgb_image: np.ndarray, depth_norm: np.ndarray, beta: float, A: np.ndarray) -> np.ndarray:
    depth_scaled = depth_norm * (10.0 - 0.1) + 0.1
    transmission = np.exp(-beta * depth_scaled)
    j_weighted = rgb_image.astype(np.float32) * transmission[..., None]
    a_weighted = A.reshape((1, 1, 3)).astype(np.float32) * (1.0 - transmission[..., None])
    return np.clip(j_weighted + a_weighted, 0, 255).astype(np.uint8)


def build_out_name(img_path: str, outdir: str, A: np.ndarray, beta: float, ext: str, tag: str = "") -> str:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    parent = os.path.basename(os.path.dirname(img_path)) or "root"
    a_vals = [float(x) for x in A.tolist()]
    if all(abs(x - round(x)) < 1e-6 for x in a_vals):
        a_str = "x".join(str(int(round(x))) for x in a_vals)
    else:
        a_str = "x".join(f"{x:.2f}" for x in a_vals)
    beta_str = f"{float(beta):.3f}"
    prefix = f"{tag}_" if tag else ""
    name = f"{prefix}{parent}_{img_name}_{a_str}_{beta_str}.{ext}"
    return os.path.join(outdir, name)


def resolve_path(config_path: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "haze" not in cfg or "combinations" not in cfg["haze"]:
        raise ValueError("Config must include haze.combinations")
    cfg["_config_path"] = str(path)
    return cfg


def parse_haze_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(config["haze"]["combinations"]):
        if "A" not in item or "beta" not in item:
            raise ValueError(f"haze.combinations[{idx}] requires A and beta")
        out.append(
            {
                "name": str(item.get("name", f"combo_{idx}")),
                "A": parse_A(item["A"]),
                "beta": float(item["beta"]),
                "tint": float(item.get("tint", 0.0)),
            }
        )
    return out


def _iter_files_by_ext(root: Path, exts: Sequence[str]) -> List[Path]:
    ext_set = {"." + e.lower().lstrip(".") for e in exts}
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ext_set:
            out.append(p)
    out.sort()
    return out


def match_image_depth_pairs(clean_dir: Path, depth_dir: Path, image_exts: Sequence[str]) -> List[Tuple[Path, Path]]:
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean dir not found: {clean_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth dir not found: {depth_dir}")

    clean_files = _iter_files_by_ext(clean_dir, image_exts)
    depth_files = _iter_files_by_ext(depth_dir, image_exts)

    depth_by_stem: Dict[str, List[Path]] = {}
    for dp in depth_files:
        depth_by_stem.setdefault(dp.stem, []).append(dp)

    pairs: List[Tuple[Path, Path]] = []
    for img_path in clean_files:
        rel = img_path.relative_to(clean_dir)
        rel_parent = rel.parent
        rel_stem = rel.stem

        candidates = []
        for ext in image_exts:
            candidate = depth_dir / rel_parent / f"{rel_stem}.{ext.lstrip('.')}"
            if candidate.exists():
                candidates.append(candidate)
        if len(candidates) == 1:
            pairs.append((img_path, candidates[0]))
            continue

        same_stem = depth_by_stem.get(img_path.stem, [])
        if len(same_stem) == 1:
            pairs.append((img_path, same_stem[0]))

    if not pairs:
        raise RuntimeError(f"No image-depth pairs found in {clean_dir} and {depth_dir}")
    return pairs


class HazeOnlineDataset(Dataset):
    """Online haze generation dataset from paired clean/depth images and haze combinations."""

    def __init__(
        self,
        image_depth_pairs: Sequence[Tuple[Path, Path]],
        haze_combinations: Sequence[Dict[str, Any]],
        random_combo: bool = True,
        resize_hw: Optional[Tuple[int, int]] = None,
        return_tensor: bool = True,
        seed: int = 123,
    ) -> None:
        self.image_depth_pairs = list(image_depth_pairs)
        self.haze_combinations = list(haze_combinations)
        self.random_combo = bool(random_combo)
        self.resize_hw = resize_hw
        self.return_tensor = bool(return_tensor)
        self.rng = random.Random(seed)

        if len(self.image_depth_pairs) == 0:
            raise ValueError("image_depth_pairs is empty")
        if len(self.haze_combinations) == 0:
            raise ValueError("haze_combinations is empty")
        if self.return_tensor and not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required when return_tensor=True")

    def __len__(self) -> int:
        if self.random_combo:
            return len(self.image_depth_pairs)
        return len(self.image_depth_pairs) * len(self.haze_combinations)

    def _choose_combo(self, idx: int) -> Dict[str, Any]:
        if self.random_combo:
            return self.haze_combinations[self.rng.randrange(len(self.haze_combinations))]
        combo_idx = idx // len(self.image_depth_pairs)
        return self.haze_combinations[combo_idx]

    def _choose_pair(self, idx: int) -> Tuple[Path, Path]:
        if self.random_combo:
            return self.image_depth_pairs[idx % len(self.image_depth_pairs)]
        img_idx = idx % len(self.image_depth_pairs)
        return self.image_depth_pairs[img_idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, depth_path = self._choose_pair(idx)
        combo = self._choose_combo(idx)

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = load_depth_image(str(depth_path))

        if self.resize_hw is not None:
            h, w = self.resize_hw
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        tinted = apply_tint(rgb, combo["A"], combo.get("tint", 0.0))
        haze = generate_haze(tinted, depth, combo["beta"], combo["A"])

        sample: Dict[str, Any] = {
            "hazy": haze,
            "clean": rgb,
            "depth": depth,
            "beta": float(combo["beta"]),
            "A": combo["A"].astype(np.float32),
            "combo_name": combo.get("name", ""),
            "img_path": str(img_path),
            "depth_path": str(depth_path),
        }

        if not self.return_tensor:
            return sample

        haze_tensor = torch.from_numpy(np.transpose(haze, (2, 0, 1))).float() / 255.0
        clean_tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float() / 255.0
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
        a_tensor = torch.from_numpy(sample["A"]).float()

        sample["hazy"] = haze_tensor
        sample["clean"] = clean_tensor
        sample["depth"] = depth_tensor
        sample["A"] = a_tensor
        return sample


def build_dataset_from_yaml(
    config_path: str,
    split: str = "train",
    return_tensor: bool = True,
    random_combo_override: Optional[bool] = None,
) -> HazeOnlineDataset:
    cfg = load_config(config_path)
    cfg_path = Path(cfg["_config_path"])  # already absolute
    split_cfg = cfg.get("dataset", {}).get(split)
    if split_cfg is None:
        raise ValueError(f"dataset.{split} is missing in config")

    clean_dir = resolve_path(cfg_path, split_cfg["clean_dir"])
    depth_dir = resolve_path(cfg_path, split_cfg["depth_dir"])
    image_exts = split_cfg.get("image_exts", ["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

    resize_hw = None
    resize_cfg = split_cfg.get("resize_hw")
    if resize_cfg is not None:
        if not isinstance(resize_cfg, Sequence) or len(resize_cfg) != 2:
            raise ValueError("resize_hw must be [height, width]")
        resize_hw = (int(resize_cfg[0]), int(resize_cfg[1]))

    combos = parse_haze_combinations(cfg)
    pairs = match_image_depth_pairs(clean_dir, depth_dir, image_exts)

    default_random_combo = bool(split_cfg.get("random_combo", True))
    random_combo = default_random_combo if random_combo_override is None else random_combo_override
    seed = int(cfg.get("seed", 123))
    return HazeOnlineDataset(
        image_depth_pairs=pairs,
        haze_combinations=combos,
        random_combo=random_combo,
        resize_hw=resize_hw,
        return_tensor=return_tensor,
        seed=seed,
    )


def build_haze_dataloader(
    config_path: str,
    split: str = "train",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    return_tensor: bool = True,
    list_collate: bool = True,
) -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to build DataLoader")

    dataset = build_dataset_from_yaml(
        config_path=config_path,
        split=split,
        return_tensor=return_tensor,
    )

    collate_fn = (lambda batch: batch) if list_collate else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def pick_combo(combos: Sequence[Dict[str, Any]], combo_name: str = "", combo_index: int = 0) -> Dict[str, Any]:
    if combo_name:
        for c in combos:
            if c.get("name") == combo_name:
                return c
        raise ValueError(f"combo_name not found: {combo_name}")
    if combo_index < 0 or combo_index >= len(combos):
        raise ValueError(f"combo_index out of range: {combo_index}")
    return combos[combo_index]


def run_single_from_local_dataloader(
    config_path: str,
    img_path: str,
    depth_path: str,
    combo_name: str = "",
    combo_index: int = 0,
    out_path: str = "",
) -> str:
    cfg = load_config(config_path)
    combos = parse_haze_combinations(cfg)
    combo = pick_combo(combos, combo_name=combo_name, combo_index=combo_index)

    dataset = HazeOnlineDataset(
        image_depth_pairs=[(Path(img_path).resolve(), Path(depth_path).resolve())],
        haze_combinations=[combo],
        random_combo=False,
        return_tensor=False,
        seed=int(cfg.get("seed", 123)),
    )
    sample = dataset[0]

    out_cfg = cfg.get("output", {})
    ext = str(out_cfg.get("ext", "jpg"))
    default_outdir = str(out_cfg.get("outdir", "./gen_haze_out"))
    cfg_path = Path(cfg["_config_path"])
    outdir = resolve_path(cfg_path, default_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if out_path:
        save_path = Path(out_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path(
            build_out_name(
                img_path=str(img_path),
                outdir=str(outdir),
                A=sample["A"],
                beta=sample["beta"],
                ext=ext,
                tag=str(sample.get("combo_name", "")),
            )
        )
    Image.fromarray(sample["hazy"]).save(str(save_path))
    return str(save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YAML-driven haze generation and online dataloader.")
    parser.add_argument(
        "--config",
        type=str,
        default=str((Path(__file__).resolve().parent / "configs" / "haze_config.yaml")),
        help="Path to YAML config",
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single", help="Generate one haze image via localized dataset workflow")
    p_single.add_argument("--img", type=str, required=True, help="Clean image path")
    p_single.add_argument("--depth", type=str, required=True, help="Depth image path")
    p_single.add_argument("--combo-name", type=str, default="", help="Combination name from haze.combinations")
    p_single.add_argument("--combo-index", type=int, default=0, help="Combination index if name is not provided")
    p_single.add_argument("--out", type=str, default="", help="Optional output file path")

    p_loader = sub.add_parser("dataloader", help="Build online dataloader and optionally save preview images")
    p_loader.add_argument("--split", type=str, default="train", help="Dataset split key under dataset.<split>")
    p_loader.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p_loader.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p_loader.add_argument("--max-batches", type=int, default=1, help="How many batches to iterate for quick check")
    p_loader.add_argument("--save-preview", action="store_true", help="Save first sample of each batch for visual validation")
    p_loader.add_argument("--preview-dir", type=str, default="./gen_haze_preview", help="Preview output directory")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "single":
        out_path = run_single_from_local_dataloader(
            config_path=args.config,
            img_path=args.img,
            depth_path=args.depth,
            combo_name=args.combo_name,
            combo_index=args.combo_index,
            out_path=args.out,
        )
        print(f"Saved: {out_path}")
        return

    if args.mode == "dataloader":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for dataloader mode")

        loader = build_haze_dataloader(
            config_path=args.config,
            split=args.split,
            batch_size=args.batch_size,
            shuffle=(args.split == "train"),
            num_workers=args.num_workers,
            return_tensor=False,
            list_collate=True,
        )

        preview_dir = Path(args.preview_dir).resolve()
        if args.save_preview:
            preview_dir.mkdir(parents=True, exist_ok=True)

        total = 0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            total += len(batch)
            if args.save_preview and len(batch) > 0:
                sample = batch[0]
                out_name = f"{args.split}_b{batch_idx:03d}_{Path(sample['img_path']).stem}.jpg"
                out_path = preview_dir / out_name
                Image.fromarray(sample["hazy"]).save(str(out_path))
                print(f"Preview saved: {out_path}")
            print(f"Batch {batch_idx}: size={len(batch)}")

        print(f"Iterated {total} samples from split={args.split}")
        return


if __name__ == "__main__":
    main()

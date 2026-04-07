#!/usr/bin/env python3
"""YAML-driven offline haze generation for single image and dataset builds."""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def load_depth_image(depth_path: str) -> np.ndarray:
    # Minimal assumption: depth files are grayscale images.
    d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if d is None:
        raise FileNotFoundError(f"Depth image not found or unreadable: {depth_path}")

    # Normalize first, then invert to match user's depth format:
    # black = far (large depth), white = near (small depth).
    depth_norm = normalize_to_01(d.astype(np.float32))
    depth_norm = 1.0 - depth_norm

    # Smooth depth to suppress local spikes that can create overly heavy distant haze.
    depth_norm = cv2.GaussianBlur(depth_norm, (5, 5), sigmaX=1.0)
    return np.clip(depth_norm, 0.0, 1.0)


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
    if "haze" not in cfg:
        raise ValueError("Config must include haze section")
    cfg["_config_path"] = str(path)
    return cfg


def parse_haze_param_grid(config: Dict[str, Any]) -> Dict[str, Any]:
    haze_cfg = config.get("haze", {})
    a_values_raw = haze_cfg.get("A_values")
    beta_values_raw = haze_cfg.get("beta_values")
    tint = float(haze_cfg.get("tint", 0.0))

    if a_values_raw is None or beta_values_raw is None:
        raise ValueError("Config must provide haze.A_values and haze.beta_values")

    a_values = [parse_A(a) for a in a_values_raw]
    beta_values = [float(b) for b in beta_values_raw]

    if len(a_values) == 0:
        raise ValueError("haze.A_values is empty")
    if len(beta_values) == 0:
        raise ValueError("haze.beta_values is empty")

    return {
        "A_values": a_values,
        "beta_values": beta_values,
        "tint": tint,
    }


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


def pick_params(
    a_values: Sequence[np.ndarray],
    beta_values: Sequence[float],
    a_index: int = 0,
    beta_index: int = 0,
) -> Tuple[np.ndarray, float]:
    if a_index < 0 or a_index >= len(a_values):
        raise ValueError(f"a_index out of range: {a_index}")
    if beta_index < 0 or beta_index >= len(beta_values):
        raise ValueError(f"beta_index out of range: {beta_index}")
    return a_values[a_index], float(beta_values[beta_index])


def _load_rgb_and_depth(
    img_path: Path,
    depth_path: Path,
    resize_hw: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = load_depth_image(str(depth_path))

    if resize_hw is not None:
        h, w = resize_hw
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    return rgb, depth


def _pick_random_a_indices(pair_idx: int, num_a: int, seed: int, pick_count: int = 2) -> List[int]:
    # Deterministic random per pair for reproducible offline generation.
    if num_a < pick_count:
        raise ValueError(f"Need at least {pick_count} A values, but got {num_a}")
    rng = random.Random(seed + pair_idx * 1000003)
    return sorted(rng.sample(range(num_a), k=pick_count))


def _write_metadata_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "output_path",
                "clean_path",
                "depth_path",
                "a_idx",
                "beta_idx",
                "A",
                "beta",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_single_from_local_dataloader(
    config_path: str,
    img_path: str,
    depth_path: str,
    a_index: int = 0,
    beta_index: int = 0,
    out_path: str = "",
) -> str:
    cfg = load_config(config_path)
    param_grid = parse_haze_param_grid(cfg)
    A, beta = pick_params(
        a_values=param_grid["A_values"],
        beta_values=param_grid["beta_values"],
        a_index=a_index,
        beta_index=beta_index,
    )

    img_p = Path(img_path).resolve()
    depth_p = Path(depth_path).resolve()
    rgb, depth = _load_rgb_and_depth(img_p, depth_p, resize_hw=None)
    tinted = apply_tint(rgb, A, param_grid["tint"])
    haze = generate_haze(tinted, depth, beta, A)

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
                A=A,
                beta=beta,
                ext=ext,
                tag=f"a{a_index}_b{beta_index}",
            )
        )
    Image.fromarray(haze).save(str(save_path))
    return str(save_path)


def build_split_offline(
    config_path: str,
    split: str,
    overwrite: bool = False,
    max_pairs: int = 0,
    seed_override: Optional[int] = None,
) -> Dict[str, Any]:
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

    param_grid = parse_haze_param_grid(cfg)
    a_values = param_grid["A_values"]
    beta_values = param_grid["beta_values"]
    tint = param_grid["tint"]
    seed = int(cfg.get("seed", 123)) if seed_override is None else int(seed_override)

    pairs = match_image_depth_pairs(clean_dir, depth_dir, image_exts)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    out_cfg = cfg.get("output", {})
    ext = str(out_cfg.get("ext", "jpg")).lstrip(".")

    # New default layout: write haze data into each split folder under datasets/<split>/haze_images.
    split_root = clean_dir.parent
    hazy_root = split_root / "haze_images"
    meta_path = split_root / "haze_metadata.csv"

    if overwrite and hazy_root.exists():
        shutil.rmtree(hazy_root)
    if overwrite and meta_path.exists():
        meta_path.unlink()
    hazy_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    total_saved = 0
    num_a = len(a_values)
    num_b = len(beta_values)

    if num_a < 2:
        raise ValueError("haze.A_values must contain at least 2 entries for current generation policy")

    for pair_idx, (img_path, depth_path) in enumerate(pairs):
        rel = img_path.relative_to(clean_dir)
        stem = rel.stem
        out_dir = hazy_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        rgb, depth = _load_rgb_and_depth(img_path, depth_path, resize_hw)

        chosen_a_indices = _pick_random_a_indices(pair_idx, num_a, seed, pick_count=2)
        for a_idx in chosen_a_indices:
            A = a_values[a_idx]
            for b_idx, beta in enumerate(beta_values):
                tinted = apply_tint(rgb, A, tint)
                haze = generate_haze(tinted, depth, beta, A)

                # Naming rule: {original_name}_{A_index}_{beta_index}
                out_path = out_dir / f"{stem}_{a_idx}_{b_idx}.{ext}"
                Image.fromarray(haze).save(str(out_path))
                total_saved += 1

                rows.append(
                    {
                        "split": split,
                        "output_path": str(out_path),
                        "clean_path": str(img_path),
                        "depth_path": str(depth_path),
                        "a_idx": a_idx,
                        "beta_idx": b_idx,
                        "A": "|".join(str(int(round(v))) for v in A.tolist()),
                        "beta": float(beta),
                    }
                )

    _write_metadata_csv(meta_path, rows)
    return {
        "split": split,
        "pairs": len(pairs),
        "images_saved": total_saved,
        "out_dir": str(hazy_root),
        "metadata": str(meta_path),
        "a_per_image": 2,
        "betas_per_a": num_b,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YAML-driven offline haze generation.")
    parser.add_argument(
        "--config",
        type=str,
        default=str((Path(__file__).resolve().parent / "configs" / "haze_config.yaml")),
        help="Path to YAML config",
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single", help="Generate one haze image")
    p_single.add_argument("--img", type=str, required=True, help="Clean image path")
    p_single.add_argument("--depth", type=str, required=True, help="Depth image path")
    p_single.add_argument("--a-index", type=int, default=0, help="Index in haze.A_values")
    p_single.add_argument("--beta-index", type=int, default=0, help="Index in haze.beta_values")
    p_single.add_argument("--out", type=str, default="", help="Optional output file path")

    p_build = sub.add_parser("build", help="Offline-generate haze images for a split or all splits")
    p_build.add_argument("--split", type=str, default="all", help="dataset split key, or 'all'")
    p_build.add_argument("--overwrite", action="store_true", help="Delete output split folder before writing")
    p_build.add_argument("--max-pairs", type=int, default=0, help="Limit number of clean-depth pairs per split (0 means all)")
    p_build.add_argument("--seed", type=int, default=None, help="Optional seed override for random_combo mode")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "single":
        out_path = run_single_from_local_dataloader(
            config_path=args.config,
            img_path=args.img,
            depth_path=args.depth,
            a_index=args.a_index,
            beta_index=args.beta_index,
            out_path=args.out,
        )
        print(f"Saved: {out_path}")
        return

    if args.mode == "build":
        cfg = load_config(args.config)
        dataset_cfg = cfg.get("dataset", {})
        if args.split == "all":
            # Process three canonical split folders to match current dataset layout.
            target_splits = [s for s in ("train", "valid", "test") if s in dataset_cfg]
        else:
            target_splits = [args.split]

        if not target_splits:
            raise ValueError("No dataset splits found in config")

        for split in target_splits:
            summary = build_split_offline(
                config_path=args.config,
                split=split,
                overwrite=args.overwrite,
                max_pairs=args.max_pairs,
                seed_override=args.seed,
            )
            print(
                f"[{summary['split']}] pairs={summary['pairs']}, "
                f"saved={summary['images_saved']}, "
                f"a_per_image={summary['a_per_image']}, betas_per_a={summary['betas_per_a']}"
            )
            print(f"  out_dir: {summary['out_dir']}")
            print(f"  metadata: {summary['metadata']}")
        return


if __name__ == "__main__":
    main()

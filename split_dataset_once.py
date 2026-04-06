#!/usr/bin/env python3
"""One-off splitter for paired original/depth images into train/valid/test (3:1:1)."""

from pathlib import Path
import random
import shutil


# Hardcoded paths and behavior for one-time use.
ROOT = Path("datasets")
ORIGINAL_DIR = ROOT / "original_images"
DEPTH_DIR = ROOT / "depth_images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")
USE_MOVE = False  # True: move files, False: copy files
RANDOM_SEED = 42  # Set int (e.g. 123) for reproducible shuffle.


def collect_by_stem(folder: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            mapping[path.stem] = path
    return mapping


def prepare_output_dirs() -> None:
    for split in SPLITS:
        split_dir = ROOT / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "original_images").mkdir(parents=True, exist_ok=True)
        (split_dir / "depth_images").mkdir(parents=True, exist_ok=True)


def transfer(src: Path, dst: Path) -> None:
    if USE_MOVE:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def main() -> None:
    if not ORIGINAL_DIR.exists() or not DEPTH_DIR.exists():
        raise FileNotFoundError(f"Missing source dirs: {ORIGINAL_DIR} or {DEPTH_DIR}")

    original_map = collect_by_stem(ORIGINAL_DIR)
    depth_map = collect_by_stem(DEPTH_DIR)

    # Pair strictly by same filename stem.
    shared = sorted(set(original_map) & set(depth_map))
    if not shared:
        raise RuntimeError("No matched pairs by filename stem.")

    # Randomize pair order, but keep original/depth pairing intact.
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(shared)

    only_original = len(set(original_map) - set(depth_map))
    only_depth = len(set(depth_map) - set(original_map))
    if only_original:
        print(f"[WARN] only in original_images: {only_original}")
    if only_depth:
        print(f"[WARN] only in depth_images: {only_depth}")

    total = len(shared)
    n_train = total * 3 // 5
    n_valid = total * 1 // 5

    split_map = {
        "train": shared[:n_train],
        "valid": shared[n_train : n_train + n_valid],
        "test": shared[n_train + n_valid :],
    }

    prepare_output_dirs()

    for split, stems in split_map.items():
        out_orig = ROOT / split / "original_images"
        out_depth = ROOT / split / "depth_images"
        for stem in stems:
            src_orig = original_map[stem]
            src_depth = depth_map[stem]
            transfer(src_orig, out_orig / src_orig.name)
            transfer(src_depth, out_depth / src_depth.name)

    print("[DONE] split completed")
    print(f"total pairs: {total}")
    for split in SPLITS:
        cnt = len(split_map[split])
        print(f"{split}: {cnt} ({cnt / total:.1%})")
    print(f"mode: {'move' if USE_MOVE else 'copy'}")


if __name__ == "__main__":
    main()

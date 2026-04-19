import csv
import os
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from util.crop import center_crop_arr


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PairedImageDataset(Dataset):
    """Paired dataset for image restoration tasks.

    It supports two pairing modes:
    1) stem: pair by file stem with fallback for names ending with
       "_<beta_idx>_<a_idx>".
    2) meta: explicit pairing from text/csv with columns:
       lq_rel_path, gt_rel_path[, label].
    """

    def __init__(
        self,
        root,
        split="train",
        lq_dirname="haze_images",
        gt_dirname="original_images",
        img_size=256,
        pairing_mode="stem",
        pair_meta=None,
        random_flip=True,
    ):
        self.root = Path(root)
        self.split = split
        self.lq_root = self.root / split / lq_dirname
        self.gt_root = self.root / split / gt_dirname
        self.img_size = img_size
        self.pairing_mode = pairing_mode
        self.pair_meta = Path(pair_meta) if pair_meta else None
        self.random_flip = random_flip

        if not self.lq_root.exists():
            raise FileNotFoundError(f"LQ directory not found: {self.lq_root}")
        if not self.gt_root.exists():
            raise FileNotFoundError(f"GT directory not found: {self.gt_root}")

        self.to_tensor = transforms.PILToTensor()

        if self.pairing_mode == "meta":
            if self.pair_meta is None:
                raise ValueError("pair_meta must be provided when pairing_mode='meta'")
            self.samples = self._build_pairs_by_meta(self.pair_meta)
        elif self.pairing_mode == "stem":
            self.samples = self._build_pairs_by_stem()
        else:
            raise ValueError(f"Unsupported pairing_mode: {self.pairing_mode}")

        if not self.samples:
            raise RuntimeError("No paired samples were found.")

    def _iter_image_files(self, root):
        return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS])

    def _stem_candidates(self, stem):
        candidates = [stem]
        parts = stem.rsplit("_", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            candidates.append(parts[0])
        return candidates

    def _build_pairs_by_stem(self):
        gt_files = self._iter_image_files(self.gt_root)
        gt_map = {p.stem: p for p in gt_files}

        pairs = []
        missed = 0
        for lq_path in self._iter_image_files(self.lq_root):
            gt_path = None
            for key in self._stem_candidates(lq_path.stem):
                if key in gt_map:
                    gt_path = gt_map[key]
                    break
            if gt_path is None:
                missed += 1
                continue
            pairs.append((lq_path, gt_path, 0))

        if not pairs:
            raise RuntimeError(
                f"No stem-matched pairs found between {self.lq_root} and {self.gt_root}."
            )
        if missed > 0:
            print(f"[PairedImageDataset] Warning: {missed} LQ files have no GT match.")

        return pairs

    def _parse_meta_line(self, row):
        if len(row) < 2:
            return None
        lq_rel = row[0].strip()
        gt_rel = row[1].strip()
        label = int(row[2]) if len(row) >= 3 and row[2].strip() != "" else 0
        return lq_rel, gt_rel, label

    def _build_pairs_by_meta(self, meta_path):
        if not meta_path.exists():
            raise FileNotFoundError(f"pair_meta not found: {meta_path}")

        pairs = []
        with open(meta_path, "r", encoding="utf-8") as f:
            content = f.read().strip().splitlines()

        for raw in content:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue

            row = next(csv.reader([raw], delimiter=","))
            if len(row) < 2:
                row = raw.split()

            parsed = self._parse_meta_line(row)
            if parsed is None:
                continue

            lq_rel, gt_rel, label = parsed
            lq_path = (self.lq_root / lq_rel) if not os.path.isabs(lq_rel) else Path(lq_rel)
            gt_path = (self.gt_root / gt_rel) if not os.path.isabs(gt_rel) else Path(gt_rel)

            if not lq_path.exists() or not gt_path.exists():
                continue
            pairs.append((lq_path, gt_path, label))

        if not pairs:
            raise RuntimeError(f"No valid pairs found in meta file: {meta_path}")
        return pairs

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        with Image.open(path) as img:
            return img.convert("RGB")

    def __getitem__(self, idx):
        lq_path, gt_path, label = self.samples[idx]

        lq_img = center_crop_arr(self._load_rgb(lq_path), self.img_size)
        gt_img = center_crop_arr(self._load_rgb(gt_path), self.img_size)

        if self.random_flip and random.random() < 0.5:
            lq_img = lq_img.transpose(Image.FLIP_LEFT_RIGHT)
            gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)

        lq = self.to_tensor(lq_img)
        gt = self.to_tensor(gt_img)
        label = torch.tensor(label, dtype=torch.long)

        return lq, gt, label
import csv
import os
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from util.crop import center_crop_arr


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PairedImageDataset(Dataset):
    """Paired dataset for image restoration tasks.

    It supports two pairing modes:
    1) stem: pair by file stem with a lightweight fallback for names that end
       with "_<beta_idx>_<a_idx>".
    2) meta: explicit pairing from a text/csv file containing 2 or 3 columns:
       lq_rel_path, gt_rel_path[, label].
    """

    def __init__(
        self,
        root,
        split="train",
        lq_dirname="haze_images",
        gt_dirname="original_images",
        img_size=256,
        pairing_mode="stem",
        pair_meta=None,
        random_flip=True,
    ):
        self.root = Path(root)
        self.split = split
        self.lq_root = self.root / split / lq_dirname
        self.gt_root = self.root / split / gt_dirname
        self.img_size = img_size
        self.pairing_mode = pairing_mode
        self.pair_meta = Path(pair_meta) if pair_meta else None
        self.random_flip = random_flip

        if not self.lq_root.exists():
            raise FileNotFoundError(f"LQ directory not found: {self.lq_root}")
        if not self.gt_root.exists():
            raise FileNotFoundError(f"GT directory not found: {self.gt_root}")

        self.to_tensor = transforms.PILToTensor()

        if self.pairing_mode == "meta":
            if self.pair_meta is None:
                raise ValueError("pair_meta must be provided when pairing_mode='meta'")
            self.samples = self._build_pairs_by_meta(self.pair_meta)
        elif self.pairing_mode == "stem":
            self.samples = self._build_pairs_by_stem()
        else:
            raise ValueError(f"Unsupported pairing_mode: {self.pairing_mode}")

        if not self.samples:
            raise RuntimeError("No paired samples were found.")

    def _iter_image_files(self, root):
        return sorted(
            [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
        )

    def _stem_candidates(self, stem):
        candidates = [stem]
        parts = stem.rsplit("_", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            candidates.append(parts[0])
        return candidates

    def _build_pairs_by_stem(self):
        gt_files = self._iter_image_files(self.gt_root)
        gt_map = {p.stem: p for p in gt_files}

        pairs = []
        missed = 0
        for lq_path in self._iter_image_files(self.lq_root):
            gt_path = None
            for key in self._stem_candidates(lq_path.stem):
                if key in gt_map:
                    gt_path = gt_map[key]
                    break
            if gt_path is None:
                missed += 1
                continue
            pairs.append((lq_path, gt_path, 0))

        if not pairs:
            raise RuntimeError(
                f"No stem-matched pairs found between {self.lq_root} and {self.gt_root}."
            )
        if missed > 0:
            print(f"[PairedImageDataset] Warning: {missed} LQ files have no GT match.")

        return pairs

    def _parse_meta_line(self, row):
        if len(row) < 2:
            return None
        lq_rel = row[0].strip()
        gt_rel = row[1].strip()
        label = int(row[2]) if len(row) >= 3 and row[2].strip() != "" else 0
        return lq_rel, gt_rel, label

    def _build_pairs_by_meta(self, meta_path):
        if not meta_path.exists():
            raise FileNotFoundError(f"pair_meta not found: {meta_path}")

        pairs = []
        with open(meta_path, "r", encoding="utf-8") as f:
            content = f.read().strip().splitlines()

        for raw in content:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue

            row = next(csv.reader([raw], delimiter=","))
            if len(row) < 2:
                row = raw.split()

            parsed = self._parse_meta_line(row)
            if parsed is None:
                continue

            lq_rel, gt_rel, label = parsed
            lq_path = (self.lq_root / lq_rel) if not os.path.isabs(lq_rel) else Path(lq_rel)
            gt_path = (self.gt_root / gt_rel) if not os.path.isabs(gt_rel) else Path(gt_rel)

            if not lq_path.exists() or not gt_path.exists():
                continue
            pairs.append((lq_path, gt_path, label))

        if not pairs:
            raise RuntimeError(f"No valid pairs found in meta file: {meta_path}")
        return pairs

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        with Image.open(path) as img:
            return img.convert("RGB")

    def __getitem__(self, idx):
        lq_path, gt_path, label = self.samples[idx]

        lq_img = center_crop_arr(self._load_rgb(lq_path), self.img_size)
        gt_img = center_crop_arr(self._load_rgb(gt_path), self.img_size)

        if self.random_flip and random.random() < 0.5:
            lq_img = lq_img.transpose(Image.FLIP_LEFT_RIGHT)
            gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)

        lq = self.to_tensor(lq_img)
        gt = self.to_tensor(gt_img)
        label = torch.tensor(label, dtype=torch.long)

        return lq, gt, label

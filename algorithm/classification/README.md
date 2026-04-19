Haze Level Classifier (Swin-T)
==============================

This script trains a 10-class haze concentration classifier from generated haze images.

Label rule
----------

Image names must end with:

{original_name}_{A_index}_{beta_index}.jpg

`beta_index` in `[0, 9]` is used as class label.

Expected data layout
--------------------

- `datasets/train/haze_images`
- `datasets/valid/haze_images`
- `datasets/test/haze_images`

Quick run
---------

```bash
.venv/bin/python Swin_Transformer/train_haze_classifier.py \
  --data-root datasets \
  --output-dir Swin_Transformer/outputs_haze_cls \
  --epochs 10 \
  --batch-size 64 \
  --num-workers 4 \
  --pretrained \
  --amp
```

Fast smoke test
---------------

```bash
.venv/bin/python Swin_Transformer/train_haze_classifier.py \
  --data-root datasets \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --max-train-batches 2 \
  --max-eval-batches 2
```

Evaluate checkpoint
-------------------

```bash
.venv/bin/python Swin_Transformer/train_haze_classifier.py \
  --data-root datasets \
  --eval-only \
  --checkpoint Swin_Transformer/outputs_haze_cls/best.pt
```

Haze Level Classification (Multi-Backbone)
==========================================

This module trains a 10-class haze concentration classifier using torchvision backbones.

Supported Models
----------------

- `resnet18`
- `resnet50`
- `mobilenet_v3_small`
- `efficientnet_b0`
- `swin_t`
- `vit_b_16`

Label Rule
----------

Image names must end with:

`{original_name}_{A_index}_{beta_index}.jpg`

`beta_index` in `[0, 9]` is used as the class label.

Expected Data Layout
--------------------

- `datasets/train/haze_images`
- `datasets/valid/haze_images`
- `datasets/test/haze_images`

Output Directory
----------------

All results are unified under:

- `result/classification/<model_name>/best.pt`
- `result/classification/<model_name>/last.pt`
- `result/classification/<model_name>/run_meta.json`
- `result/classification/summary.json`

Train Single Model
------------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --model resnet50 \
  --data-root datasets \
  --output-dir result/classification \
  --epochs 10 \
  --batch-size 64 \
  --num-workers 4 \
  --pretrained \
  --amp
```

Train Multiple Models in One Run
--------------------------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --models resnet18,resnet50,swin_t,vit_b_16 \
  --data-root datasets \
  --output-dir result/classification \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 4 \
  --pretrained \
  --amp
```

Fast Smoke Test
---------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --models resnet18,swin_t \
  --data-root datasets \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --max-train-batches 2 \
  --max-eval-batches 2
```

Evaluate Checkpoint
-------------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --model resnet18 \
  --data-root datasets \
  --eval-only \
  --checkpoint result/classification/resnet18/best.pt
```

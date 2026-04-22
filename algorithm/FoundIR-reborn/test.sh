#!/usr/bin/env bash

# Example FoundIR inference command.
# Replace DATA_ROOT, META_PATH and CHECKPOINT_DIR with your own paths.

python test.py \
  --dataroot /path/to/dataset_root \
  --meta /path/to/test_meta.txt \
  --dataset_mode meta_info \
  --checkpoint_dir ./ckpt_single_multi \
  --checkpoint_milestone 2000 \
  --output_dir ./results \
  --image_size 1024 \
  --sampling_timesteps 4 \
  --crop_phase im2overlap \
  --crop_stride 512

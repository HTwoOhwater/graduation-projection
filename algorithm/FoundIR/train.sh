#!/usr/bin/env bash

# Example FoundIR training command.
# Replace DATA_ROOT and META_PATH with your own dataset paths.

python train.py \
  --dataroot /path/to/dataset_root \
  --meta /path/to/train_meta.txt \
  --dataset_mode meta_info \
  --phase train \
  --batch_size 8 \
  --image_size 512 \
  --crop_size 256 \
  --sampling_timesteps 10 \
  --train_num_steps 500000 \
  --results_folder ./ckpt_single_multi

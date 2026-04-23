#!/usr/bin/env bash

# FoundIR-reborn training examples.
# 单卡可直接用 python。
# 多卡建议用 accelerate launch，不要再用 split_batches 去强行切一个固定总 batch。

# Single GPU
python train.py \
  --dataroot /mnt/workspace/Dehaze/datasets \
  --dataset_mode split_stem \
  --split train \
  --lq_dirname haze_images \
  --gt_dirname original_images \
  --batch_size 4 \
  --image_size 512 \
  --crop_size 256 \
  --sampling_timesteps 10 \
  --train_num_steps 500000 \
  --gradient_accumulate_every 2 \
  --mixed_precision fp16 \
  --num_workers 4 \
  --output_dir /mnt/workspace/Dehaze/result/foundir_reborn_train

# Multi GPU with accelerate
# accelerate launch --num_processes 2 train.py \
#   --dataroot /mnt/workspace/Dehaze/datasets \
#   --dataset_mode split_stem \
#   --split train \
#   --lq_dirname haze_images \
#   --gt_dirname original_images \
#   --batch_size 4 \
#   --image_size 512 \
#   --crop_size 256 \
#   --sampling_timesteps 10 \
#   --train_num_steps 500000 \
#   --gradient_accumulate_every 2 \
#   --mixed_precision fp16 \
#   --num_workers 2 \
#   --output_dir /mnt/workspace/Dehaze/result/foundir_reborn_train

# 2026.4.7
初始化了仓库，把之前的内容总结一下：
构建数据：
图像分类：简单使用


# 2026.4.22
cd /mnt/workspace/Dehaze/algorithm/FoundIR-reborn

CUDA_VISIBLE_DEVICES=0 /mnt/workspace/Dehaze/.venv/bin/python train.py \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --dataset_mode split_stem \
    --split train \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --batch_size 4 \
    --image_size 512 \
    --crop_size 256 \
    --sampling_timesteps 10 \
    --train_num_steps 30000 \
    --save_and_sample_every 5000 \
    --gradient_accumulate_every 2 \
    --optimizer adamw \
    --train_lr 1e-4 \
    --weight_decay 1e-4 \
    --mixed_precision fp16 \
    --num_workers 4 \
    --output_dir /mnt/workspace/Dehaze/result/foundir_reborn_longrun_30k \
    --resume_checkpoint /mnt/workspace/Dehaze/result/foundir_reborn_finetune_clean5000/model-last.pt


cd /mnt/workspace/Dehaze/algorithm/FFA-Net/net

CUDA_VISIBLE_DEVICES=1 /mnt/workspace/Dehaze/.venv/bin/python main.py \
    --net ffa \
    --trainset dehaze_train \
    --testset dehaze_test \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --train_split train \
    --test_split valid \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --crop \
    --crop_size 240 \
    --bs 4 \
    --steps 100000 \
    --eval_step 5000 \
    --eval_limit 100 \
    --eval_print_freq 20 \
    --gps 3 \
    --blocks 19 \
    --lr 1e-4 \
    --num_workers 0 \
    --output_dir /mnt/workspace/Dehaze/result/ffa_transfer_from_its \
    --pretrained_checkpoint /mnt/workspace/Dehaze/algorithm/FFA-Net/its_train_ffa_3_19.pk

cd /mnt/workspace/Dehaze/algorithm/FoundIR-reborn # 已经发散

CUDA_VISIBLE_DEVICES=2 /mnt/workspace/Dehaze/.venv/bin/python train.py \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --dataset_mode split_stem \
    --split train \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --batch_size 8 \
    --image_size 512 \
    --crop_size 256 \
    --sampling_timesteps 10 \
    --train_num_steps 30000 \
    --save_and_sample_every 5000 \
    --gradient_accumulate_every 1 \
    --optimizer adamw \
    --train_lr 5e-4 \
    --weight_decay 1e-4 \
    --mixed_precision fp16 \
    --num_workers 4 \
    --output_dir /mnt/workspace/Dehaze/result/foundir_reborn_longrun_30k_high_lr \
    --resume_checkpoint /mnt/workspace/Dehaze/result/foundir_reborn_finetune_clean5000/model-last.pt


cd /mnt/workspace/Dehaze/algorithm/FFA-Net/net

CUDA_VISIBLE_DEVICES=3 /mnt/workspace/Dehaze/.venv/bin/python main.py \
    --net ffa \
    --trainset dehaze_train \
    --testset dehaze_test \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --train_split train \
    --test_split valid \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --crop \
    --crop_size 240 \
    --bs 4 \
    --steps 30000 \
    --eval_step 5000 \
    --eval_limit 100 \
    --eval_print_freq 20 \
    --gps 3 \
    --blocks 19 \
    --lr 5e-4 \
    --num_workers 0 \
    --output_dir /mnt/workspace/Dehaze/result/ffa_transfer_from_its_high_lr \
    --pretrained_checkpoint /mnt/workspace/Dehaze/algorithm/FFA-Net/its_train_ffa_3_19.pk

推荐补充：

如果想改用 OTS 预训练，把最后一行替换为：
    --pretrained_checkpoint /mnt/workspace/Dehaze/algorithm/FFA-Net/ots_train_ffa_3_19.pk

如果想从本地训练结果继续续训，使用：
    --resume_checkpoint /mnt/workspace/Dehaze/result/ffa_transfer_from_its/model-last.pt

于2026.4.22 17：28写下：学习率1e-3对于FFANet较高，损失飘忽不定，很不稳定，在写下这段文字的过程中，即便已经到5e-4了，损失还是很夸张（初始1e-4）。

agent 补充：
- FFANet 当前本地版本建议优先用 `--pretrained_checkpoint` 做迁移训练，不要直接混用旧 checkpoint 状态。
- `num_workers` 在当前 Python 3.12 环境下建议先设为 `0`，否则容易出现 `/tmp/pymp-*` 清理报错。
- 若图片某一边小于 `crop_size`，当前本地 dataloader 已经会先放大再裁剪，不会再直接因为 `RandomCrop` 崩掉。
- FFANet 默认 `5000` 步会触发一次验证；如果 `valid` 很大，建议加 `--eval_limit 100` 之类限制验证规模，否则看起来会像“卡死”。
- 当前本地版本会在验证时按 `--eval_print_freq` 输出进度，便于判断是在评估而不是卡住。

不太妙，这个学习率发散了.
小batch多积累，效率更高，比纯batch大要快


# 2026.4.23
今天的训练计划：

先重新训练 FFANet。原因是训练循环已经修正，之前每一步都在重新 `next(iter(loader_train))`，训练行为不正常，旧结果参考价值有限。

FFANet 重新微调，优先从 ITS 预训练迁移，学习率降到 `1e-5`：

cd /mnt/workspace/Dehaze/algorithm/FFA-Net/net

CUDA_VISIBLE_DEVICES=1 /mnt/workspace/Dehaze/.venv/bin/python main.py \
    --net ffa \
    --trainset dehaze_train \
    --testset dehaze_test \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --train_split train \
    --test_split valid \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --crop \
    --crop_size 240 \
    --bs 4 \
    --steps 30000 \
    --eval_step 5000 \
    --eval_limit 100 \
    --eval_print_freq 20 \
    --gps 3 \
    --blocks 19 \
    --lr 1e-5 \
    --num_workers 0 \
    --output_dir /mnt/workspace/Dehaze/result/ffa_transfer_from_its_lr1e5_fixedloop \
    --pretrained_checkpoint /mnt/workspace/Dehaze/algorithm/FFA-Net/its_train_ffa_3_19.pk

如果想用 OTS 预训练，把最后一行替换为：
    --pretrained_checkpoint /mnt/workspace/Dehaze/algorithm/FFA-Net/ots_train_ffa_3_19.pk

如果 `1e-5` 太保守，可以再试一条 `2e-5` 对照：

cd /mnt/workspace/Dehaze/algorithm/FFA-Net/net

CUDA_VISIBLE_DEVICES=3 /mnt/workspace/Dehaze/.venv/bin/python main.py \
    --net ffa \
    --trainset dehaze_train \
    --testset dehaze_test \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --train_split train \
    --test_split valid \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --crop \
    --crop_size 240 \
    --bs 4 \
    --steps 30000 \
    --eval_step 5000 \
    --eval_limit 100 \
    --eval_print_freq 20 \
    --gps 3 \
    --blocks 19 \
    --lr 2e-5 \
    --num_workers 0 \
    --output_dir /mnt/workspace/Dehaze/result/ffa_transfer_from_its_lr2e5_fixedloop \
    --pretrained_checkpoint /mnt/workspace/Dehaze/algorithm/FFA-Net/its_train_ffa_3_19.pk

FoundIR 这边先保持稳态续训，作为主线继续观察：

cd /mnt/workspace/Dehaze/algorithm/FoundIR-reborn

CUDA_VISIBLE_DEVICES=0 /mnt/workspace/Dehaze/.venv/bin/python train.py \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --dataset_mode split_stem \
    --split train \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --batch_size 4 \
    --image_size 512 \
    --crop_size 256 \
    --sampling_timesteps 10 \
    --train_num_steps 30000 \
    --save_and_sample_every 5000 \
    --gradient_accumulate_every 2 \
    --optimizer adamw \
    --train_lr 1e-4 \
    --weight_decay 1e-4 \
    --mixed_precision fp16 \
    --num_workers 4 \
    --output_dir /mnt/workspace/Dehaze/result/foundir_reborn_longrun_30k_continue \
    --resume_checkpoint /mnt/workspace/Dehaze/result/foundir_reborn_longrun_30k/model-last.pt


# 2026.4.24
今晚 FoundIR-reborn 类别损失实验：

基础说明：
- 当前版本已经支持 `haze_level_label + airlight_label` 双头分类辅助监督
- 推理逻辑不变，只在训练时增加类别损失
- 这轮实验主要比较 `class_loss_weight` 的比例

统一设置：
- 基础权重：`/mnt/workspace/Dehaze/result/foundir_reborn_longrun_30k_continue/model-last.pt`
- 类别数：`haze=10`, `airlight=6`
- 训练步数：`30000`
- 优化器：`adamw`

GPU0：`class_loss_weight = 0.02`

cd /mnt/workspace/Dehaze/algorithm/FoundIR-reborn

CUDA_VISIBLE_DEVICES=0 /mnt/workspace/Dehaze/.venv/bin/python train.py \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --dataset_mode split_stem \
    --split train \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --batch_size 4 \
    --image_size 512 \
    --crop_size 256 \
    --sampling_timesteps 10 \
    --train_num_steps 30000 \
    --save_and_sample_every 5000 \
    --gradient_accumulate_every 2 \
    --optimizer adamw \
    --train_lr 1e-4 \
    --weight_decay 1e-4 \
    --mixed_precision fp16 \
    --num_workers 4 \
    --use_class_loss \
    --class_loss_weight 0.02 \
    --num_haze_classes 10 \
    --num_airlight_classes 6 \
    --output_dir /mnt/workspace/Dehaze/result/foundir_reborn_classloss_w002 \
    --resume_checkpoint /mnt/workspace/Dehaze/result/foundir_reborn_longrun_30k_continue/model-last.pt

GPU1：`class_loss_weight = 0.05`

cd /mnt/workspace/Dehaze/algorithm/FoundIR-reborn

CUDA_VISIBLE_DEVICES=1 /mnt/workspace/Dehaze/.venv/bin/python train.py \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --dataset_mode split_stem \
    --split train \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --batch_size 4 \
    --image_size 512 \
    --crop_size 256 \
    --sampling_timesteps 10 \
    --train_num_steps 30000 \
    --save_and_sample_every 5000 \
    --gradient_accumulate_every 2 \
    --optimizer adamw \
    --train_lr 1e-4 \
    --weight_decay 1e-4 \
    --mixed_precision fp16 \
    --num_workers 4 \
    --use_class_loss \
    --class_loss_weight 0.05 \
    --num_haze_classes 10 \
    --num_airlight_classes 6 \
    --output_dir /mnt/workspace/Dehaze/result/foundir_reborn_classloss_w005 \
    --resume_checkpoint /mnt/workspace/Dehaze/result/foundir_reborn_longrun_30k_continue/model-last.pt

GPU2：`class_loss_weight = 0.10`

cd /mnt/workspace/Dehaze/algorithm/FoundIR-reborn

CUDA_VISIBLE_DEVICES=2 /mnt/workspace/Dehaze/.venv/bin/python train.py \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --dataset_mode split_stem \
    --split train \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --batch_size 4 \
    --image_size 512 \
    --crop_size 256 \
    --sampling_timesteps 10 \
    --train_num_steps 30000 \
    --save_and_sample_every 5000 \
    --gradient_accumulate_every 2 \
    --optimizer adamw \
    --train_lr 1e-4 \
    --weight_decay 1e-4 \
    --mixed_precision fp16 \
    --num_workers 4 \
    --use_class_loss \
    --class_loss_weight 0.10 \
    --num_haze_classes 10 \
    --num_airlight_classes 6 \
    --output_dir /mnt/workspace/Dehaze/result/foundir_reborn_classloss_w010 \
    --resume_checkpoint /mnt/workspace/Dehaze/result/foundir_reborn_longrun_30k_continue/model-last.pt

GPU3：`class_loss_weight = 0.20`

cd /mnt/workspace/Dehaze/algorithm/FoundIR-reborn

CUDA_VISIBLE_DEVICES=3 /mnt/workspace/Dehaze/.venv/bin/python train.py \
    --dataroot /mnt/workspace/Dehaze/datasets \
    --dataset_mode split_stem \
    --split train \
    --lq_dirname haze_images \
    --gt_dirname original_images \
    --batch_size 4 \
    --image_size 512 \
    --crop_size 256 \
    --sampling_timesteps 10 \
    --train_num_steps 30000 \
    --save_and_sample_every 5000 \
    --gradient_accumulate_every 2 \
    --optimizer adamw \
    --train_lr 1e-4 \
    --weight_decay 1e-4 \
    --mixed_precision fp16 \
    --num_workers 4 \
    --use_class_loss \
    --class_loss_weight 0.20 \
    --num_haze_classes 10 \
    --num_airlight_classes 6 \
    --output_dir /mnt/workspace/Dehaze/result/foundir_reborn_classloss_w020 \
    --resume_checkpoint /mnt/workspace/Dehaze/result/foundir_reborn_longrun_30k_continue/model-last.pt

建议观察：
- `recon_loss`
- `cls_loss`
- `haze_ce`
- `airlight_ce`
- 5000 / 10000 / 15000 / 30000 步的推理图

如果大权重实验出现去雾能力变弱、颜色变怪、或者输出变保守，优先保留小权重版本。

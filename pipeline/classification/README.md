雾霾等级分类（多骨干网络）
===========================

本模块使用 torchvision 提供的主干网络，训练一个 10 分类的雾霾浓度分类器。

支持模型
--------

- `resnet18`
- `resnet50`
- `mobilenet_v3_small`
- `efficientnet_b0`
- `swin_t`
- `vit_b_16`

标签规则
--------

图像文件名需满足以下后缀格式：

`{original_name}_{beta_index}_{A_index}.jpg`

其中 `beta_index` 取值为 `[0, 9]`，作为最终分类标签。
训练脚本同时兼容旧命名 `{original_name}_{A_index}_{beta_index}.jpg`。

数据目录结构
------------

- `datasets/train/haze_images`
- `datasets/valid/haze_images`
- `datasets/test/haze_images`

输出目录
--------

所有训练结果统一输出到：

- `result/classification/<model_name>/best.pt`
- `result/classification/<model_name>/last.pt`
- `result/classification/<model_name>/run_meta.json`
- `result/classification/summary.json`

功能说明
--------

- 每个 epoch 的 train/valid/test 阶段均带有 tqdm 进度条（主进程显示）。
- 支持 DDP 多卡分布式训练（推荐用 `torchrun` 启动）。
- 支持早停机制（仅保留 `--early-stop-patience`，判定规则为“验证集 acc 只要高于历史最好即视为提升”）。
- 当前默认预处理为固定分辨率缩放：`Resize((image_size, image_size))`，避免 batch 尺寸不一致报错。
- 默认会在 `beta=0` 样本上屏蔽 A 头损失与 A 头评估（因为该情况下 A 不可辨识）；如需包含可加 `--include-beta0-in-a-loss`。

常用参数（最新）
----------------

- `--image-size`：输入分辨率（默认 `224`）
- `--early-stop-patience`：早停耐心轮数，`0` 表示关闭
- `--distributed`：开启 DDP
- `--dist-backend`：分布式后端（默认 `nccl`）
- `--include-beta0-in-a-loss`：将 `beta=0` 样本纳入 A 头损失/评估（默认关闭）

单模型训练（单卡）
------------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --model resnet18 \
  --data-root datasets \
  --output-dir result/classification \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda \
  --pretrained \
  --amp \
  --early-stop-patience 3
```

多模型顺序训练（单卡）
----------------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --models resnet18,resnet50,swin_t,vit_b_16 \
  --data-root datasets \
  --output-dir result/classification \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda \
  --pretrained \
  --amp \
  --early-stop-patience 3
```

DDP 多卡训练（单模型示例）
-------------------------

```bash
OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 pipeline/classification/train.py \
  --distributed \
  --dist-backend nccl \
  --model resnet18 \
  --data-root datasets \
  --output-dir result/classification \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda \
  --pretrained \
  --amp \
  --early-stop-patience 3
```

DDP 多卡训练（多模型示例）
-------------------------

```bash
OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 pipeline/classification/train.py \
  --distributed \
  --dist-backend nccl \
  --models resnet18,resnet50,vit_b_16,swin_t \
  --data-root datasets \
  --output-dir result/classification \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda \
  --pretrained \
  --amp \
  --early-stop-patience 3
```

快速冒烟测试
------------

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

仅评估已有权重
--------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --model resnet18 \
  --data-root datasets \
  --eval-only \
  --checkpoint result/classification/resnet18/best.pt
```

雾霾分类（支持单分类 beta 与双分类 beta+A）
============================================

本模块使用 torchvision 提供的主干网络，训练共享主干 + 双头分类器，支持两种任务模式：

- `--task-mode dual`：双分类（beta + A，默认）
- `--task-mode beta`：单分类（仅 beta）

双头定义：

- beta 头：10 分类雾霾浓度（主任务）
- A 头：环境光类型分类（辅助任务，仅 dual 模式生效）

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

训练目标
--------

- dual 模式：`loss = loss_beta + loss_a_weight * loss_a`
- beta 模式：`loss = loss_beta`（A 头损失与 A 头评估自动忽略）
- dual 默认设置下（推荐）：`beta=0` 样本不参与 A 头损失与 A 头评估
- 如需让 `beta=0` 也参与 A 头监督，可加 `--include-beta0-in-a-loss`（仅 dual 模式）

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
- dual 模式默认会在 `beta=0` 样本上屏蔽 A 头损失与 A 头评估（因为该情况下 A 不可辨识）；如需包含可加 `--include-beta0-in-a-loss`。
- 早停与 best checkpoint 基于 `valid beta_acc`（主任务指标）。

常用参数（最新）
----------------

- `--image-size`：输入分辨率（默认 `224`）
- `--early-stop-patience`：早停耐心轮数，`0` 表示关闭
- `--distributed`：开启 DDP
- `--dist-backend`：分布式后端（默认 `nccl`）
- `--task-mode`：`dual`（默认）或 `beta`
- `--beta-classes`：beta 类别数（默认 `10`）
- `--a-classes`：A 类别数（默认 `6`）
- `--loss-a-weight`：A 头损失权重（默认 `0.5`）
- `--include-beta0-in-a-loss`：将 `beta=0` 样本纳入 A 头损失/评估（默认关闭，仅 dual 模式）

单模型单分类训练（beta-only，单卡）
--------------------------------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --task-mode beta \
  --model resnet18 \
  --data-root datasets \
  --output-dir result/classification \
  --beta-classes 10 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda \
  --pretrained \
  --amp \
  --early-stop-patience 3
```

单模型双分类训练（单卡）
------------------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --task-mode dual \
  --model resnet18 \
  --data-root datasets \
  --output-dir result/classification \
  --beta-classes 10 \
  --a-classes 6 \
  --loss-a-weight 0.5 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda \
  --pretrained \
  --amp \
  --early-stop-patience 3
```

多模型双分类训练（单卡）
------------------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --task-mode dual \
  --models resnet18,resnet50,swin_t,vit_b_16 \
  --data-root datasets \
  --output-dir result/classification \
  --beta-classes 10 \
  --a-classes 6 \
  --loss-a-weight 0.5 \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda \
  --pretrained \
  --amp \
  --early-stop-patience 3
```

DDP 双分类训练（单模型示例）
----------------------------

```bash
OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 pipeline/classification/train.py \
  --distributed \
  --dist-backend nccl \
  --task-mode dual \
  --model resnet18 \
  --data-root datasets \
  --output-dir result/classification \
  --beta-classes 10 \
  --a-classes 6 \
  --loss-a-weight 0.5 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda \
  --pretrained \
  --amp \
  --early-stop-patience 3
```

DDP 双分类训练（多模型示例）
----------------------------

```bash
OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 pipeline/classification/train.py \
  --distributed \
  --dist-backend nccl \
  --task-mode dual \
  --models resnet18,resnet50,vit_b_16,swin_t \
  --data-root datasets \
  --output-dir result/classification \
  --beta-classes 10 \
  --a-classes 6 \
  --loss-a-weight 0.5 \
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
  --task-mode dual \
  --models resnet18,swin_t \
  --data-root datasets \
  --beta-classes 10 \
  --a-classes 6 \
  --loss-a-weight 0.5 \
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
  --task-mode dual \
  --model resnet18 \
  --data-root datasets \
  --eval-only \
  --checkpoint result/classification/resnet18/best.pt
```

评估结果说明
------------

- 训练/验证/测试日志同时输出：`beta_acc` 与 `a_acc`
- `summary.json` 中会记录：`test_beta_acc`、`test_a_acc`、`per_class_beta`、`per_class_a`
- 早停与 best 模型选择以 `beta_acc` 为准（主任务优先）
- 在 `--task-mode beta` 时，A 相关指标记为 `-1`（表示该模式下不参与评估）

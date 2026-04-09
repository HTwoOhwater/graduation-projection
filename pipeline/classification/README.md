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

`{original_name}_{A_index}_{beta_index}.jpg`

其中 `beta_index` 取值为 `[0, 9]`，作为最终分类标签。

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

单模型训练（单卡）
------------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --model resnet50 \
  --data-root datasets \
  --output-dir result/classification \
  --epochs 10 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda \
  --pretrained \
  --amp
```

多模型顺序训练（单卡）
----------------------

```bash
.venv/bin/python pipeline/classification/train.py \
  --models resnet18,resnet50,swin_t,vit_b_16 \
  --data-root datasets \
  --output-dir result/classification \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda \
  --pretrained \
  --amp
```

DDP 多卡训练（单模型示例）
-------------------------

```bash
torchrun --nproc_per_node=4 pipeline/classification/train.py \
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
  --amp
```

DDP 多卡训练（多模型示例）
-------------------------

```bash
torchrun --nproc_per_node=4 pipeline/classification/train.py \
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
  --amp
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

# FoundIR-reborn 本地项目说明

## 1. 这份说明的用途

这不是原始 `FoundIR` 仓库说明，而是你当前本地版本 `algorithm/FoundIR-reborn` 的使用说明。

写这份文档的原因是：
- 原始 `README` 主要服务作者公开仓库，不完全适合你当前的本地去雾实验。
- 你本地已经做过多轮清理和重构，实际使用方式和原仓库不同。
- 后续本地实验应优先参考这份说明，而不是直接照抄原始 `README`。

---

## 2. 当前目录定位

- 原始仓库对照版：`algorithm/FoundIR`
- 本地重构实验版：`algorithm/FoundIR-reborn`

`FoundIR-reborn` 是你当前真正继续开发和训练的版本。

---

## 3. 当前本地版本做过哪些关键改动

### 3.1 结构整理
- 训练入口和测试入口改成了正式 `argparse` 风格。
- `Trainer` 已从原先的 `src/model.py` 中拆出到独立文件。
- 公共工具函数已抽到 `src/utils.py`。
- 删除了与当前主线无关的大量附属目录和历史残留内容。

### 3.2 数据集适配
- 当前版本支持你自己的目录结构：

```text
datasets/
  train/
    haze_images/
    original_images/
  valid/
    haze_images/
    original_images/
  test/
    haze_images/
    original_images/
```

- 当前版本支持 `split_stem` 配对模式。
- 文件名支持这种映射：

```text
cloudy_0007_1_2.jpg -> cloudy_0007.jpg
```

其中：
- 倒数第二个后缀：雾霾浓度类别
- 倒数第一个后缀：环境光类别

这两个标签当前已能被 dataset 读出，虽然训练主干还没有直接用到。

### 3.3 推理与保存行为
- `test.py` 已支持显式 checkpoint 路径和输出路径。
- 推理保存名已经按 `LQ` 文件名保存，避免不同输入因为共用同一个 GT 名称而被覆盖。

### 3.4 checkpoint 保存逻辑
- 现在中途保存按 `save_and_sample_every` 字面值触发。
- 训练结束后会额外强制保存一份：

```text
model-last.pt
```

---

## 4. 当前训练入口最重要的语义

这里有一个最容易误解的点：

### `train_num_steps` 不是“在 resume 基础上再训练多少步”

它的真实语义是：

> 训练到总步数多少为止

也就是说，如果你从一个 `step=1300000` 的 checkpoint 恢复：

```text
--resume_milestone 2000
--train_num_steps 5000
```

那么训练会立刻结束，因为逻辑比较的是：

```python
while self.step < self.train_num_steps:
```

等价于：

```text
1300000 < 5000   # False
```

所以如果你希望从 `1300000` 继续多跑 `5000` 步，应该写成：

```text
--train_num_steps 1305000
```

这是当前版本必须牢记的使用规则。

---

## 5. 单卡训练

示例：

```bash
cd /mnt/workspace/Dehaze/algorithm/FoundIR-reborn

/mnt/workspace/Dehaze/.venv/bin/python train.py \
  --dataroot /mnt/workspace/Dehaze/datasets \
  --dataset_mode split_stem \
  --split train \
  --lq_dirname haze_images \
  --gt_dirname original_images \
  --batch_size 4 \
  --image_size 512 \
  --crop_size 256 \
  --sampling_timesteps 10 \
  --train_num_steps 1305000 \
  --gradient_accumulate_every 2 \
  --mixed_precision fp16 \
  --num_workers 4 \
  --results_folder /mnt/workspace/Dehaze/result/foundir_reborn_train \
  --resume_milestone 2000
```

说明：
- `batch_size` 在单卡时就是该卡 batch size。
- `mixed_precision fp16` 对 3090 这类卡通常更合适。

---

## 6. 多卡训练

当前版本已经适配 `accelerate launch`。

示例：

```bash
cd /mnt/workspace/Dehaze/algorithm/FoundIR-reborn

accelerate launch --num_processes 2 train.py \
  --dataroot /mnt/workspace/Dehaze/datasets \
  --dataset_mode split_stem \
  --split train \
  --lq_dirname haze_images \
  --gt_dirname original_images \
  --batch_size 4 \
  --image_size 512 \
  --crop_size 256 \
  --sampling_timesteps 10 \
  --train_num_steps 1305000 \
  --gradient_accumulate_every 2 \
  --mixed_precision fp16 \
  --num_workers 2 \
  --results_folder /mnt/workspace/Dehaze/result/foundir_reborn_train \
  --resume_milestone 2000
```

注意：
- 这里的 `batch_size` 更应理解为每个进程 / 每张卡的 batch size。
- 不建议默认打开 `--split_batches`。
- 如果多卡反而更慢，优先检查：
  - `num_workers` 是否过大
  - batch 是否太小
  - 是否正确使用了 `accelerate launch`

---

## 7. 推理

当前本地推理入口为：

- `algorithm/FoundIR-reborn/test.py`

示例：

```bash
cd /mnt/workspace/Dehaze/algorithm/FoundIR-reborn

/mnt/workspace/Dehaze/.venv/bin/python test.py \
  --dataroot /mnt/workspace/Dehaze/datasets \
  --dataset_mode split_stem \
  --split test \
  --lq_dirname haze_images \
  --gt_dirname original_images \
  --checkpoint_dir /mnt/workspace/Dehaze/algorithm/FoundIR-reborn \
  --checkpoint_milestone 2000 \
  --output_dir /mnt/workspace/Dehaze/result/foundir_test_split_stem \
  --image_size 1024 \
  --sampling_timesteps 4 \
  --crop_phase im2overlap \
  --crop_stride 512
```

---

## 8. 当前更适合参考的文件

如果你之后忘了从哪里看起，优先看这些：

- 训练入口：`algorithm/FoundIR-reborn/train.py`
- 推理入口：`algorithm/FoundIR-reborn/test.py`
- 数据读取：`algorithm/FoundIR-reborn/data/combined_dataset.py`
- 训练调度：`algorithm/FoundIR-reborn/src/trainer.py`
- 扩散与模型主体：`algorithm/FoundIR-reborn/src/model.py`

---

## 9. 一句话提醒

以后你在 `FoundIR-reborn` 上做实验时，最重要的两个提醒是：

1. `train_num_steps` 表示目标总步数，不是增量步数。
2. 本地实验流程以这份说明为准，不以原始仓库 README 为准。

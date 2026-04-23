# FoundIR-reborn 改进日志（2026-04-22）

## 1. 本阶段目标

本阶段工作的重点，不再是最初那轮“结构清理”，而是围绕 `FoundIR-reborn` 继续把它变成一个更适合本地去雾实验的版本，具体目标包括：

- 把当前重构版与原始 `FoundIR` 做一次干净对照，确认问题是不是出在后续修改上。
- 修正训练 checkpoint 的保存逻辑。
- 让训练入口和推理入口真正按“输出目录”和“权重路径”工作，而不是继续依赖 milestone 拼接路径。
- 修正多卡训练效率问题，使其更适合 `accelerate launch`。
- 解决“resume 会把优化器、EMA、step 和旧超参数状态一起覆盖回来”的问题。
- 用微调后的权重实际跑多张去雾图，观察阶段性效果。

---

## 2. `FoundIR` 与 `FoundIR-reborn` 的分离

为避免继续在同一份目录上混合“原始仓库”和“本地重构版”，做了以下处理：

- 当前本地重构版保留为：
  - `algorithm/FoundIR-reborn`
- 重新克隆了一份原始仓库作为对照：
  - `algorithm/FoundIR`

这样后续判断问题时，不再需要靠记忆区分“哪些逻辑是原始代码，哪些是后加的”。

---

## 3. 原始版与重构版的推理对照

### 3.1 对照目的

当时最大的疑问是：

> `FoundIR` 推理出来的结果变化不明显，是不是后续重构改坏了？

为了回答这个问题，使用了同一个预训练权重：

- `model-2000.pt`

并对同一批测试图分别用：

- 原始 `FoundIR`
- `FoundIR-reborn`

进行推理。

### 3.2 对照结果

对照结果表明：

- 至少两张样本，原始版和 `FoundIR-reborn` 推理结果逐像素完全一致。
- 说明此前 `FoundIR` 在你数据上“变化很小”或“效果一般”的现象，主因不是后续重构把代码改坏了。

### 3.3 当前结论

更合理的解释仍然是：

- 预训练权重与当前去雾数据域匹配有限。
- 当前任务中的雾类型和颜色分布更复杂。
- 训练步数还不够，模型仍处在早期适配阶段。

---

## 4. 修正 checkpoint 保存逻辑

### 4.1 原问题

原训练逻辑中，checkpoint 保存条件写成了：

```python
self.step % (self.save_and_sample_every * 10) == 0
```

这会导致：

- 参数名明明是 `save_and_sample_every`
- 实际保存却还要再乘 `10`

对微调和冒烟训练来说，这种行为非常不友好，容易出现：

- 训练确实发生了
- 但一直没有新权重落盘

### 4.2 修改内容

在 `algorithm/FoundIR-reborn/src/trainer.py` 中：

- 保存条件改为按字面值触发：

```python
self.step % self.save_and_sample_every == 0
```

- 并在训练结束后增加强制兜底保存：

```python
self.save('last')
```

### 4.3 修改后的行为

现在：

- 设多少步保存，就按多少步保存
- 训练结束后一定会生成：
  - `model-last.pt`

---

## 5. 修正训练输出路径语义

### 5.1 原问题

之前训练入口里：

- `results_folder`
- `output_dir`

语义混在一起，容易让人误以为：

- 一个是训练输出
- 另一个只是别名

但实际上训练保存与 resume 恢复逻辑又和这个目录强耦合，容易让人误判“结果到底保存到哪里”。

### 5.2 修改内容

在 `algorithm/FoundIR-reborn/train.py` 中：

- 训练输出统一走：
  - `--output_dir`
- `--results_folder` 保留为兼容旧写法的别名

也就是说，现在训练保存位置应该优先按：

```bash
--output_dir /path/to/dir
```

理解。

### 5.3 冒烟验证

已经做过 1 step 训练冒烟，验证：

- `output_dir` 路径可正常落盘
- 可生成：
  - `model-1.pt`
  - `model-last.pt`

验证目录示例：

- `result/foundir_reborn_outputdir_smoke`

---

## 6. 修正 checkpoint 加载方式

### 6.1 原问题

原来的 checkpoint 恢复逻辑是：

- 训练时依赖 `resume_milestone`
- 推理时依赖 `checkpoint_dir + checkpoint_milestone`

本质上都是在根据“编号”拼接：

```text
model-<milestone>.pt
```

这样的问题是：

- 输出目录和权重加载目录容易混淆
- 一旦你有多个训练目录，或者想直接拿某个具体权重做测试，就会很不直观

### 6.2 修改内容

在 `algorithm/FoundIR-reborn/src/trainer.py` 中：

- `load()` 已改成支持直接传 checkpoint 文件路径

在入口层：

- `train.py` 改成：
  - `--resume_checkpoint /path/to/model.pt`
- `test.py` 改成：
  - `--checkpoint /path/to/model.pt`

### 6.3 修改后的行为

现在训练恢复和推理加载都改为：

- 直接按权重文件路径工作

而不是再依赖：

- milestone
- checkpoint_dir
- 手动拼接 `model-xxx.pt`

这一步显著降低了实验管理的混乱程度。

---

## 7. 修正多卡训练逻辑，适配 `accelerate`

### 7.1 发现的问题

在检查并行训练速度时，发现当前版本存在几个明显问题：

1. `Accelerator(split_batches=True)` 会把固定总 batch 切到多卡上  
   这会让多卡总 batch 不变，只是被拆开，吞吐提升很有限，甚至可能更慢。

2. 原训练循环手写 `gradient_accumulate_every`，但没有按 `accelerate` 的标准 accumulate 语义来写  
   多卡下会导致多余的梯度同步。

3. `num_workers=8` 被硬编码在 `Trainer` 内部  
   多卡时会按进程放大，容易拖慢 CPU、图像解码和 IO。

4. 混合精度、pin_memory 等参数没有在入口层显式暴露  
   不便于根据硬件和并行策略调整。

### 7.2 修改内容

在 `train.py` 中新增：

- `--mixed_precision`
- `--split_batches`
- `--num_workers`
- `--no_pin_memory`

在 `src/trainer.py` 中：

- `Accelerator` 初始化改为带：
  - `gradient_accumulation_steps=...`
- 训练循环改为使用：
  - `with accelerator.accumulate(self.model):`
- dataloader 的：
  - `num_workers`
  - `pin_memory`
  改为由入口参数传入，不再硬编码

### 7.3 修改后的结果

当前版本现在同时兼容：

- 单卡：
  - `python train.py ...`
- 多卡：
  - `accelerate launch --num_processes N train.py ...`

并且更符合 `accelerate` 的标准使用方式。

---

## 8. 修正 resume 语义与旧状态覆盖问题

### 8.1 原问题

你之前在改这个项目时就遇到过一个很头疼的问题：

> 加载预训练权重时，不只是模型参数被加载回来，连优化器状态、EMA、step、scaler、甚至旧学习率状态也一并恢复，覆盖了当前这次训练自己的设置。

这在“微调已有权重”时非常不方便，因为你通常想要的是：

- 继承模型参数
- 但重新定义新的优化器、学习率和训练计划

### 8.2 修改内容

在 `algorithm/FoundIR-reborn/src/trainer.py` 中：

- `load()` 增加可控选项：
  - `load_step`
  - `load_optimizer`
  - `load_ema`

在 `train.py` 中新增：

- `--resume_load_step`
- `--resume_load_optimizer`
- `--resume_load_ema`

### 8.3 当前默认行为

训练时如果你写：

```bash
--resume_checkpoint /path/to/model.pt
```

默认只加载：

- 模型权重

默认不会加载：

- step
- optimizer
- scaler
- ema

这更适合“拿预训练权重做新一轮微调”的常见场景。

只有在你显式加上：

- `--resume_load_step`
- `--resume_load_optimizer`
- `--resume_load_ema`

时，才会恢复这些旧状态。

---

## 9. 新增优化器可配置能力

### 9.1 原问题

原来训练器里优化器写死为：

- `Adam`

不方便直接测试不同优化器对微调稳定性的影响。

### 9.2 修改内容

在 `train.py` 中新增参数：

- `--optimizer`
- `--beta1`
- `--beta2`
- `--weight_decay`

在 `src/trainer.py` 中新增支持：

- `adam`
- `adamw`

### 9.3 当前用途

现在你可以很方便地对比：

- `Adam`
- `AdamW`

而不用再改训练器源码。

---

## 10. 微调训练与推理验证

### 10.1 微调恢复方式

后续已经用新的“路径式 resume”逻辑做过验证：

- 从：
  - `algorithm/FoundIR-reborn/model-2000.pt`
- 成功恢复
- 并在新输出目录中正常保存：
  - `model-1300001.pt`
  - `model-last.pt`

验证目录示例：

- `result/foundir_reborn_resume_path_smoke`

### 10.2 5000 步微调

之后已基于预训练权重继续训练到：

- `1305000` 这一级别的 checkpoint

输出目录示例：

- `result/foundir_reborn_finetune_1305000`

其中包含：

- `model-1301.pt`
- `model-1302.pt`
- `model-1303.pt`
- `model-1304.pt`
- `model-1305.pt`
- `model-last.pt`

### 10.3 推理验证

随后使用：

- `result/foundir_reborn_finetune_1305000/model-last.pt`

进行了多轮推理测试，包括：

- 5 张有雾图推理
- 10 张有雾图推理
- 三联图对比（Input / Output / GT）

阶段性结论是：

- 目前仍然不能说效果已经成熟
- 但可以明确看出，这条线现在至少已经具备：
  - 可训练
  - 可恢复
  - 可保存
  - 可批量推理
  - 可持续对比验证

---

## 11. 当前阶段总结

到这一阶段为止，`FoundIR-reborn` 已经从“结构上能整理”推进到“实验流程能真正闭环”：

- 训练输出位置明确
- 权重恢复方式明确
- 训练状态是否覆盖可控
- 多卡启动方式明确
- 优化器可配置
- 微调与推理都已经做过实际验证

虽然当前视觉效果仍然一般，但现在的工程状态已经比最初的原始 `FoundIR` 明显更适合你继续做本地去雾实验。

---

## 12. 当前最重要的提醒

这一阶段之后，最需要记住的事情有三点：

1. 训练输出优先使用 `--output_dir`
2. checkpoint 恢复优先使用：
   - `--resume_checkpoint`
   - `--checkpoint`
3. 微调预训练权重时，默认只加载模型权重，不自动覆盖你新的优化器和训练状态

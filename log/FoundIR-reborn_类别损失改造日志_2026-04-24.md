# FoundIR-reborn 类别损失改造日志

日期：2026-04-24

## 1. 本次改造目标

在 `FoundIR-reborn` 当前重建主线已经较稳定的前提下，引入利用数据标签的辅助监督，验证：

- `haze_level_label`
- `airlight_label`

能否帮助模型更好地建模不同雾浓度与环境光条件。

本次改造的定位是：

- 不改推理链路
- 不改采样逻辑
- 不把分类分支变成主任务
- 仅把类别损失作为训练期辅助监督

---

## 2. 当前数据侧基础

在此前的 `split_stem` 数据改造中，`CombinedDataset` 已经能够从 LQ 文件名中提取：

- `haze_level_label`
- `airlight_label`

因此这次不需要再改数据命名规则，数据层已经具备加入类别损失的条件。

---

## 3. 模型改造内容

### 3.1 瓶颈层分类头

在 `algorithm/FoundIR-reborn/src/model.py` 中，对 `Unet` / `UnetRes` 做了最小改造：

- `Unet` 新增了 `forward_features()`
- 除主重建输出外，显式返回瓶颈层特征

在 `UnetRes` 中新增了一个训练时辅助分支：

- 先对瓶颈层特征做 `AdaptiveAvgPool2d(1)`
- 再通过两个线性分类头分别输出：
  - `haze_level_logits`
  - `airlight_logits`

### 3.2 设计原则

这次分类头挂在 **U-Net 瓶颈层**，原因是：

- 语义最集中
- 与恢复输出主干耦合相对最弱
- 第一版最稳，不容易把现有去雾能力直接扰乱

---

## 4. loss 改造内容

在 `ResidualDiffusion.p_losses()` 中，保留原有重建 loss，同时新增两个交叉熵分类损失：

- `CE(haze_level_logits, haze_level_label)`
- `CE(airlight_logits, airlight_label)`

最终总损失形式为：

```python
total_loss = recon_loss + class_loss_weight * (haze_ce + airlight_ce)
```

当前默认策略：

- 分类损失只作为辅助项
- 初始建议小权重起步

同时加了一个安全处理：

- 如果某个样本标签无效（如 `-1`），对应分类损失会自动跳过，不会因为异常命名样本直接报错

---

## 5. 训练入口参数新增

在 `algorithm/FoundIR-reborn/train.py` 中新增了以下参数：

- `--use_class_loss`
- `--class_loss_weight`
- `--num_haze_classes`
- `--num_airlight_classes`

当前默认类别数约定：

- `num_haze_classes = 10`
- `num_airlight_classes = 6`

这与当前数据文件名中观测到的类别范围一致。

---

## 6. 训练日志改进

在 `algorithm/FoundIR-reborn/src/trainer.py` 中，训练进度条现在会额外显示：

- `recon`
- `cls`
- `haze`
- `air`

也就是：

- 主重建损失
- 分类总损失
- 雾浓度分类损失
- 环境光分类损失

这样后续训练时可以直接判断：

- 分类头是否真正学到标签
- 类别损失是否过强干扰了重建

---

## 7. checkpoint 兼容策略

由于旧 checkpoint 中不包含新加的分类头参数，如果继续使用严格加载会直接报错。

因此在 `Trainer.load()` 中改为：

- `strict=False`

并补充打印：

- `missing keys`
- `unexpected keys`

当前语义是：

- 加载旧版 `FoundIR-reborn` checkpoint 时，主干权重正常恢复
- 新增分类头随机初始化

这满足当前“从已有恢复模型继续微调加入类别损失”的需求。

---

## 8. 冒烟验证结果

本次改造后已做两类验证：

### 8.1 语法检查

已通过：

- `train.py`
- `src/model.py`
- `src/trainer.py`

### 8.2 假数据前向冒烟

使用随机张量构造了最小 batch，验证：

- `UnetRes` 分类头可正常前向
- `ResidualDiffusion` 可同时计算：
  - `recon_loss`
  - `haze_ce`
  - `airlight_ce`
  - `cls_loss`
  - `total_loss`

说明本次改造已经达到“可训练”的状态。

---

## 9. 当前推荐实验策略

第一轮不建议大幅改结构，而是优先做 `class_loss_weight` 的对照实验。

建议优先测试：

- `0.02`
- `0.05`
- `0.10`
- `0.20`

目的：

- 看小权重辅助监督能否稳定提升
- 避免一开始就让分类目标把重建主任务带偏

---

## 10. 当前结论

截至本次改造完成，可以明确认为：

- `FoundIR-reborn` 已具备类别损失训练能力
- 数据标签已成功接入训练主线
- 推理路径未被破坏
- 接下来最关键的不是继续改代码，而是做权重比例实验，观察主观效果与指标变化


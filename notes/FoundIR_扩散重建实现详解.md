# FoundIR 中用于图像重建的扩散模型实现详解

## 1. 背景与目标

FoundIR 并不是把扩散模型当作“无条件图像生成器”来用，而是将其改造成**条件图像重建器**：

- 输入：退化图像（LQ）
- 输出：复原图像（GT 估计）
- 学习目标：优先学习“退化到清晰”的残差映射

这套思路的核心是：

1. 用条件输入约束扩散轨迹（不是纯噪声起步）
2. 用残差建模降低学习难度（比直接回归 GT 更稳定）
3. 通过少步采样完成重建（而不是长链生成）

---

## 2. 代码入口与模块分工

### 2.1 训练入口

- `algorithm/FoundIR/train.py`

这里完成以下事情：

- 构建配对数据集 `CombinedDataset`
- 设定 `condition=True`
- 设定目标 `objective='pred_res'`
- 构建 `UnetRes` 与 `ResidualDiffusion`
- 交给 `Trainer.train()` 执行训练

### 2.2 推理入口

- `algorithm/FoundIR/test.py`

这里会：

- 加载训练好的 checkpoint
- 使用 `ResidualDiffusion.sample()` 做恢复
- 支持大图切块推理（`crop_phase='im2overlap'`）

### 2.3 核心扩散实现

- `algorithm/FoundIR/src/model.py`

核心类：

- `UnetRes`：扩散主干网络封装
- `ResidualDiffusion`：重建版扩散过程（前向加噪 + 反向采样 + 训练损失）
- `Trainer`：训练循环与采样流程

### 2.4 数据组织

- `algorithm/FoundIR/data/combined_dataset.py`

关键点：

- 返回配对样本：`adap`（LQ）与 `gt`（GT）
- 可通过 meta 文件显式指定配对路径

---

## 3. 与标准 DDPM 的关键差异

标准 DDPM（简化）通常学习：

- `epsilon_theta(x_t, t)` 或 `x0_theta(x_t, t)`
- 前向过程主要是围绕 `x_0` 与噪声构造

FoundIR 改为“条件残差扩散”：

1. **条件输入参与网络输入**
   - `x_in = cat(x_t, x_input)`，即把当前扩散状态与 LQ 一起送入网络。

2. **学习目标优先是残差**
   - `x_res = x_input - x_start`（LQ 与 GT 的差）
   - 主配置为 `objective='pred_res'`

3. **前向噪声注入方程引入条件项**
   - 在 `q_sample` 中不仅有噪声项，还有显式 condition 驱动项。

4. **采样从条件邻域初始化**
   - 初始状态不是纯高斯噪声，而是 `x_input + sigma * noise` 形式。

这四点使模型更像“重建器”，而非“无条件生成器”。

---

## 4. 数学层面（按代码语义）

设：

- `x_input`：退化图（LQ）
- `x_start`：清晰图（GT）
- `x_res = x_input - x_start`
- `x_t`：扩散时刻状态

### 4.1 训练时构造前向样本

代码中（`ResidualDiffusion.q_sample`）可写成：

\[
 x_t = x_{start} + \alpha_t x_{res} + \beta_t \epsilon - \delta_t x_{input}
\]

其中：

- `alpha_t` 对应 `alphas_cumsum`
- `beta_t` 对应 `betas_cumsum`
- `delta_t` 对应 `delta_cumsum`

这一步本质是“把 LQ-GT 残差信息 + 条件图信息”同时编码进扩散状态。

### 4.2 网络预测目标

默认目标是 `pred_res`：

- 网络输出 `pred_res`
- 用 L1 损失对齐 `x_res`

\[
 \mathcal{L} = \|pred\_res - x\_{res}\|_1
\]

### 4.3 重建图恢复

在 `model_predictions` 中：

\[
 \hat{x}_0 = x_{input} - pred\_res
\]

这一步非常关键：把扩散输出转成直接重建结果。

### 4.4 反向采样更新

DDIM 分支中，更新公式不是标准 DDPM 原样，而是显式包含残差与条件项（代码里 `img = img - alpha * pred_res + delta * x_input + ...`）。

这说明它在反演链路中持续利用条件图，强化“朝重建目标收敛”。

---

## 5. 训练流程（逐步）

1. `CombinedDataset` 读取 `(LQ, GT)` 配对。
2. 在 `p_losses` 中得到：
   - `x_input = 2*LQ-1`
   - `x_start = 2*GT-1`
   - `x_res = x_input - x_start`
3. 随机采样时间步 `t`。
4. 用 `q_sample` 合成 `x_t`。
5. 拼接输入 `cat(x_t, x_input)` 送入 UNet。
6. 预测 `pred_res`。
7. 用 L1 监督 `pred_res` 与 `x_res`。
8. 反向传播更新参数。
9. EMA 维护稳定推理权重。

---

## 6. 推理流程（逐步）

1. 输入 LQ 图（可整图或切块）。
2. 采样初态：以 LQ 为中心加噪初始化。
3. 按时间序列执行 DDIM 反演（少步）。
4. 每步预测残差并更新状态。
5. 最终得到恢复图（通过 `x_input - pred_res` 或等价状态收敛结果）。
6. 若为切块模式，最后做重叠融合回整图。

---

## 7. 这套改法为什么可行

1. **重建任务本质是条件映射**，把 LQ 作为显式条件是正确方向。
2. **残差目标更稀疏、更低频**，比直接拟合 GT 往往更易优化。
3. **条件邻域初始化减少采样漂移**，少步采样仍有可用效果。
4. **L1 对重建任务友好**，能稳定收敛并保留结构一致性。

---

## 8. 可借鉴点（给你的项目）

如果你要把自己的扩散重建模型做得“可跑且有效”，FoundIR 的以下点值得借鉴：

1. 输入层面：`[x_t, condition]` 通道拼接。
2. 目标层面：优先做 `pred_res`（或 `pred_x0` + residual 辅助）。
3. 采样层面：从 condition 邻域初始化，减少纯噪声依赖。
4. 工程层面：保留切块推理与重叠融合，支持大图。

---

## 9. 代码中存在的工程问题（阅读时要注意）

FoundIR 思路是可行的，但实现层面有一些明显风险：

1. 训练脚本参数处理较脆弱（`sys.argv` 与 argparse 混用）。
2. 设备配置有硬编码（多卡 id 写死）。
3. 训练分支中存在布尔判断写法不严谨（字符串判断恒真风险）。
4. 训练阶段与两阶段脚本之间依赖“手动注释切换”，可维护性一般。

建议：借鉴其“建模方法”，而不是直接照搬其工程实现。

---

## 10. 一句话总结

FoundIR 的关键改造可以概括为：

**把扩散从“生成噪声到图像”改成“在条件图约束下学习并反演退化残差”，从而将扩散模型落地为图像重建器。**

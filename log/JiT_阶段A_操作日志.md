# JiT 阶段A（最小可跑条件化）操作日志

## 记录时间
- 2026-04-16

## 本阶段目标
将 JiT 从类别条件生成改为图像条件重建的最小可跑版本（不追求一次到位最优性能）。

## 依据文档
- notes/JiT_条件生成改造方案.md

## 阶段A拆分任务
1. 新增配对数据管线（LQ/GT）。
2. 修改前向接口支持条件图像 `y_cond`。
3. 实现最小条件注入方案（通道拼接）。
4. 先保持采样器框架不大改，验证训练可跑与基本重建能力。

## 当前已完成
1. 明确 JiT 原始机制与 FoundIR 条件重建机制差异。
2. 形成《JiT_条件生成改造方案》并确定分阶段路线。
3. 确认先从“阶段A最小改动”开始实施。
4. 建立本日志文件用于持续记录实验过程。

## 本阶段执行顺序（下一步）
1. 数据层
   - 新增 JiT 配对数据集类（读取 LQ/GT 配对）。
   - 支持通过文件名主干或 meta 文件进行配对。
2. 模型接口层
   - 将网络/封装前向改为接收 `(z_t, t, y_cond, optional_label)`。
3. 条件注入层
   - 使用通道拼接：`x_in = cat(z_t, y_cond)`。
   - patch embed 输入通道数对应调整。
4. 训练层
   - 先跑 1 epoch 冒烟，确认 loss 曲线下降、输出不崩。

## 计划中的验收标准
1. 训练脚本可运行，无 shape/接口错误。
2. 冒烟实验（小 batch + 少步）可完整走完。
3. 输出样例中，重建结果与输入存在有意义差异且不发散。

## 备注
- 当前仅记录阶段A启动与任务分解，代码改造尚未开始。
- 代码改造开始后，将在本文件继续追加“每次改动-验证结果-问题与修复”。

---

## 2026-04-17 阶段A执行记录（严格按阶段A重落地）

### 1) 本次改动（仅阶段A）
1. 数据层（完成）
   - 新增 `algorithm/JiT/util/paired_dataset.py`。
   - 支持 `stem` 与 `meta` 两种配对模式，输出 `(y_cond, x_gt, label)`。
2. 模型接口层（完成）
   - 修改 `algorithm/JiT/denoiser.py`：`forward(x, labels=None, y_cond=None)`。
   - paired 模式下默认不启用类别条件，仅在 `--use_paired_label_condition` 开启时使用 label。
3. 条件注入层（完成）
   - 修改 `algorithm/JiT/model_jit.py`：`forward(x, t, y=None, y_cond=None)`。
   - 使用通道拼接：`cat(z_t, y_cond)`。
   - patch embed 输入通道与输出通道支持配置（paired 默认输入 6 通道、输出 3 通道）。
4. 训练层（完成）
   - 修改 `algorithm/JiT/main_jit.py`：新增阶段A所需参数（`dataset_mode/train_split/lq_dirname/gt_dirname/pairing_mode/pair_meta/max_train_steps` 等）。
   - 修改 `algorithm/JiT/engine_jit.py`：训练循环适配 paired batch，保持原始 JiT 训练目标不变。

### 2) 冒烟验证（venv）
1. 命令：
   - `python main_jit.py --dataset_mode paired --data_path /mnt/workspace/Dehaze/datasets --train_split train --lq_dirname haze_images --gt_dirname original_images --pairing_mode stem --model JiT-B/16 --img_size 256 --batch_size 2 --epochs 1 --num_workers 2 --output_dir /mnt/workspace/Dehaze/result/jit_stageA_reapply --max_train_steps 2 --save_last_freq 1 --device cuda`
2. 结果：
   - 模型输入通道显示为 `Conv2d(6, 128, ...)`。
   - 训练正常进入 `Epoch: [0]`。
   - 首步 loss 为 `0.5996`。
   - 触发 `Reach max_train_steps=2` 并正常退出。
   - 训练结束：`Training time: 0:00:10`。

### 3) 边界确认（按你的要求）
1. 当前仅执行阶段A最小改造，不引入阶段B/C的目标函数和采样路径重构。
2. 后续若进入阶段B/C，将以 JiT 当前架构与训练机制为第一约束，不套用 FoundIR 的实现细节。

---

## 2026-04-17 阶段A执行记录（需求核对 + 损失函数可配）

### 1) 需求核对结论
1. 核心机制保持不变：
   - 仍预测速度场 `v`，未改为 `pred_res` 或 `pred_x0`。
   - 仍使用 JiT 原生 ODE 采样器（Euler/Heun）路径。
   - 仍是纯 ViT 主干与阶段A通道拼接条件注入。
2. 按最新要求，不推进阶段B/C实现，当前仅做阶段A范围内可控修改。

### 2) 本次改动（仅损失函数部分）
1. 修改 `algorithm/JiT/main_jit.py`，新增参数：
   - `--loss_type {l2,l1,l1_l2}`
   - `--loss_l1_weight`
   - `--loss_l2_weight`
2. 修改 `algorithm/JiT/denoiser.py`：
   - 保持监督目标为 `v` 不变。
   - 在 `v - v_pred` 上支持三种惩罚形式：L2、L1、加权 L1+L2。

### 3) 冒烟验证（venv）
1. 命令：
   - `python main_jit.py --dataset_mode paired --data_path /mnt/workspace/Dehaze/datasets --train_split train --lq_dirname haze_images --gt_dirname original_images --pairing_mode stem --model JiT-B/16 --img_size 256 --batch_size 2 --epochs 1 --num_workers 2 --output_dir /mnt/workspace/Dehaze/result/jit_stageA_lossmix --max_train_steps 2 --save_last_freq 1 --device cuda --loss_type l1_l2 --loss_l1_weight 0.7 --loss_l2_weight 0.3`
2. 结果：
   - 参数读取成功（`loss_type='l1_l2'`）。
   - 训练正常进入 `Epoch: [0]`。
   - 首步 loss 为 `0.6833`。
   - 按 `max_train_steps=2` 正常结束，`Training time: 0:00:09`。

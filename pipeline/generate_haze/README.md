Gen Haze
========

`gen_haze.py` 是纯离线雾图生成脚本（不包含在线 DataLoader 逻辑）。

支持两种模式：

- `single`：单张图像生成（快速质检）
- `build`：按配置离线批量生成（用于训练前一次性构建数据集）

安装依赖
--------

```bash
pip install pyyaml
```

配置文件
--------

默认配置文件：`configs/haze_config.yaml`

关键字段：

- `haze.A_values`：A 候选集合
- `haze.beta_values`：beta 基础序列（也可作为插值端点）
- `haze.beta_interpolation`：beta 插值配置（可选）
- `dataset`：建议包含 `train`、`valid`、`test` 三个 split，`--split all` 会依次处理

当前离线生成策略
----------------

当前策略已经更新为：

- 每张原图对所有 beta 完整采样（遍历全部 beta level）
- 在每个 beta level 上随机抽取 1 个 A 索引（可复现，受 seed 控制）
- 若 beta 有 10 个，则每张原图生成 10 张雾图
- 命名格式：`{原图名称}_{beta_index}_{A_index}`

beta 插值（第二方案）
--------------------

支持在 `[beta_min, beta_max]` 区间内做指数均匀插值：

```yaml
haze:
  beta_values: [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
  beta_interpolation:
    enabled: true
    method: exp
    num_levels: 10
    curve: 2.0
```

说明：

- `method: exp`：指数插值（推荐）
- `method: linear`：线性插值
- 插值端点会强制对齐到 `beta_values` 的最小/最大值
- 实际使用的 beta 序列会打印在终端并记录在 summary 中

自动备份机制
-----------

`build` 时若发现已有输出，会先自动备份：

- `datasets/<split>/haze_images`（旧）
- `datasets/<split>/haze_metadata.csv`（旧）

备份到：

- `datasets/<split>/backup_haze/<timestamp>/haze_images`
- `datasets/<split>/backup_haze/<timestamp>/haze_metadata.csv`

这样可以保留原始数据集，避免覆盖后无法回滚。

单张生成
--------

```bash
python gen_haze.py single \
  --config ./configs/haze_config.yaml \
  --img /path/to/clean.jpg \
  --depth /path/to/depth.png \
  --a-index 0 \
  --beta-index 1
```

离线批量生成
------------

生成所有 split：

```bash
python gen_haze.py build \
  --config ./configs/haze_config.yaml \
  --split all \
  --overwrite
```

只生成 train，且限制前 1000 对样本做快速试跑：

```bash
python gen_haze.py build \
  --config ./configs/haze_config.yaml \
  --split train \
  --max-pairs 1000 \
  --overwrite
```

输出结构
--------

批量生成后输出到：

- `datasets/<split>/haze_images/...`：雾图
- `datasets/<split>/haze_metadata.csv`：每张输出图对应的 clean/depth 路径与 A/beta 参数

终端会打印：

- `pairs`、`images_saved`
- `beta_values`（本次实际使用的 beta 序列）
- `backup`（若触发备份）

注意事项
--------

- 脚本为离线流程设计，适合先构建固定训练/验证/测试集，再进行模型训练。
- 深度图当前按灰度读取并进行归一化/反转/平滑处理。
- 图像与深度图匹配优先按相对路径同名，再回退到文件名 stem 匹配。

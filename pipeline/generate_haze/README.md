Gen Haze
========

`gen_haze.py` 现在是纯离线生成脚本，不再包含在线 DataLoader 逻辑。

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

默认配置文件在 `configs/haze_config.yaml`。

关键字段：

- `haze.A_values`：A 候选集合
- `haze.beta_values`：beta 候选集合
- `dataset` 建议包含 `train`、`valid`、`test` 三个 split，`--split all` 会依次处理这三项

当前离线生成策略固定为：

- 每张原图随机选择 2 个 A 索引（可复现，受 seed 控制）
- 与全部 beta 组合生成
- 若 beta 有 10 个，则每张原图生成 20 张雾图
- 命名格式：`{原图名称}_{A_index}_{beta_index}`

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

注意事项
--------

- 脚本为离线流程设计，适合先构建固定训练/验证/测试集，再进行模型训练。
- 深度图支持灰度或彩色编码图，彩色深度图会按 HSV 的 H 通道映射远近。
- 图像与深度图匹配优先按相对路径同名，再回退到文件名 stem 匹配。

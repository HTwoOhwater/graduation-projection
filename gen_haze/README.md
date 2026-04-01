Gen Haze
========

`gen_haze.py` 已改为 YAML 配置驱动，并提供两种模式：

- `single`：单张图像生成（用于快速验证效果）
- `dataloader`：在线生成 DataLoader（用于训练时按需生成）

两种模式共用同一套核心雾霾生成逻辑，`single` 本质是 `dataset/dataloader` 的本地化单样本实现。

安装依赖
--------

```bash
pip install pyyaml
```

如果你要用 `dataloader` 模式，还需要 PyTorch。

配置文件
--------

默认配置文件在 `configs/haze_config.yaml`。

关键字段：

- `haze.combinations`：雾霾参数组合（每组包含 `A`、`beta`、可选 `tint`、可选 `name`）
- `dataset.train` / `dataset.val`：数据路径与加载策略
- `output`：输出目录与图像后缀

示例：

```yaml
seed: 123
output:
	ext: jpg
	outdir: ./gen_haze_out

haze:
	combinations:
		- name: dusk_light
			A: [220, 120, 80]
			beta: 0.13
			tint: 0.15
		- name: sandstorm_heavy
			A: [210, 190, 140]
			beta: 0.22
			tint: 0.08

dataset:
	train:
		clean_dir: ../datasets/original_images/sunny
		depth_dir: ../datasets/depth_images/sunny
		image_exts: [jpg, jpeg, png, bmp, tif, tiff]
		random_combo: true
	val:
		clean_dir: ../datasets/original_images/cloudy
		depth_dir: ../datasets/depth_images/cloudy
		image_exts: [jpg, jpeg, png, bmp, tif, tiff]
		random_combo: false
```

单张生成
--------

```bash
python gen_haze.py single \
	--config ./configs/haze_config.yaml \
	--img /path/to/clean.jpg \
	--depth /path/to/depth.png \
	--combo-name dusk_light
```

或使用索引：

```bash
python gen_haze.py single \
	--config ./configs/haze_config.yaml \
	--img /path/to/clean.jpg \
	--depth /path/to/depth.png \
	--combo-index 0
```

在线 DataLoader 预检查
---------------------

```bash
python gen_haze.py dataloader \
	--config ./configs/haze_config.yaml \
	--split train \
	--batch-size 4 \
	--max-batches 2 \
	--save-preview \
	--preview-dir ./gen_haze_preview
```

这会在线生成雾霾图并迭代 DataLoader，同时可保存预览图做快速质检。

代码中训练接入
------------

你可以直接导入：

```python
from gen_haze import build_haze_dataloader

loader = build_haze_dataloader(
		config_path="./gen_haze/configs/haze_config.yaml",
		split="train",
		batch_size=8,
		shuffle=True,
		num_workers=4,
		return_tensor=True,
		list_collate=False,
)
```

如果你的训练图像尺寸不一致，建议：

- 在 YAML 里配置 `resize_hw: [H, W]` 保证同尺寸，或
- 保持 `list_collate=True` 自定义后续拼接逻辑

注意事项
--------

- 深度图支持灰度或彩色编码图。彩色深度图会按 HSV 的 H 通道映射远近。
- 深度会归一化到 `[0,1]` 后再进入指数透射率模型。
- 图像与深度图匹配优先按相对路径同名，再回退到文件名 stem 匹配。

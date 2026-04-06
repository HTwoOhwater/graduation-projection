from pathlib import Path
# import random

p = Path.cwd() / 'datasets'

p_depth = p / 'depth_images'
p_original = p / 'original_images'

depth_images = list(map(lambda x: x, (p_depth).iterdir()))
original_images = list(map(lambda x: x, (p_original).iterdir()))
# random.shuffle(depth_images)

# for i, depth_image in enumerate(depth_images):
#     original_image = p_original / depth_image.name
#     if original_image.exists():
#         depth_image.rename(p / 'train' / 'depth_images' / depth_image.name)
#         original_image.rename(p / 'train' / 'original_images' / original_image.name)

for depth_image, original_image in zip(depth_images, original_images):
    depth_image.rename(p_depth / (depth_image.name[6:]))
    original_image.rename(p_original / (original_image.name[9:]))
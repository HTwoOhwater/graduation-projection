import argparse
import os
from pathlib import Path

from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils


def should_skip_zero_suffix(path):
    parts = Path(path).stem.rsplit('_', 2)
    return len(parts) == 3 and parts[1] == '0'


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='folder of input hazy images')
parser.add_argument('--output_dir', type=str, required=True, help='folder to save predictions')
parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint path')
parser.add_argument('--gps', type=int, default=3)
parser.add_argument('--blocks', type=int, default=19)
parser.add_argument('--limit', type=int, default=0, help='optional cap on processed images')
parser.add_argument('--skip_zero_suffix', action='store_true', help='skip samples like *_0_*')
opt = parser.parse_args()

input_dir = Path(opt.input_dir)
output_dir = Path(opt.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(opt.checkpoint, map_location=device, weights_only=False)
net = FFA(gps=opt.gps, blocks=opt.blocks)
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net = net.to(device)
net.eval()

transform = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
])

image_paths = sorted([p for p in input_dir.iterdir() if p.is_file()])
processed = 0
for image_path in image_paths:
    if opt.skip_zero_suffix and should_skip_zero_suffix(image_path):
        continue
    print(f'processing {image_path.name}', flush=True)
    haze = Image.open(image_path).convert('RGB')
    haze_tensor = transform(haze)[None, :].to(device)
    with torch.no_grad():
        pred = net(haze_tensor)
    save_path = output_dir / f'{image_path.stem}_FFA.png'
    vutils.save_image(torch.squeeze(pred.clamp(0, 1).cpu()), save_path)
    processed += 1
    if opt.limit > 0 and processed >= opt.limit:
        break

print(f'saved {processed} predictions to {output_dir}')

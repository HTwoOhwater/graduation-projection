import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=100000)
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument('--resume',action='store_true',help='resume from output_dir/model-last.pt if it exists')
parser.add_argument('--eval_step',type=int,default=5000)
parser.add_argument('--eval_print_freq',type=int,default=100,help='print validation progress every N images')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default='',help='deprecated prefix-style checkpoint path')
parser.add_argument('--output_dir',type=str,default='./trained_models',help='directory to save checkpoints')
parser.add_argument('--pretrained_checkpoint',type=str,default='',help='load model weights only')
parser.add_argument('--resume_checkpoint',type=str,default='',help='resume full training state from checkpoint path')
parser.add_argument('--trainset',type=str,default='its_train')
parser.add_argument('--testset',type=str,default='its_test')
parser.add_argument('--net',type=str,default='ffa')
parser.add_argument('--gps',type=int,default=3,help='residual_groups')
parser.add_argument('--blocks',type=int,default=20,help='residual_blocks')
parser.add_argument('--bs',type=int,default=16,help='batch size')
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=240,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',action='store_true',help='no lr cos schedule')
parser.add_argument('--perloss',action='store_true',help='perceptual loss')
parser.add_argument('--dataroot',type=str,default='/mnt/workspace/Dehaze/datasets',help='dataset root')
parser.add_argument('--train_split',type=str,default='train',help='train split for custom dehaze dataset')
parser.add_argument('--test_split',type=str,default='valid',help='eval split for custom dehaze dataset')
parser.add_argument('--lq_dirname',type=str,default='haze_images',help='input image folder name')
parser.add_argument('--gt_dirname',type=str,default='original_images',help='gt image folder name')
parser.add_argument('--num_workers',type=int,default=4,help='dataloader workers')
parser.add_argument('--train_limit',type=int,default=0,help='optional cap on train samples, 0 means all')
parser.add_argument('--eval_limit',type=int,default=0,help='optional cap on eval samples, 0 means all')

opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'
model_name=opt.trainset+'_'+opt.net.split('.')[0]+'_'+str(opt.gps)+'_'+str(opt.blocks)
if opt.model_dir:
	opt.output_dir = opt.model_dir
opt.output_dir = os.path.abspath(opt.output_dir)
opt.last_checkpoint = os.path.join(opt.output_dir, 'model-last.pt')
log_dir='logs/'+model_name

print(opt)
print('output_dir:',opt.output_dir)


if not os.path.exists(opt.output_dir):
	os.makedirs(opt.output_dir, exist_ok=True)
if not os.path.exists('numpy_files'):
	os.mkdir('numpy_files')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('samples'):
	os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
	os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)

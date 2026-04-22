import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
from pathlib import Path
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'clear')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,20000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split('_')[0]
        clear_name=id+self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)


def lq_stem_to_gt_stem(stem):
    parts = stem.rsplit('_', 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return stem


class DehazeSplitDataset(data.Dataset):
    def __init__(self, root, split, train, size=crop_size, lq_dirname='haze_images', gt_dirname='original_images', limit=0):
        super().__init__()
        self.size = size
        self.train = train
        self.lq_dir = Path(root) / split / lq_dirname
        self.gt_dir = Path(root) / split / gt_dirname
        lq_paths = sorted([Path(entry.path) for entry in os.scandir(self.lq_dir) if entry.is_file()])
        gt_map = {Path(entry.path).stem: Path(entry.path) for entry in os.scandir(self.gt_dir) if entry.is_file()}
        self.samples = []
        for lq_path in lq_paths:
            gt_stem = lq_stem_to_gt_stem(lq_path.stem)
            gt_path = gt_map.get(gt_stem)
            if gt_path is not None:
                self.samples.append((str(lq_path), str(gt_path)))
                if limit > 0 and len(self.samples) >= limit:
                    break
        print(f'dehaze split={split} pairs={len(self.samples)} crop size={size}')

    def __getitem__(self, index):
        haze_path, clear_path = self.samples[index]
        haze = Image.open(haze_path).convert('RGB')
        clear = Image.open(clear_path).convert('RGB')
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze, clear = self.augData(haze, clear)
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.samples)

import os
pwd=os.getcwd()
print(pwd)
path=opt.dataroot


def build_loader(name):
    if name == 'its_train':
        return DataLoader(
            dataset=RESIDE_Dataset(path+'/RESIDE/ITS', train=True, size=crop_size),
            batch_size=BS,
            shuffle=True,
            num_workers=opt.num_workers
        )
    if name == 'its_test':
        return DataLoader(
            dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/indoor', train=False, size='whole img'),
            batch_size=1,
            shuffle=False,
            num_workers=opt.num_workers
        )
    if name == 'ots_train':
        return DataLoader(
            dataset=RESIDE_Dataset(path+'/RESIDE/OTS', train=True, format='.jpg'),
            batch_size=BS,
            shuffle=True,
            num_workers=opt.num_workers
        )
    if name == 'ots_test':
        return DataLoader(
            dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/outdoor', train=False, size='whole img', format='.png'),
            batch_size=1,
            shuffle=False,
            num_workers=opt.num_workers
        )
    if name == 'dehaze_train':
        return DataLoader(
            dataset=DehazeSplitDataset(
                path,
                split=opt.train_split,
                train=True,
                size=crop_size,
                lq_dirname=opt.lq_dirname,
                gt_dirname=opt.gt_dirname,
                limit=opt.train_limit
            ),
            batch_size=BS,
            shuffle=True,
            num_workers=opt.num_workers
        )
    if name == 'dehaze_test':
        return DataLoader(
            dataset=DehazeSplitDataset(
                path,
                split=opt.test_split,
                train=False,
                size='whole img',
                lq_dirname=opt.lq_dirname,
                gt_dirname=opt.gt_dirname,
                limit=opt.eval_limit
            ),
            batch_size=1,
            shuffle=False,
            num_workers=opt.num_workers
        )
    raise KeyError(f'Unknown loader name: {name}')

if __name__ == "__main__":
    pass

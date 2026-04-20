"""FoundIR 训练/测试调度器。

这个文件承载的是“工程调度层”，而不是模型定义层。
它负责：
1. dataloader 接入
2. optimizer / EMA / checkpoint
3. train() 训练循环
4. test() 推理调度

之所以单独拆出来，是为了让你以后调训练流程时，不用把手伸进 model.py。
"""

import math
import os
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from accelerate import Accelerator
from ema_pytorch import EMA
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm

from src.utils import cycle, exists, has_int_squareroot


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        opts,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_unet=1,
        num_samples=25,
        results_folder='./results/sample',
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
        condition=False,
        sub_dir=False,
        data_mode='paired',
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.accelerator.native_amp = amp
        self.num_unet = num_unet
        self.model = diffusion_model
        self.results_folder = results_folder

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition
        self.data_mode = data_mode
        self.ema = EMA(diffusion_model, beta=ema_decay,
              update_every=ema_update_every)

        if self.condition:
            if opts.phase == "train":
                # data_mode 现在是显式配置，不再靠 results_folder 名称隐式决定行为。
                if self.data_mode in ('combined', 'all', 'paired', 'meta_info'):
                    self.dl = cycle(self.accelerator.prepare(DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)))
                elif self.data_mode == 'paired_split':
                    self.dl_light = cycle(self.accelerator.prepare(DataLoader(dataset[0], batch_size=32, shuffle=True, pin_memory=True, num_workers=8)))
                    self.dl_night = cycle(self.accelerator.prepare(DataLoader(dataset[1], batch_size=32, shuffle=True, pin_memory=True, num_workers=8)))
                else:
                    raise ValueError(f"Unsupported Trainer data_mode: {self.data_mode}")

            else:
                self.sample_dataset = dataset

        self.opt0 = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        if self.accelerator.is_main_process:
            self.set_results_folder(results_folder)

        self.step = 0

        self.model, self.opt0 = self.accelerator.prepare(self.model, self.opt0)

        device = self.accelerator.device
        self.device = device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt0': self.opt0.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.results_folder) / f'model-{milestone}.pt'
        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            self.model = self.accelerator.unwrap_model(self.model)

            self.model.load_state_dict(data['model'])
            self.step = data['step']

            self.opt0.load_state_dict(data['opt0'])
            self.opt0.param_groups[0]['capturable'] = True

            self.ema.load_state_dict(data['ema'])

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            print("load model - "+str(path))

    def train(self):
        # 训练循环留在调度层，后续替换 optimizer、日志、resume 逻辑时更容易定位。
        accelerator = self.accelerator

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = [0]

                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        # 这里把 dataloader 输出整理成模型真正需要的 [gt, cond_input, task] 结构。
                        if self.data_mode in ('combined', 'all', 'paired', 'meta_info'):
                            data = next(self.dl)
                        elif self.data_mode == 'paired_split':
                            batch1 = next(self.dl_light)
                            batch2 = next(self.dl_night)
                            data = {}
                            for k, v in batch1.items():
                                if 'path' in k:
                                    data[k] = batch1[k] + batch2[k]
                                else:
                                    data[k] = torch.cat([batch1[k], batch2[k]], dim=0)
                        else:
                            raise ValueError(f"Unsupported Trainer data_mode: {self.data_mode}")
                        gt = data["gt"].to(self.device)
                        cond_input = data["adap"].to(self.device)

                        task = data["A_paths"]
                        data = [gt, cond_input, task]
                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        for i in range(self.num_unet):
                            loss[i] = loss[i] / self.gradient_accumulate_every
                            total_loss[i] = total_loss[i] + loss[i].item()

                    for i in range(self.num_unet):
                        self.accelerator.backward(loss[i])

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt0.step()
                self.opt0.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    if self.step != 0 and self.step % (self.save_and_sample_every * 10) == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)

                pbar.set_description(f'loss_unet0: {total_loss[0]:.4f}')
                pbar.update(1)

        accelerator.print('training complete')

    def test(self, sample=False, last=True, FID=False, crop_phase=None, crop_size=None, crop_stride=None):
        # 当前 test() 仍然偏重，后续如果继续重构，优先把 patch 推理辅助逻辑再拆出去。
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = DataLoader(
                dataset=self.sample_dataset,
                batch_size=1)
            i = 0
            tran = transforms.ToTensor()
            for items in loader:
                if self.condition:
                    file_ = items["B_paths"][0]
                    file_name = file_.split('/')[-3]
                    lq_path = items["A_paths"][0]
                    output_name = os.path.basename(lq_path).replace('_fake_B', '')
                else:
                    file_name = f'{i}.png'
                    output_name = file_name

                i += 1

                start_time = time.time()

                with torch.no_grad():
                    batches = self.num_samples

                    data = items
                    x_input_sample = data["adap"].to(self.device)
                    _, _, h, w = x_input_sample.shape
                    if crop_phase == 'weight' and crop_size is not None:
                        if (h * w) > crop_size**2:
                            patches, positions = self.split_image(x_input_sample, crop_size, crop_stride)
                            processed_patches = []
                            for p in patches:
                                p_images_list = list(self.ema.ema_model.sample(
                                p, batch_size=batches, last=last, task=file_))
                                p_images_list = [p_images_list[-1]]
                                p_images = torch.cat(p_images_list, dim=0)
                                processed_patches.append(p_images)

                            all_images = self.merge_patches_with_weights(processed_patches, positions, x_input_sample.shape, crop_size, crop_stride)
                        else:
                            all_images_list = list(self.ema.ema_model.sample(
                            x_input_sample, batch_size=batches, last=last, task=file_))
                            all_images_list = [all_images_list[-1]]
                            all_images = torch.cat(all_images_list, dim=0)

                    elif crop_phase == 'im2overlap' and crop_size is not None:
                        # 大图推理时走 patch 切分与重组，避免直接整图爆显存。
                        if (h * w) > crop_size**2:
                            patches, idx, size = self.img2patch(x_input_sample, scale=1, crop_size=crop_size)

                            with torch.no_grad():
                                n = len(patches)
                                outs = []
                                m = 1
                                i = 0
                                while i < n:
                                    j = i + m
                                    if j >= n:
                                        j = n
                                    pred = output = self.ema.ema_model.sample(patches[i:j], batch_size=batches, last=last, task=file_)
                                    if isinstance(pred, list):
                                        pred = pred[-1]
                                    outs.append(pred.detach())
                                    i = j
                                output = torch.cat(outs, dim=0)

                            all_images = self.patch2img(output, idx, size, scale=1, crop_size=crop_size)
                        else:
                            all_images_list = list(self.ema.ema_model.sample(
                            x_input_sample, batch_size=batches, last=last, task=file_))
                            all_images_list = [all_images_list[-1]]
                            all_images = torch.cat(all_images_list, dim=0)

                    else:
                        all_images_list = list(self.ema.ema_model.sample(
                            x_input_sample, batch_size=batches, last=last, task=file_))
                        all_images_list = [all_images_list[-1]]
                        all_images = torch.cat(all_images_list, dim=0)

                print(time.time()-start_time)

                if last:
                    nrow = int(math.sqrt(self.num_samples))
                else:
                    nrow = all_images.shape[0]
                save_path = str(self.results_folder / file_name)
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, output_name)
                utils.save_image(all_images, full_path, nrow=nrow)
                print("test-save "+full_path)

        print("test end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)

    def split_image(self, image, patch_size, overlap):
        b, c, h, w = image.shape
        patches = []
        positions = []

        for i in range(0, w, patch_size - overlap):
            for j in range(0, h, patch_size - overlap):
                right = min(i + patch_size, w)
                bottom = min(j + patch_size, h)
                patch = image[:, :, j:bottom, i:right]
                patches.append(patch)
                positions.append((i, j))
        return patches, positions

    def create_weight_map(self, patch_size, overlap):
        weight_map = torch.ones((patch_size, patch_size), dtype=torch.float32).to(self.device)
        ramp = torch.linspace(0, 1, overlap).to(self.device)
        weight_map[:overlap, :] *= ramp[:, None]
        weight_map[-overlap:, :] *= ramp.flip(0)[:, None]
        weight_map[:, :overlap] *= ramp[None, :]
        weight_map[:, -overlap:] *= ramp.flip(0)[None, :]
        return weight_map

    def merge_patches_with_weights(self, patches, positions, image_size, patch_size, overlap):
        b, c, h, w = image_size
        result = torch.zeros((b, c, h, w), dtype=torch.float32).to(self.device)
        weight = torch.zeros((b, c, h, w), dtype=torch.float32).to(self.device)
        weight_map = self.create_weight_map(patch_size, overlap).unsqueeze(0).unsqueeze(0)

        for patch, (i, j) in zip(patches, positions):
            patch_h, patch_w = patch.shape[2:]
            weighted_patch = patch * weight_map[:, :, :patch_h, :patch_w]
            result[:, :, j:j+patch_h, i:i+patch_w] += weighted_patch
            weight[:, :, j:j+patch_h, i:i+patch_w] += weight_map[:, :, :patch_h, :patch_w]

        result /= weight
        return result

    def img2patch(self, lq, scale=1, crop_size=1024, overlap=512):
        # 这类 patch 工具函数现在先留在 Trainer，后续可以再抽到独立 inference utils。
        b, c, hl, wl = lq.size()
        h, w = hl * scale, wl * scale
        sr_size = (b, c, h, w)
        assert b == 1

        crop_size_h, crop_size_w = crop_size // scale * scale, crop_size // scale * scale

        step_j = crop_size_w - overlap
        step_i = crop_size_h - overlap

        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(lq[:, :, i // scale : (i + crop_size_h) // scale, j // scale : (j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        return torch.cat(parts, dim=0), idxes, sr_size

    def patch2img(self, outs, idxes, sr_size, scale=1, crop_size=1024):
        preds = torch.zeros(sr_size).to(outs.device)
        b, c, h, w = sr_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        crop_size_h, crop_size_w = crop_size // scale * scale, crop_size // scale * scale

        for cnt, each_idx in enumerate(idxes):
            i = each_idx['i']
            j = each_idx['j']

            preds[0, :, i : i + crop_size_h, j : j + crop_size_w] += outs[cnt]
            count_mt[0, 0, i : i + crop_size_h, j : j + crop_size_w] += 1.

        count_mt = torch.clamp(count_mt, min=1.0)

        return (preds / count_mt).to(outs.device)

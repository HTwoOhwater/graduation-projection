# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# 本软件遵循 Monodepth2 许可协议，仅允许非商业用途。
# 完整许可条款请查看 LICENSE 文件。

from __future__ import absolute_import, division, print_function

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import ipdb
from torchvision import transforms, datasets

import algorithm.depth_generation.networks as networks
from algorithm.depth_generation.layers import disp_to_depth
from algorithm.depth_generation.utils import download_model_if_doesnt_exist
from scipy.signal import convolve2d
import skimage.filters.rank as sfr
from skimage.morphology import disk
from skimage.util import img_as_ubyte


def parse_args():
    parser = argparse.ArgumentParser(
        description='Monodepthv2 模型的简易测试与数据生成脚本。')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--save_path', type=str,
                        help='path to save image or numpy', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--add_fog_method', type=str,
                        help='different methods',
                        choices=[
                            "mean",
                            "mean+min+max",
                            "mean+min+max+th"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """对单张图像或文件夹中的图像执行深度预测与雾图合成。"""
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # 加载预训练模型
    print("   Loading pretrained encoder")  # 编码器部分
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # 读取该模型训练时使用的输入分辨率，并加载模型权重
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)  # 加载模型权重
    encoder.to(device)  # 移动模型
    encoder.eval()  # 测试模式

    print("   Loading pretrained decoder")  # 解码器部分
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)  # 一样的就不多讲了
    depth_decoder.to(device)
    depth_decoder.eval()

    # 收集待处理图像路径
    if os.path.isfile(args.image_path):
        # 仅测试单张图像
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*/*.{}'.format(args.ext)))
        output_directory = args.save_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    train_id = 0
    val_id = 0
    test_id = 0
    train_arr = np.random.choice(len(paths) * 10, int(len(paths) * 10 * 0.7), replace=False)
    val_arr = train_arr[int(len(paths) * 10 * 0.6):]
    train_arr = train_arr[:int(len(paths) * 10 * 0.6)]

    img_path_train = []
    label_train = []
    img_path_val = []
    label_val = []
    img_path_test = []
    label_test = []

    # 逐张图像执行预测与合成
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            # import ipdb;
            # ipdb.set_trace()

            if image_path.endswith(("_disp.jpg", "_fog.jpg", "smooth.jpg", ".npy")):
                # 避免对已生成的结果图再次做深度预测
                continue

            # 读取并预处理输入图像
            input_image = pil.open(image_path).convert('RGB')
            input_image_tosave = np.array(input_image)
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # 深度预测，这里比较有意思，深度特征解耦了，所以可以看到这个模型的encoder和decoder
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]  # 获得深度图
            disp_resized = torch.nn.functional.interpolate(  # 做插值，因为这个模型本身输出分辨率有限且固定
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            deep_scale = 100
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # 生成伪彩色深度图所需的归一化参数（此处默认未保存）
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            vmin = disp_resized_np.min()
            normalizer = mpl.colors.Normalize(vmin, vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            # im = pil.fromarray(colormapped_im)

            # name_dest_im = os.path.join(output_directory, "{}_des{}_disp.jpg".format(output_name,deep_scale))
            # im.save(name_dest_im)

            # print("   Processed {:d} of {:d} images - saved prediction to {}".format(idx + 1, len(paths), name_dest_im))

            # 深度图平滑处理配置
            w_max = 128
            w_min = 128
            w_mean = 96

            if args.add_fog_method == 'mean+min+max':
                print('add min_max filter')
                # 最值滤波平滑
                disp_resized_np = sfr.maximum(disp_resized_np, disk(w_max)) / 255
                disp_resized_np = disp_resized_np.astype(np.float32)
                disp_resized_np = sfr.minimum(disp_resized_np, disk(w_min)) / 255
                disp_resized_np = disp_resized_np.astype(np.float32)

            if args.add_fog_method == 'mean+min+max+th':
                print('add th filter')
                # 阈值约束 + 最值滤波平滑
                disp_resized_np_max = sfr.maximum(disp_resized_np, disk(w_max)) / 255
                disp_resized_np_max = disp_resized_np_max.astype(np.float32)
                th = np.percentile(disp_resized_np, 15)
                for i in range(0, original_height):
                    for j in range(0, original_width):
                        if disp_resized_np[i][j] < th:
                            disp_resized_np_max[i][j] = disp_resized_np[i][j]
                disp_resized_np = sfr.minimum(disp_resized_np_max, disk(w_min)) / 255
                disp_resized_np = disp_resized_np.astype(np.float32)

            # 均值卷积平滑
            sigma = 1
            window_mean = np.ones((w_mean, w_mean))
            window_mean /= np.sum(window_mean)
            disp_resized_np_smooth = convolve2d(disp_resized_np, window_mean, mode="same", boundary="symm")
            vmax = np.percentile(disp_resized_np_smooth, 95)

            # ============================
            # 雾霾合成核心代码（重点学习）
            # ============================
            # 第一步：将平滑后的视差转换为“可用于物理衰减计算”的深度值
            # disp_to_depth 返回反深度/深度结果；这里使用 depth_resized_np_smooth 继续处理
            deep_scale = 100
            disp_test_level, depth_resized_np_smooth = disp_to_depth(disp_resized_np_smooth, 0.1, deep_scale)

            # 第二步：把深度归一化并映射到 [0.1, 10] 区间
            # 这样做是为了控制指数衰减项 exp(-beta * d) 的数值范围，避免过大或过小
            scale_max = depth_resized_np_smooth.max()
            scale_min = depth_resized_np_smooth.min()
            depth_resized_np_smooth = ((depth_resized_np_smooth - scale_min) / (scale_max - scale_min)) * (
                        10 - 0.1) + 0.1

            # 第三步：定义不同雾浓度 beta（共 10 档）与全局大气光分位数 alph
            # beta 越大，透射率 t = exp(-beta*d) 越小，雾越浓
            betas = [0, 0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69]
            alphs = [70]
            J = input_image_tosave
            I = np.empty_like(J)
            # import ipdb;
            # ipdb.set_trace()
            for alph in alphs:
                for b in range(0, 10):
                    beta = betas[b]

                    # 第四步：依据大气散射模型计算透射率
                    # t(x) = exp(-beta * d(x))
                    t = np.exp(-beta * depth_resized_np_smooth)

                    # 第五步：估计大气光 A（此处用原图像素分位数近似）
                    A = np.percentile(J, alph)

                    # 第六步：逐通道执行散射合成
                    # I(x) = J(x) * t(x) + A * (1 - t(x))
                    # 其中：
                    # J(x)：清晰图像
                    # I(x)：合成雾图
                    # A：全局大气光
                    # t(x)：透射率
                    I[:, :, 0] = J[:, :, 0] * t + A * (1 - t)
                    I[:, :, 1] = J[:, :, 1] * t + A * (1 - t)
                    I[:, :, 2] = J[:, :, 2] * t + A * (1 - t)
                    fog_im = pil.fromarray(I)

                    # 第七步：将每个雾浓度样本按随机索引划分到 train/val/test
                    if idx * 10 + b in train_arr:
                        save_path = os.path.join(output_directory, "train/" + "{}.jpg".format(train_id))
                        img_path_train.append(image_path)
                        label_train.append(int(b))
                        train_id = train_id + 1

                    elif idx * 10 + b in val_arr:
                        save_path = os.path.join(output_directory, "val/" + "{}.jpg".format(val_id))
                        img_path_val.append(image_path)
                        label_val.append(int(b))
                        val_id = val_id + 1

                    else:
                        save_path = os.path.join(output_directory, "test/" + "{}.jpg".format(test_id))
                        img_path_test.append(image_path)
                        label_test.append(int(b))
                        test_id = test_id + 1
                    fog_im.save(save_path)
                print("   Processed {:d} of {:d} images ".format(idx, len(paths)))

    names = locals()
    for name in ["train", "val", "test"]:
        # 保存标签与来源路径，便于后续分类训练加载
        label_savepath = os.path.join(output_directory, "{}/{}_label.npy".format(name, name))
        path_savepath = os.path.join(output_directory, "{}/{}_path.npy".format(name, name))
        img_path = np.array(names['img_path_' + name])
        label = np.array(names['label_' + name])
        np.save(label_savepath, label)
        print(" saved label_{}.npy".format(name))
        np.save(path_savepath, img_path)
        print(" saved img_path_{}.npy".format(name))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)

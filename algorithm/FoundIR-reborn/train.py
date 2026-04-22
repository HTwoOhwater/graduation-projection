"""FoundIR 训练入口。

当前布局刻意把“训练入口参数解析”和“模型/数据构建”放在这一层，
而不是散落到 model 或 trainer 里。
这样做的目的是把“实验配置”留在入口，把“算法实现”留在内部模块，
后续你切换数据集、checkpoint 路径或训练超参数时，不需要反复改底层代码。
"""

import argparse
from src.model import ResidualDiffusion, UnetRes, set_seed
from src.trainer import Trainer
from data.combined_dataset import CombinedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train FoundIR on a paired restoration dataset")
    parser.add_argument("--dataroot", type=str, required=True, help="Dataset root containing LQ/GT or paths referenced by meta")
    parser.add_argument("--meta", type=str, default=None, help="Optional meta file describing GT/LQ pairs")
    parser.add_argument("--dataset_mode", type=str, default="split_stem", choices=["meta_info", "paired", "split_stem"])
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--split", type=str, default="train", help="Dataset split used in split_stem mode")
    parser.add_argument("--lq_dirname", type=str, default="haze_images", help="LQ directory name used in split_stem mode")
    parser.add_argument("--gt_dirname", type=str, default="original_images", help="GT directory name used in split_stem mode")
    parser.add_argument("--max_dataset_size", type=int, default=int(1e9))
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of dataloader")
    parser.add_argument("--load_size", type=int, default=268, help="Scale images to this size before crop")
    parser.add_argument("--crop_size", type=int, default=256, help="Crop images to this size")
    parser.add_argument("--direction", type=str, default="AtoB", help="AtoB or BtoA")
    parser.add_argument("--preprocess", type=str, default="crop",
                        help="resize_and_crop | crop | scale_width | scale_width_and_crop | none")
    parser.add_argument("--no_flip", action="store_true", help="Disable random flip augmentation")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--sampling_timesteps", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--train_num_steps", type=int, default=500000)
    parser.add_argument("--save_and_sample_every", type=int, default=1000)
    parser.add_argument("--train_lr", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulate_every", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--split_batches", action="store_true",
                        help="Let Accelerate split a fixed global batch across devices. Leave off for normal per-device batching.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers per process")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable dataloader pin_memory")
    parser.add_argument("--objective", type=str, default="pred_res")
    parser.add_argument("--test_res_or_noise", type=str, default="res")
    parser.add_argument("--sum_scale", type=float, default=0.01)
    parser.add_argument("--delta_end", type=float, default=1.4e-3)
    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--results_folder", type=str, default="./ckpt_single_multi")
    parser.add_argument("--resume_milestone", type=int, default=0,
                        help="Resume from model-<milestone>.pt in results_folder if > 0")
    return parser.parse_args()


def build_dataset(args):
    # 数据构建单独封装，便于以后替换成你自己的 dataset 类或别的数据模式。
    return CombinedDataset(
        args,
        args.image_size,
        augment_flip=not args.no_flip,
        equalizeHist=True,
        crop_patch=True,
        generation=False,
        task=args.dataset_mode,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    # 入口层只负责“组装对象”，不直接写训练细节。
    dataset = build_dataset(args)
    num_unet = 1
    condition = True

    # 主干网络与扩散过程分开实例化，方便你之后单独替换 backbone 或 diffusion 配置。
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=condition,
        objective=args.objective,
        test_res_or_noise=args.test_res_or_noise,
    )

    diffusion = ResidualDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        delta_end=args.delta_end,
        sampling_timesteps=args.sampling_timesteps,
        objective=args.objective,
        loss_type=args.loss_type,
        condition=condition,
        sum_scale=args.sum_scale,
        test_res_or_noise=args.test_res_or_noise,
    )

    # Trainer 负责训练调度，不再和模型定义混在一个文件里。
    trainer = Trainer(
        diffusion,
        dataset,
        args,
        train_batch_size=args.batch_size,
        num_samples=1,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        amp=args.mixed_precision != "no",
        fp16=args.mixed_precision == "fp16",
        mixed_precision=args.mixed_precision,
        split_batches=args.split_batches,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        convert_image_to="RGB",
        results_folder=args.results_folder,
        condition=condition,
        save_and_sample_every=args.save_and_sample_every,
        num_unet=num_unet,
        data_mode=args.dataset_mode,
    )

    if args.resume_milestone > 0:
        trainer.load(args.resume_milestone)
    trainer.train()


if __name__ == "__main__":
    main()

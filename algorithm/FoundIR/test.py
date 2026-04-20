"""FoundIR 推理入口。

这一层只负责：
1. 解析推理参数
2. 构建数据集
3. 构建模型与扩散器
4. 调用 Trainer.test()

这样布局的核心原因是：推理行为应当由显式参数驱动，而不是靠手改脚本。
"""

import argparse
from data.combined_dataset import CombinedDataset
from src.model import ResidualDiffusion, UnetRes, set_seed
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run FoundIR inference on a paired restoration dataset")
    parser.add_argument("--dataroot", type=str, required=True, help="Dataset root containing LQ/GT or paths referenced by meta")
    parser.add_argument("--meta", type=str, default=None, help="Optional meta file describing GT/LQ pairs")
    parser.add_argument("--dataset_mode", type=str, default="meta_info", choices=["meta_info", "paired"])
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("--max_dataset_size", type=int, default=int(1e9))
    parser.add_argument("--load_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--direction", type=str, default="AtoB")
    parser.add_argument("--preprocess", type=str, default="none",
                        help="resize_and_crop | crop | scale_width | scale_width_and_crop | none")
    parser.add_argument("--no_flip", action="store_true", help="Disable flip augmentation")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--sampling_timesteps", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--objective", type=str, default="pred_res")
    parser.add_argument("--test_res_or_noise", type=str, default="res")
    parser.add_argument("--sum_scale", type=float, default=0.01)
    parser.add_argument("--ddim_sampling_eta", type=float, default=0.0)
    parser.add_argument("--delta_end", type=float, default=1.4e-3)
    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model-<milestone>.pt")
    parser.add_argument("--checkpoint_milestone", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--crop_phase", type=str, default="im2overlap",
                        choices=["none", "weight", "im2overlap"])
    parser.add_argument("--crop_stride", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def build_dataset(args):
    # 测试入口和训练入口共用同一个数据类，保证数据读取语义一致。
    return CombinedDataset(
        args,
        args.image_size,
        augment_flip=not args.no_flip,
        equalizeHist=True,
        crop_patch=False,
        generation=False,
        task=args.dataset_mode,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    # 推理入口的结构尽量和训练入口保持一致，减少后续维护心智负担。
    dataset = build_dataset(args)
    num_unet = 1
    condition = True

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
        ddim_sampling_eta=args.ddim_sampling_eta,
        objective=args.objective,
        loss_type=args.loss_type,
        condition=condition,
        sum_scale=args.sum_scale,
        test_res_or_noise=args.test_res_or_noise,
    )

    # checkpoint_dir 只表示权重来源；output_dir 单独控制输出位置，避免语义混淆。
    trainer = Trainer(
        diffusion,
        dataset,
        args,
        train_batch_size=args.batch_size,
        num_samples=1,
        train_lr=2e-4,
        train_num_steps=1,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        results_folder=args.checkpoint_dir,
        condition=condition,
        save_and_sample_every=1000,
        num_unet=num_unet,
        data_mode=args.dataset_mode,
    )

    if trainer.accelerator.is_local_main_process:
        trainer.load(args.checkpoint_milestone)
        trainer.set_results_folder(args.output_dir)
        crop_phase = None if args.crop_phase == "none" else args.crop_phase
        trainer.test(
            last=True,
            crop_phase=crop_phase,
            crop_size=args.image_size if crop_phase is not None else None,
            crop_stride=args.crop_stride,
        )


if __name__ == "__main__":
    main()

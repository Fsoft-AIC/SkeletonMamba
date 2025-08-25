import argparse
import os
from pathlib import Path
import yaml
import wandb
import torch
import torch.nn.functional as F
from model.diffusion import GaussianDiffusion
from utils.logger import setup_logger
from trainer import Trainer
import logging
from model.condition import ConditionModule
from SkeletonMamba.model.model_skeletonmamba import SkeletonMamba
from SkeletonMamba.model.model_mamba_vanilla import Mamba_Vanilla
def get_trainer(opt, device):
        # Prepare Directories
    save_dir = Path(opt.exp_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Experiments results will be saved to: {wdir}")
    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Create model
    denoised_model = SkeletonMamba(
                input_dim=opt.input_dim,
                embed_dim=opt.embed_dim,
                depth=opt.depth,
                num_joints=24,
                has_text=True,
                d_context=opt.embed_dim,
                device=device,
                scan_type="sm",
                use_pe=2,
                fused_add_norm=False,
                rms_norm=False,
                video_frames = opt.window,
            )
    condition_module = ConditionModule(
        window=opt.window,
        d_feats=opt.embed_dim,
    )
    logging.info("Loss type: {}".format(opt.loss_type))
    logging.info("Initializing model...")
    diffusion_model = GaussianDiffusion(
        denoise_fn=denoised_model,
        condition_module=condition_module,
        device = device,
        timesteps=1000,
        objective=opt.objective, 
        loss_type=opt.loss_type,
    )
    params = 0
    for p in diffusion_model.parameters():
        if p.requires_grad:
            params += p.numel()
    logging.info("Trainable params: {}".format(params))
    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size,
        train_lr=opt.learning_rate,
        train_num_steps=opt.train_num_steps,
        gradient_accumulate_every=opt.gradient_accumulate_every,
        save_and_sample_every=opt.save_and_sample_every,
        ema_decay=0.995,
        amp=opt.use_amp,
        results_folder=str(wdir),
        device=device
    )

    return trainer 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', default='Experiments/exp1', help='experiment directory')
    parser.add_argument('--device', default='0', help='cuda device')

    ## model arguments
    parser.add_argument('--input_dim', type=int, default=9, help='input dimension')
    parser.add_argument('--embed_dim', type=int, default=256, help='embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='depth')
    parser.add_argument('--window', type=int, default=120, help='window size')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')
    parser.add_argument('--train_num_steps', type=int, default=800000, help='the number of training steps')
    parser.add_argument('--gradient_accumulate_every', type=int, default=2, help='the number of training steps')
    parser.add_argument('--save_and_sample_every', type=int, default=1000, help='save interval')
    parser.add_argument('--use_amp', action="store_true")
    parser.add_argument('--log_interval', type=int, default=1000, help='log interval')
    parser.add_argument("--objective", type=str, default="pred_x0")
    parser.add_argument("--loss_type", type=str, default="l1")
    # For testing sampled results
    parser.add_argument("--test_sample_res", action="store_true")

    # For data representation
    parser.add_argument("--canonicalize_init_head", action="store_true")

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    setup_logger(f"{opt.exp_dir}/log/log-train")
    logging.info(opt)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    trainer = get_trainer(opt, device)
    logging.info("Training...")
    trainer.train()
    torch.cuda.empty_cache()
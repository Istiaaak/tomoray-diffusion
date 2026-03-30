import sys
import os
import torch
from monai.data.meta_tensor import MetaTensor
torch.serialization.add_safe_globals([MetaTensor])
import hydra
from omegaconf import DictConfig, open_dict
from accelerate import Accelerator

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ddpm.diffusion import Unet3D, GaussianDiffusion, Trainer
from features_fusion.fusion import Fusion
from train.get_dataset import get_dataset

@hydra.main(version_base=None, config_path="../config", config_name="base_cfg")
def run_test(cfg: DictConfig):
    
    accelerator = Accelerator(mixed_precision="bf16" if cfg.model.amp else "no")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, 
            cfg.dataset.name, 
            cfg.model.name
        )

    latent_size = cfg.model.diffusion_img_size
    latent_channels = cfg.model.diffusion_num_channels
    fusion_channels = 128

    fusion_model = Fusion(
        volume_shape=(latent_size, latent_size, latent_size),
        vol_size_mm=cfg.model.vol_size_mm,
        det_size_mm=cfg.model.det_size_mm,
        n_features=fusion_channels
    )

    unet = Unet3D(
        dim=64,
        dim_mults=cfg.model.dim_mults,
        channels=latent_channels,
        cond_channels=fusion_channels
    )

    diffusion = GaussianDiffusion(
        unet,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=latent_size,
        num_frames=latent_size,
        channels=latent_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type
    )

    train_dataset, val_dataset, test_dataset = get_dataset(cfg)

    trainer = Trainer(
        diffusion_model=diffusion,
        fusion_model=fusion_model,
        cfg=cfg,
        dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        accelerator=accelerator,
        train_batch_size=cfg.model.batch_size,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        save_and_sample_every=cfg.model.save_and_sample_every,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        debug_overfit=False
    )

    if accelerator.is_main_process:
        trainer.test(milestone=95)

if __name__ == '__main__':
    run_test()
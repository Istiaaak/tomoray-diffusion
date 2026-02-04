import sys
import os
import torch
import hydra
from omegaconf import DictConfig, open_dict


from ddpm.diffusion import Unet3D, GaussianDiffusion, Trainer
from features_fusion.fusion import Fusion
from get_dataset import get_dataset

@hydra.main(version_base=None, config_path="./config", config_name="base_cfg")
def run(cfg: DictConfig):

    if torch.cuda.is_available():

        torch.cuda.set_device(cfg.model.gpus)
        print(f"Device is set to GPU:{cfg.model.gpus}")
    else:
        print("Cuda is not available")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, 
            cfg.dataset.name, 
            cfg.model.name
        )
    
    print(f"Results folder: {cfg.model.results_folder}")
    os.makedirs(cfg.model.results_folder, exist_ok=True)

    latent_size = cfg.model.diffusion_img_size
    latent_channels = cfg.model.diffusion_num_channels
    fusion_channels = 128

    fusion_model = Fusion(
        volume_shape=(latent_size, latent_size, latent_size),
        vol_size_mm=cfg.model.vol_size_mm,
        det_size_mm=cfg.det_size_mm,
        n_features=fusion_channels
    ).cuda()

    print("Initialization of 3D U-Net")
    unet = Unet3D(
        dim=64,
        dim_mults=cfg.model.dim_mults,
        channels=latent_channels,
        cond_channels=fusion_channels
    ).cuda()

    print("Initialization of diffusion")
    diffusion = GaussianDiffusion(
        unet,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=latent_size,
        num_frames=latent_size,
        channels=latent_channels,
        timesteps=cfg.model.timesteps,
        loss_type='l2'

    ).cuda()



    train_dataset, val_dataset, _ = get_dataset(cfg)

    trainer = Trainer(
        diffusion_model=diffusion,
        fusion_model=fusion_model,
        cfg=cfg,
        dataset=train_dataset,
        val_dataset=val_dataset,

        train_batch_size=cfg.model.batch_size,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        save_and_sample_every=cfg.model.save_and_sample_every,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,

    )

    if cfg.model.load_milestone != -1:
         trainer.load(cfg.model.load_milestone)
    elif cfg.model.load_milestone == -1 and os.path.exists(os.path.join(cfg.model.results_folder, 'checkpoints')):
         trainer.load(-1) # Auto-resume

    print("Starting Training...")
    trainer.train()

if __name__ == '__main__':
    run()


    
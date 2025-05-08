from omegaconf import DictConfig, OmegaConf
from data import ImageNetDataset
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
import torch
from models import DiT
import wandb
import hydra
import numpy as np
import torch
from utils import add_noise_to_latent_rep
import torch.nn.functional as F
from pathlib import Path

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set wandb
    if cfg.wandb.enable:
        wandb.init(
            project="diffusion-training",
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.use_gpu_if_available else "cpu")

    # Load all modules used
    dit = DiT(embedding_dim=384, num_heads=6, dit_depth=12, patch_dim=16) # TODO: Maybe port these to Hydra as well?
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.eval()
    train_dataloader = DataLoader(ImageNetDataset(data_path=Path(cfg.data.path + "/train")), batch_size=cfg.train.batch_size,
                        shuffle=True, num_workers=0)
    val_dataloader = DataLoader(ImageNetDataset(data_path=Path(cfg.data.path + "/val")), batch_size=cfg.train.batch_size,
                    shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(dit.parameters(), lr=cfg.train.lr)

    # First get alphas, betas, and alpha_cumprod
    betas = torch.linspace(start = 0.0001, end = 0.02, steps=cfg.train.timesteps) # Choose noise param
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Start train loop
    print(f"Training for {cfg.train.epochs}")
    best_val_loss = float('inf')
    for itr in range(cfg.train.epochs):
        dit.train()
        # Sample from dataloader
        total_train_loss = 0.0
        for sample in train_dataloader:
            imgs = sample['img']
            labels = sample['label']
            ts = torch.randint(low=0, high=cfg.train.timesteps, size =(cfg.train.batch_size,))
            
            # Get noised latent from image 
            with torch.no_grad():
                z_0 = vae.encode(imgs).latent_dist.sample() * 0.18215
                # adding noise
                z_t, epsilon = add_noise_to_latent_rep(ts=ts, alphas_cumprod=alphas_cumprod, z_0=z_0)
            
            pred_noise, log_var = dit(z_t, labels, ts)
            train_loss = ((0.5*torch.exp(-log_var))*F.mse_loss(input=pred_noise, target=epsilon, reduction='none') + 0.5*log_var).mean()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()

        if itr % cfg.val.freq == 0:
            total_val_loss = 0.0
            dit.eval()
            for sample in val_dataloader:
                imgs = sample['img']
                labels = sample['label']
                ts = torch.randint(low=0, high=cfg.train.timesteps, size =(cfg.train.batch_size,))
                
                # Get noised latent from image 
                with torch.no_grad():
                    z_0 = vae.encode(imgs).latent_dist.sample() * 0.18215
                    # adding noise
                    z_t, epsilon = add_noise_to_latent_rep(ts=ts, alphas_cumprod=alphas_cumprod, z_0=z_0)
                
                pred_noise, log_var = dit(z_t, labels, ts)
                val_loss = ((0.5*torch.exp(-log_var))*F.mse_loss(input=pred_noise, target=epsilon, reduction='none') + 0.5*log_var).mean()
                total_val_loss += val_loss.item()
            
            total_train_loss /= len(train_dataloader)
            total_val_loss /= len(val_dataloader)
            if total_val_loss < best_val_loss:
                # Save model
                torch.save(dit.state_dict(), f"epoch_{itr}_loss_{round(total_val_loss,2)}.pth")
            print(f"[{itr+1}/{cfg.train.epochs}] Train_loss:{train_loss}, Val_loss:{total_val_loss}")

        if cfg.wandb.enable:
            wandb.log({"train_loss": train_loss.item(), "val_loss": total_val_loss.item(), "step": itr})

    wandb.finish()
    
if __name__ == '__main__':
    main()
    # patchifyer = ImagePatchifyer()
    # data = ImageNetDataset()
    # context_emb = ContextEmbeddings()
    # vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    # vae.eval()
    # dataloader = DataLoader(data, batch_size=1,
    #                     shuffle=True, num_workers=0)
    # next_img_sample = next(iter(dataloader))    
    # img = next_img_sample["img"]
    # label = next_img_sample["label"]
    # t = torch.Tensor([1])
    # context = context_emb(label, t)
    # with torch.no_grad():
    #     # How to get latent dim
    #     z = vae.encode(img).latent_dist.sample() * 0.18215
    # img_embedding = patchifyer(z)
    # B, T, C = img_embedding.shape
    # dit = DiT(embedding_dim=C, num_heads=6, dit_depth=12, patch_dim=16)
    # output = dit(z, label, t)
    # print(output[0].shape)

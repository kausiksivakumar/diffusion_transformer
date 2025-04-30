import torch
import numpy as np
from utils import get_sinusoidal_position_embeddings

class ImagePatchifyer(torch.nn.Module):
    
    def __init__(self, latent_rep_div: int = 8) -> None:
        # TODO: I need to use a pretrained autoencoder from stable diffusion and pass that as latent
        super().__init__()
        '''
        Assumes all images are of size 256 x 256
        '''
        assert 256% latent_rep_div == 0, "We need the image dims to be divisible by patch size for effective pathification"
        patch_size = 2 # hardcoded for now, we are only working with patch_size = 2 for this reimplementation
        
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size) 

        self.linear_layer = torch.nn.Linear(16, 384) # d -> 384 (size of each token embedding)

    def forward(self, latent_z: torch.Tensor) -> torch.Tensor: 
        assert latent_z.shape[1:] == (4, 32, 32), f"Image's latent dimensions must be 4x32x32 in this implementation but got {latent_z.shape[1:]}"
        # latent_rep = self.latent_conv(latent_z) # (B, 4, 32, 32)        
        # Now get patches of size 2
        latent_img_patches = self.unfold(latent_z).permute(0,2,1) # (B, 256, 16)
        img_tokens = self.linear_layer(latent_img_patches) # (B, T , d) --> (B, T, 384)
        
        # TODO: Add position embeddings
        B, T, d = img_tokens.shape
        with torch.no_grad():
            # No need for gradient of position embeddings 
            pos_embeddings = get_sinusoidal_position_embeddings(token_dim=T, embedding_dim=d)
            img_tokens = img_tokens + pos_embeddings
        return img_tokens


if __name__ == '__main__':
    from data import ImageNetDataset
    from torch.utils.data import DataLoader
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.eval()

    patchifyer = ImagePatchifyer()
    data = ImageNetDataset()
    dataloader = DataLoader(data, batch_size=1,
                        shuffle=True, num_workers=0)
    next_img_sample = next(iter(dataloader))
    img = next_img_sample["img"]
    with torch.no_grad():
        # How to get latent dim
        z = vae.encode(img).latent_dist.sample() * 0.18215

    output = patchifyer(z)
    print(output.shape)

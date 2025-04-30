import torch
import numpy as np
from utils import get_sinusoidal_position_embeddings, get_timestep_embedding
import torch.nn.functional as F
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

class TimeStepEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 128 is the timestep embedding (non learned), the mlp projection adds non-linearity
        self._linear_layer_timestep = torch.nn.Linear(128, 384)
        self.timestep_embed = torch.nn.Sequential(
                            torch.nn.Linear(128, 384),
                            torch.nn.SiLU(),
                            torch.nn.Linear(384, 384),
                        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            timestep__embedding = get_timestep_embedding(t=t) # (B, 128)
        timestep__embedding = self._linear_layer_timestep(timestep__embedding) # (B, d)
        return timestep__embedding

class ContextEmbeddings(torch.nn.Module):
    '''
    Concatenating class and time embedding for cross attention
    '''
    def __init__(self) -> None:
        super().__init__()
        self.timestep_embedding = TimeStepEmbedding()
        self.class_embedding    = torch.nn.Embedding(1000, 384)

    def forward(self, label:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        time_embedding = self.timestep_embedding(t).unsqueeze(1) # (B,1, d)
        class_embedding =   self.class_embedding(label).unsqueeze(1) # (B,1, d)
        context_embeddings = torch.cat([time_embedding, class_embedding], dim = 1) # (B, 2, d)
        return context_embeddings

class SelfAttention(torch.nn.Module):
    '''
    Just one block of self attention
    '''
    def __init__(self, embedding_dim: int, head_size:int) -> None:
        super().__init__()
        self.q = torch.nn.Linear(embedding_dim, head_size)
        self.k = torch.nn.Linear(embedding_dim, head_size)
        self.v = torch.nn.Linear(embedding_dim, head_size)
    
    def forward(self, img_embeddings: torch.Tensor) -> torch.Tensor:
        query_tensor = self.q(img_embeddings) # (B, T, head_size)
        key_tensor  = self.k(img_embeddings) # (B, T, head_size)
        value_tensor = self.v(img_embeddings) # (B, T, head_size)
        B, T, head_size = query_tensor.shape
        wei = (query_tensor @ key_tensor.permute(0,2,1)) * (head_size**-0.5) # for normalizing softmax
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        output = wei @ value_tensor # (B, T, head_size)
        return output

class DiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

if __name__ == '__main__':
    from data import ImageNetDataset
    from torch.utils.data import DataLoader
    from diffusers.models import AutoencoderKL
    patchifyer = ImagePatchifyer()
    data = ImageNetDataset()
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.eval()
    dataloader = DataLoader(data, batch_size=1,
                        shuffle=True, num_workers=0)
    next_img_sample = next(iter(dataloader))    
    img = next_img_sample["img"]
    with torch.no_grad():
        # How to get latent dim
        z = vae.encode(img).latent_dist.sample() * 0.18215
    img_embedding = patchifyer(z)

    self_attention = SelfAttention(384, 128)
    attention_output = self_attention(img_embedding)
    print(attention_output.shape)

    # output = patchifyer(z)

    # ctx_embedding = ContextEmbeddings()
    # label = torch.tensor([0,1], dtype=torch.int32)
    # t = torch.tensor([3,2])
    # emb = ctx_embedding(label,t)
    # print(emb.shape)


    # from data import ImageNetDataset
    # from torch.utils.data import DataLoader
    # from diffusers.models import AutoencoderKL
    # vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    # vae.eval()

    # patchifyer = ImagePatchifyer()
    # data = ImageNetDataset()
    # dataloader = DataLoader(data, batch_size=1,
    #                     shuffle=True, num_workers=0)
    # next_img_sample = next(iter(dataloader))
    # img = next_img_sample["img"]
    # with torch.no_grad():
    #     # How to get latent dim
    #     z = vae.encode(img).latent_dist.sample() * 0.18215

    # output = patchifyer(z)
    # print(output.shape)

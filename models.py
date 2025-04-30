import torch
import numpy as np

class ImagePatchifyer(torch.nn.Module):
    
    def __init__(self, latent_rep_div: int = 8) -> None:
        super().__init__()
        '''
        Assumes all images are of size 256 x 256
        '''
        assert 256% latent_rep_div == 0, "We need the image dims to be divisible by patch size for effective pathification"
        patch_size = 2 # hardcoded for now, we are only working with patch_size = 2 for this reimplementation
        
        self.latent_conv = torch.nn.Conv2d(in_channels=3, 
                                          out_channels=4, 
                                          kernel_size=latent_rep_div, 
                                          stride=latent_rep_div)
        
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size) 

        self.linear_layer = torch.nn.Linear(16, 384) # d -> 384 (size of each token embedding)

    def forward(self, img: torch.Tensor) -> torch.Tensor: 
        assert img.shape[1:] == (3, 256, 256), f"Image dimensions must be 256x256x3 in this implementation but got {img.shape[1:]}"
        latent_rep = self.latent_conv(img) # (B, 4, 32, 32)        
        # Now get patches of size 2
        latent_img_patches = self.unfold(latent_rep).permute(0,2,1) # (B, 256, 16)
        img_tokens = self.linear_layer(latent_img_patches) # (B, T , d) --> (B, T, 384)

        # TODO: Add position embeddings
        return img_tokens


if __name__ == '__main__':
    from data import ImageNetDataset
    from torch.utils.data import DataLoader
    patchifyer = ImagePatchifyer()
    data = ImageNetDataset()
    dataloader = DataLoader(data, batch_size=1,
                        shuffle=True, num_workers=0)
    next_img_sample = next(iter(dataloader))
    img = next_img_sample["img"]
    output = patchifyer(img)


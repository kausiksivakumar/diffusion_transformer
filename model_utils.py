import torch
import numpy as np

class ImagePatchifyer(torch.nn.Module):
    
    def __init__(self, patch_div: int = 8) -> None:
        super().__init__()
        '''
        Assumes all images are of size 256 x 256
        '''
        assert 256% patch_div == 0, "We need the image dims to be divisible by patch size for effective pathification"
        patch_h = 256//patch_div
        patch_w = 256//patch_div
        
        self.patch_conv = torch.nn.Conv2d(in_channels=3, 
                                          out_channels=4, 
                                          kernel_size=(patch_h, patch_w), 
                                          stride=patch_h)

    def forward(self, img: np.ndarray) -> torch.Tensor: 

        pass
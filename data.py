from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Union
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
DATA_PATH = "/net/acadia15a/data/kausik/imagenet_tiny"

def get_all_jpegs_in_path(root_folder: Path) -> List[Path] :
    root = Path(root_folder)
    jpeg_files = list(root.rglob('*.jpg')) + list(root.rglob('*.JPEG'))
    return jpeg_files

class ImageNetDataset(Dataset):
    "ImageNet mini dataset"
    def __init__(self, 
                 data_path: Path = Path(DATA_PATH), 
                 transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])) -> None:
        super().__init__()
        self._all_jpegs_path =   get_all_jpegs_in_path(root_folder=data_path)
        self._all_parent_name = sorted(list(set([path.parent.name for path in self._all_jpegs_path]))) # Gives 1000 labels
        self._parent_name_to_class_idx =  {s: i for i, s in enumerate(self._all_parent_name)}
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self._all_jpegs_path)

    def __getitem__(self, idx: Union[torch.Tensor, List, int]) -> dict[str, Union[np.ndarray, torch.Tensor]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self._all_jpegs_path[idx]
        label   =   torch.tensor([self._parent_name_to_class_idx[img_path.parent.name]], dtype=torch.int32)
        img     =   cv2.imread(str(img_path))
        if self.transform:
            img_pil = Image.fromarray(img.astype('uint8'))
            img = self.transform(img_pil)
        sample = {'img': img, 'label': label}

        return sample 


if __name__ == '__main__':
    img_tranform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = ImageNetDataset(transform=img_tranform)
    for idx in range(5):
        img = dataset[idx]["img"]
        label = dataset[idx]["label"]
        if isinstance(img, torch.Tensor):
            img = img.permute(1,2,0).numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
        print(f"Saving image with shape {img.shape}")
        cv2.imwrite(f"tmp/{idx}_{label.item()}.png", img)


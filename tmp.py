from pathlib import Path
import cv2 
import torch 
# data_path = Path("/home/ma/kausik/workspace/datasets/ImageNet")

# train_path = data_path / "train"
# val_path = data_path / "val"

# def get_all_jpegs(root_folder):
#     root = Path(root_folder)
#     jpeg_files = list(root.rglob('*.jpg')) + list(root.rglob('*.JPEG'))
#     return jpeg_files

# jpegs = get_all_jpegs(root_folder=train_path)
# print(f"There are {len(jpegs)} image files")

# # read one image from jpegs
# test_img = 4
# img = cv2.imread(str(jpegs[test_img]))
# resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
# print(f"Image dims is {resized_img.shape}")
# cv2.imshow("test_resized_img", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Try to unfold with pytorch 
random_img = torch.Tensor([[1,1],[1,1]])
hor_stacked_img = torch.hstack((random_img, 2*random_img))
full_img = torch.vstack((hor_stacked_img, 3*hor_stacked_img)).unsqueeze(0).unsqueeze(0)
full_img = full_img.repeat(1,4,1,1) # (1, 4, 4, 4) -> (B, C, H, W)
unfold = torch.nn.Unfold((2,2), stride = (2,2)) # Does it column wise
unfold(full_img)

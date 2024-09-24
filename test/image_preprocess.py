from utils.loader import get_function_from_comfyui
from utils.image_process import resize_image_for_sd
from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from ComfyUI.custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.normalbae import NormalBaeDetector
from ComfyUI.custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.zoe import ZoeDetector
from ComfyUI.custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.oneformer import OneformerSegmentor

from ComfyUI.custom_nodes.comfyui_controlnet_aux.utils import common_annotator_call

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.filenames = [fname for fname in os.listdir(image_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]
        self.filenames.sort()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image_resize_and_cropped = resize_image_for_sd(image)
        image_resize_and_cropped_tensor = self.transform(image_resize_and_cropped)
        return image_resize_and_cropped_tensor.permute(1, 2, 0), filename

device = torch.device("cuda")

root = '/home/gkalstn000/type_4'
preprocess_type = {'seg_ofcoco': (OneformerSegmentor.from_pretrained(filename="150_16_swin_l_oneformer_coco_100ep.pth"), 'seg_map'),
                   'depth': (ZoeDetector.from_pretrained(), 'depth_map'), # ZoeD_M12_N.pt
                   'normalmap_bae': (NormalBaeDetector.from_pretrained(), 'normal_map'),

                   }

transform = transforms.Compose([
    transforms.ToTensor(),
])

bucket_root = os.path.join(root, 'bucket')
batch_size = 1

dataset = CustomImageDataset(image_dir=os.path.join(root, 'images'), transform=transform)
image_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

for process_type, (preprocessor, save_name) in preprocess_type.items():
    save_root = os.path.join(root, save_name)
    os.makedirs(save_root, exist_ok=True)
    for batch_idx, (images, filename) in enumerate(tqdm(image_loader)):
        b, h, w, c = images.size()
        save_path = os.path.join(save_root, filename[0])
        if os.path.exists(save_path): continue
        preprocessor = preprocessor.to(device)
        images = images.to(device)

        image_processed_tensor = common_annotator_call(preprocessor, images, resolution=512).squeeze()
        image_processed_tensor = (image_processed_tensor.detach().cpu() * 255).numpy().astype(np.uint8)
        image_processed = Image.fromarray(image_processed_tensor).resize((w, h))
        image_processed.save(save_path)

    del preprocessor
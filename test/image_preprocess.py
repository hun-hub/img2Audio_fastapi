from cgen_utils import set_comfyui_packages
set_comfyui_packages()
from cgen_utils.image_process import resize_image_for_sd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
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

from multiprocessing.dummy import Pool as ThreadPool  # 멀티스레딩용 Pool


class CustomImageDataset(Dataset):
    def __init__(self, filepaths, transform=None):
        self.transform = transform
        self.filepaths = filepaths
        self.filepaths.sort()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        image = Image.open(filepath).convert("RGB")
        image_resize_and_cropped = resize_image_for_sd(image)
        image_resize_and_cropped_tensor = self.transform(image_resize_and_cropped)
        filename = filepath.split("/")[-1]
        return image_resize_and_cropped_tensor.permute(1, 2, 0), filename


transform = transforms.Compose([
    transforms.ToTensor(),
])


root = '/home/gkalstn000/coco_2014'
image_root = os.path.join(root, 'val2014_img')
image_bucket = defaultdict(list)

def process_image(filepath):
    try:
        image = Image.open(filepath).convert("RGB")
        image_resize_and_cropped = resize_image_for_sd(image)
        w, h = image_resize_and_cropped.size
        return (w, h, filepath)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None  # 에러 발생 시 None 반환

filepaths = [os.path.join(image_root, filename) for filename in os.listdir(image_root)]

pool = ThreadPool(os.cpu_count() * 2)

# 멀티스레딩으로 이미지 처리
results = []
for result in tqdm(pool.imap_unordered(process_image, filepaths), total=len(filepaths)):
    if result is not None:
        w, h, filepath = result
        image_bucket[(w, h)].append(filepath)

pool.close()
pool.join()

dataloader_list = []
batch_size = 20
for (w, h), filepaths in image_bucket.items():
    dataset = CustomImageDataset(filepaths=filepaths, transform=transform)
    image_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 8)
    dataloader_list.append(image_loader)

preprocess_type = {
    'depth': (ZoeDetector.from_pretrained(), 'depth_map'),
    'seg_ofcoco': (OneformerSegmentor.from_pretrained(filename="150_16_swin_l_oneformer_coco_100ep.pth"), 'seg_map'),
    'normalmap_bae': (NormalBaeDetector.from_pretrained(), 'normal_map'),
                   }
device = torch.device("cuda")

def process_and_save(args):
    imaeg_tensor, filename, save_root, w, h = args
    save_path = os.path.join(save_root, filename)
    image_processed = Image.fromarray(imaeg_tensor).resize((w, h))
    image_processed.save(save_path)

def do_jump(save_root, filenames) :
    for filename in filenames:
        save_path = os.path.join(save_root, filename)
        if not os.path.exists(save_path):
            return False
    return True
# 최상위 수준에서 Pool 생성
pool = ThreadPool(os.cpu_count() * 2)

for process_type, (preprocessor, save_name) in preprocess_type.items():
    save_root = os.path.join(root, save_name)
    os.makedirs(save_root, exist_ok=True)
    preprocessor = preprocessor.to(device)
    for image_loader in tqdm(dataloader_list):
        for batch_idx, (images, filenames) in enumerate(tqdm(image_loader)):
            if do_jump(save_root, filenames) : continue
            b, h, w, c = images.size()
            images = images.to(device)

            image_processed_tensor = common_annotator_call(preprocessor, images, resolution=512).squeeze()
            image_processed_tensor = (image_processed_tensor.detach().cpu() * 255).numpy().astype(np.uint8)
            args_list = [(imaeg_tensor, filename, save_root, w, h)
                         for imaeg_tensor, filename in zip(image_processed_tensor, filenames)]

            pool.map(process_and_save, args_list)
    del preprocessor

pool.close()
pool.join()
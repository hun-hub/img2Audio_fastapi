import numpy as np
import io, base64
import torch
from PIL import Image
from cgen_utils.loader import load_controlnet_preprocessor


def convert_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return image_base64

def convert_base64_to_image(image_base64):
    image_data = base64.b64decode(image_base64)
    image_rgb = Image.open(io.BytesIO(image_data))
    return image_rgb
def convert_image_array_to_base64(image_array: np.ndarray) -> str:
    '''
    image_array size (H, W, 3)
    image_array range [0, 255]
    '''
    image_rgb = Image.fromarray(image_array.astype(np.uint8))
    image_base64 = convert_image_to_base64(image_rgb)
    return image_base64

def convert_base64_to_image_array(image_base64: str) -> np.ndarray:
    '''
    image_array size (H, W, 3)
    image_array range [0, 255]
    '''
    image_rgb = convert_base64_to_image(image_base64)
    image_array = np.array(image_rgb)
    return image_array

def convert_image_tensor_to_base64(image_tensor: torch.Tensor) -> str:
    '''
    image_tensor size (H, W, 3)
    image_tensor range [0, 255]
    '''
    image_array = image_tensor.squeeze().cpu().detach().numpy().astype(np.uint8)
    image_base64 = convert_image_array_to_base64(image_array)
    return image_base64

def convert_base64_to_image_tensor(image_base64: str) -> torch.Tensor:
    '''
    image_tensor size (H, W, 3)
    image_tensor range [0, 255]
    '''
    image_array = convert_base64_to_image_array(image_base64)
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    return image_tensor.unsqueeze(0)


def _crop_image(image: Image):
    w, h = image.size

    h_ = h - h % 64
    w_ = w - w % 64

    # 중앙을 기준으로 크롭할 영역 계산
    left = (w - w_) // 2
    top = (h - h_) // 2
    right = left + w_
    bottom = top + h_
    image = image.crop((left, top, right, bottom))
    return image

def resize_image_for_sd(image: Image, is_mask=False, resolution = 1024) :
    w, h = image.size
    scale = (resolution ** 2 / (w * h)) ** 0.5

    scale = 1 if scale > 1 else scale
    w_scaled = w * scale
    h_scaled = h * scale
    interpolation = Image.NEAREST if is_mask else Image.BICUBIC
    image_resized = image.resize((int(w_scaled), int(h_scaled)), interpolation)
    image_resized_cropped = _crop_image(image_resized)
    return image_resized_cropped


def controlnet_image_preprocess(images, preprocessor_type, sd_version, resolution=512) :
    preprocessor = load_controlnet_preprocessor()
    image_preprocessed, _ = preprocessor.detect_controlnet(images, preprocessor_type, sd_version=sd_version, resolution=resolution)
    return image_preprocessed

if __name__ == '__main__':
    image = ''
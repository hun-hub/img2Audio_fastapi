import requests
from cgen_utils.handler import handle_response
from cgen_utils.image_process import (convert_base64_to_image_array,
                                      convert_image_to_base64,
                                      )
from types import NoneType
from PIL import Image
import torch
import numpy as np

def tensor_to_pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def normalize_mask(mask_tensor):
    max_val = torch.max(mask_tensor)
    min_val = torch.min(mask_tensor)

    if max_val == min_val:
        return mask_tensor

    normalized_mask = (mask_tensor - min_val) / (max_val - min_val)

    return normalized_mask

def sned_nukki_request_to_api(
        model_name,
        image,
        nukki,
        edit_mode,

        ip_addr
) :
    background, layers, composite = nukki.values()
    if background.sum() != 0:
        image = background[:, :, 0]
        mask = layers[0][:, :, 3]

        image = convert_image_to_base64(Image.fromarray(image))
        mask = convert_image_to_base64(Image.fromarray(mask))
        request_body = {
            'image': image,
            'mask': mask,
            'edit_mode': edit_mode
        }

        url = f"http://{ip_addr}/functions/mask_edit"
    else :
        if isinstance(image, NoneType):
            raise ValueError("image가 None입니다. 올바른 이미지 객체를 전달하세요.")

        image = convert_image_to_base64(Image.fromarray(image))

        request_body = {
            'nukki_model': model_name,
            'init_image': image,
        }

        url = f"http://{ip_addr}/nukki"

    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image = convert_base64_to_image_array(image_base64)
    return [image]

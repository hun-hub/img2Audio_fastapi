from cgen_utils import set_comfyui_packages
set_comfyui_packages()

import requests
from cgen_utils.image_process import (convert_base64_to_image_array,
                                      convert_image_array_to_base64,
                                      resize_image_for_sd,
                                      convert_image_to_base64,
                                      convert_base64_to_image_tensor,
                                      controlnet_image_preprocess
                                      )
from cgen_utils.handler import handle_response
import os
from PIL import Image
import numpy as np
from itertools import product
from tqdm import tqdm

image_root = '/home/gkalstn000/datasets/option_exp'
image = Image.open(os.path.join(image_root, 'image.png')).convert('RGB')
mask = Image.open(os.path.join(image_root, 'mask.png')).convert('RGB')

image = resize_image_for_sd(image)
image = convert_image_to_base64(image)
mask = resize_image_for_sd(mask, is_mask=True)
mask = convert_image_to_base64(mask)

prompt = 'A photograph of a crispy, golden-brown pork cutlet (Tonkatsu) on a sleek white marble table against a minimalist solid color backdrop.  A polished fork and knife are placed elegantly beside the Tonkatsu, ready for a delicious meal. The scene is captured in high-quality, realistic detail. '


denoise_range = [0.2, 0.25, 0.3]
model_range = [ 'SDXL_copaxPhotoxl_v2.safetensors']
controlnet_range = ['SDXL_Canny.safetensors', 'SDXL_Canny_sai_xl_canny_256lora.safetensors']

out_iteration = len(denoise_range) * len(model_range) * len(controlnet_range)

for model, controlnet, denoise in tqdm(product(model_range, controlnet_range, denoise_range), total=out_iteration):
    save_name = f"{model.replace('.safetensors', '')}_{controlnet.replace('.safetensors', '')}_{denoise}"
    save_root = os.path.join(image_root, save_name)
    os.makedirs(save_root, exist_ok=True)

    strength_range = np.arange(0.05, 1.05, 0.05)
    end_percent_range = np.arange(0.05, 1.05, 0.05)
    total_iterations = len(strength_range) * len(end_percent_range)

    for end_percent, strength in tqdm(product(end_percent_range, strength_range), total=total_iterations):
        save_path = os.path.join(save_root, f'end_percent_{int(round(end_percent * 100)):03d}_strength_{int(round(strength * 100)):03d}.png')
        if os.path.exists(save_path):
            continue

        strength = round(strength, 2)
        end_percent = round(end_percent, 2)

        request_body = {
            'checkpoint': model,
            'init_image': image,
            'mask': mask,
            "prompt_positive": prompt,
            "prompt_negative": "",
            'width': 1024,
            'height': 1024,
            'steps': 20,
            'cfg': 7,
            'denoise': denoise,
            'seed': 6481362365,
            'controlnet_requests': []
        }

        canny_body = {
            'controlnet': controlnet,
            'type': 'canny',
            'image': mask,
            'preprocessor_type': 'canny',
            'strength': strength,
            'start_percent': 0,
            'end_percent': end_percent,
        }
        request_body['controlnet_requests'].append(canny_body)

        url = f"http://117.52.72.82:7861/sdxl/half_inpainting"
        try :
            response = requests.post(url, json=request_body)
            data = handle_response(response)
            image_base64 = data['image_base64']
            output = convert_base64_to_image_array(image_base64)
            output = Image.fromarray(output)

            output.save(save_path)
        except:
            print(f'end_percent: {end_percent}, strength:{strength} failed')


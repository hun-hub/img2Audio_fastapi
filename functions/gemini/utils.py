from utils.image_process import (convert_base64_to_image_array,
                                 convert_image_array_to_base64,
                                 resize_image_for_sd,
                                 convert_image_to_base64,
                                 convert_base64_to_image_tensor,
                                 convert_image_tensor_to_base64, convert_base64_to_image
                                 )
from types import NoneType
from PIL import Image
import requests
import numpy as np
import os
import torch
from utils.handler import handle_response

def send_gemini_request_to_api(
        query_type,
        image = None,
        user_prompt = '',
        object_description = '',
        background_description = '',
) :
    if isinstance(image, torch.Tensor):
        image_base64 = convert_image_tensor_to_base64(image)
        image = convert_base64_to_image(image_base64)
    if isinstance(image, np.ndarray):
        image_base64 = convert_image_array_to_base64(image)
        image = convert_base64_to_image(image_base64)
    if not isinstance(image, NoneType):
        image = resize_image_for_sd(image)
        image = convert_image_to_base64(image)

    request_body = {
        'user_prompt': user_prompt,
        'object_description': object_description,
        'background_description': background_description,
        'query_type': query_type,
        'image': image
    }

    gamini_addr = os.getenv('GEMINI_ADDR')
    url = f"http://{gamini_addr}/gemini"

    response = requests.post(url, json=request_body)
    data = handle_response(response)
    prompt = data['prompt']
    return prompt


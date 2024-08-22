from utils.image_process import (convert_base64_to_image_array,
                                 convert_image_array_to_base64,
                                 resize_image_for_sd,
                                 convert_image_to_base64,
                                 convert_base64_to_image_tensor
                                 )
from types import NoneType
from PIL import Image
import requests
from utils.handler import handle_response

def send_gemini_request_to_api(
        user_image,
        query,
        user_prompt,

        ip_addr
) :
    if not isinstance(user_image, NoneType):
        user_image = resize_image_for_sd(Image.fromarray(user_image))
        user_image = convert_image_to_base64(user_image)

    request_body = {
        'user_prompt': user_prompt,
        'query': query,
        'user_image': user_image
    }

    url = f"http://{ip_addr}:7861/gemini"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    prompt = data['prompt']
    return prompt
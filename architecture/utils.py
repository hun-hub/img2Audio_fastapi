from types import NoneType
from PIL import Image
import requests
from utils.handler import handle_response
from utils.image_process import (convert_base64_to_image_array,
                                 convert_image_array_to_base64,
                                 resize_image_for_sd,
                                 convert_image_to_base64,

                                 )


# TODO: Gemini만 폴더없이 따로 되어있어서 구조가 통일되지 않음.
def gemini_api(image_array, query, user_prompt, ip_addr) :
    image_bases64 = None
    if not isinstance(image_array, NoneType) :
        image = Image.fromarray(image_array)
        image_resized = resize_image_for_sd(image)
        image_bases64 = convert_image_to_base64(image_resized)
    request_body = {'user_prompt': user_prompt,
                    'query': query,
                    'user_image': image_bases64}

    url = f"http://{ip_addr}:7861/gemini"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    prompt = data['prompt']
    return prompt
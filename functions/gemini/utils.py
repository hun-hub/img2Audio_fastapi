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
        user_prompt,
        object_description,
        background_description,
        query_type,
        image,

        ip_addr
) :
    if not isinstance(image, NoneType):
        image = resize_image_for_sd(Image.fromarray(image))
        image = convert_image_to_base64(image)

    request_body = {
        'user_prompt': user_prompt,
        'object_description': object_description,
        'background_description': background_description,
        'query_type': query_type,
        'image': image
    }

    url = f"http://{ip_addr}:7861/gemini"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    prompt = data['prompt']
    return prompt
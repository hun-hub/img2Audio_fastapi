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
from . import *

query_dict = {'product_description': product_description,
              'image_description': image_description,
              'prompt_refine': prompt_refine,
              'prompt_refine_with_image': prompt_refine_with_image,
              'synthesized_image_description': synthesized_image_description,
              'decompose_background_and_product': decompose_background_and_product,
              'iclight_keep_background': iclight_keep_background,
              'iclight_gen_background': iclight_gen_background}

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

    if query_type not in query_dict: # 자유 형식의 query문
        query = query_type
    else :
        query = query_dict[query_type]
    user_prompt = ', '.join(user_prompt.split(' '))
    query = query.format(user_prompt = user_prompt,
                         object_description = object_description,
                         background_description = background_description)

    request_body = {
        'query': query,
        'image': image
    }

    url = f"http://{ip_addr}:7861/gemini"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    prompt = data['prompt']
    return prompt
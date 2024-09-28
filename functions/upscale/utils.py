import requests
from cgen_utils.handler import handle_response
from cgen_utils.image_process import (convert_base64_to_image_array,
                                      convert_image_to_base64,
                                      )
from types import NoneType
from PIL import Image
import torch

@torch.inference_mode()
def upscale_with_model(upscale_model, image) :
    from ComfyUI.comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
    upsclaer = ImageUpscaleWithModel()
    image_upscaled = upsclaer.upscale(upscale_model, image)[0]
    return image_upscaled
@torch.inference_mode()
def image_scale_by(image, method, scale) :
    from ComfyUI.nodes import ImageScaleBy
    image_scaler = ImageScaleBy()
    image_scaled = image_scaler.upscale(image, method, scale)[0]
    return image_scaled

def sned_upscale_request_to_api(
        model_name,
        image,
        method,
        scale,

        ip_addr
) :
    if isinstance(image, NoneType):
        raise ValueError("inpaint_image가 None입니다. 올바른 이미지 객체를 전달하세요.")

    image = convert_image_to_base64(Image.fromarray(image))

    request_body = {
        'upscale_model': model_name,
        'init_image': image,
        'method': method,
        "scale": scale,
    }

    url = f"http://{ip_addr}/upscale"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image = convert_base64_to_image_array(image_base64)
    return [image]

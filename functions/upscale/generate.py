import torch
from cgen_utils.loader import load_upscaler
from cgen_utils.image_process import convert_image_tensor_to_base64, convert_base64_to_image_tensor, convert_image_array_to_base64
from .utils import upscale_with_model, image_scale_by

@torch.inference_mode()
def upscale(cached_model_dict, request_data):
    upscaler = load_upscaler(request_data.upscale_model)
    init_image = convert_base64_to_image_tensor(request_data.init_image) / 255

    image_upscaled = upscale_with_model(upscaler, init_image)

    scale = request_data.scale / 4

    image_upscaled = image_scale_by(image_upscaled, request_data.method, scale)

    image_base64 = convert_image_tensor_to_base64(image_upscaled * 255)

    return image_base64

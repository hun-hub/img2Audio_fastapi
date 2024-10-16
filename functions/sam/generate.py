import torch
from cgen_utils.image_process import convert_image_tensor_to_base64, convert_base64_to_image_tensor
from cgen_utils.comfyui import (encode_prompt,
                                sample_image,
                                decode_latent,
                                encode_image,
                                )
from cgen_utils.loader import (load_sam,
                               load_dino_model,
                               load_dino_segment_module)
from functions.gemini.utils import send_gemini_request_to_api
import random

@torch.inference_mode()
def predict(cached_model_dict, request_data):
    image = convert_base64_to_image_tensor(request_data.image) / 255

    try:
        sam_model = load_sam(model_name=request_data.sam_model)
    except:
        sam_model = load_sam(model_name=request_data.sam_model)

    dino_model = load_dino_model(model_name=request_data.dino_model)
    segment_module = load_dino_segment_module()

    image, mask = segment_module.main(dino_model, sam_model, image, request_data.prompt, request_data.threshold)
    mask_inv = 1 - mask

    image_base64 = convert_image_tensor_to_base64(image * 255)
    mask_base64 = convert_image_tensor_to_base64(mask * 255)
    mask_inv_base64 = convert_image_tensor_to_base64(mask_inv * 255)

    return image_base64, mask_base64, mask_inv_base64

import torch
from utils.text_process import image_caption, prompt_refine


@torch.inference_mode()
def generate_prompt(request_data):
    if request_data.user_image :
        prompt = image_caption(request_data.query,
                               request_data.user_prompt,
                               request_data.user_image)
    else :
        prompt = prompt_refine(request_data.query,
                               request_data.user_prompt)

    return prompt
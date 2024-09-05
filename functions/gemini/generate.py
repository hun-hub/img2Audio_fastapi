import torch
from utils.text_process import gemini_with_prompt, gemini_with_prompt_and_image


@torch.inference_mode()
def generate_prompt(request_data):
    if request_data.image :
        prompt = gemini_with_prompt_and_image(
            request_data.query,
            request_data.image)
    else :
        prompt = gemini_with_prompt(request_data.query)

    return prompt
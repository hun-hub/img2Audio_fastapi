from types import NoneType

import torch

from .utils import (construct_condition,
                    expand_mask,
                    frequency_separate,
                    frequency_combination,
                    was_image_blend,
                    remap_image)
from utils import set_comfyui_packages
from utils.loader import load_checkpoint
from utils.image_process import convert_image_tensor_to_base64, convert_base64_to_image_tensor
from utils.comfyui import (encode_prompt,
                           sample_image,
                           decode_latent,
                           encode_image,
                           encode_image_for_inpaint,
                           apply_controlnet,
                           controlnet_preprocessor,
                           get_init_noise,
                           mask_blur)
import random


# prompt_post_fix = ", RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
@torch.inference_mode()
def generate_image(cached_model_dict, request_data):
    unet = cached_model_dict['unet']['sd15'][1]
    vae = cached_model_dict['vae']['sd15'][1]
    clip = cached_model_dict['clip']['sd15'][1]

    start_base = int(request_data.steps - request_data.steps * request_data.denoise)
    end_base = request_data.steps

    init_image = convert_base64_to_image_tensor(request_data.init_image) / 255
    mask = convert_base64_to_image_tensor(request_data.mask)[:, :, :, 0] / 255
    mask = mask_blur(mask)
    light = None
    if not isinstance(request_data.light_condition, NoneType) :
        light_strength = int((request_data.light_strength - 0.5) * 200)
        light = convert_base64_to_image_tensor(request_data.light_condition)
        light = expand_mask(light.numpy()[0], light_strength) / 255
        light = torch.Tensor(light).unsqueeze(0)

    if not request_data.keep_background  :
        init_image = torch.where(mask.unsqueeze(-1) > 0.5, 0.5, init_image)
    init_noise = encode_image(vae, init_image)

    if light is not None:
        light_latnet = encode_image(vae, light)
    else :
        b, h, w, c = init_image.size()
        light_latnet = get_init_noise(w, h, 1)

    seed = random.randint(1, int(1e9)) if request_data.seed == -1 else request_data.seed

    positive_cond, negative_cond = encode_prompt(clip,
                                                 request_data.prompt_positive ,
                                                 request_data.prompt_negative)

    unet, positive_cond, negative_cond = construct_condition(unet,
                                                             request_data.iclight_model,
                                                             positive_cond,
                                                             negative_cond,
                                                             init_noise)
    latent_image = sample_image(
        unet= unet,
        positive_cond= positive_cond,
        negative_cond= negative_cond,
        latent_image= light_latnet,
        seed= seed,
        steps= request_data.steps,
        cfg= request_data.cfg,
        sampler_name= request_data.sampler_name,
        scheduler= request_data.scheduler,
        start_at_step= start_base,
        end_at_step= end_base,)

    gen_image = decode_latent(vae, latent_image)

    if request_data.keep_background :
        true_high, true_low = frequency_separate(init_image)
        fake_high, fake_low = frequency_separate(gen_image)

        low_batch = torch.cat([true_low, fake_low], dim=0)
        low_batch_avg = torch.mean(low_batch.detach().clone(), dim=0, keepdim=True)

        blended_image = was_image_blend(low_batch_avg,
                                        true_low,
                                        request_data.blending_mode_1,
                                        request_data.blending_percentage_1)
        blended_image = was_image_blend(blended_image,
                                        true_low,
                                        request_data.blending_mode_2,
                                        request_data.blending_percentage_2)

        gen_image = frequency_combination(true_high, blended_image)
        gen_image = remap_image(gen_image,
                                request_data.remap_min_value,
                                request_data.remap_max_value)

    image_base64 = convert_image_tensor_to_base64(gen_image * 255)
    return image_base64


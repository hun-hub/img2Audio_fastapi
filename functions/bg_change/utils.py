import torch
import requests
from cgen_utils.handler import handle_response
from cgen_utils.image_process import (convert_base64_to_image_array,
                                      convert_image_array_to_base64,
                                      resize_image_for_sd,
                                      convert_image_to_base64,
                                      convert_base64_to_image_tensor,
                                      controlnet_image_preprocess, print_image_tensor
                                      )
from cgen_utils.comfyui import (apply_controlnet,
                                apply_ipadapter,
                                make_image_batch, get_default_args, mask_blur)

from cgen_utils.loader import (load_clip_vision,
                               load_text_encoder_advanced,
                               load_condition_combiner,
                               load_mask_outliner,
                               load_mask_resizer,
                               load_advanced_controlnet_applier, load_timestep_kf)
from ComfyUI.custom_nodes.comfyui_controlnet_aux.node_wrappers.inpaint import InpaintPreprocessor
from types import NoneType
from PIL import Image
import numpy as np

@torch.inference_mode()
def apply_advanced_controlnet(
        positive,
        negative,
        controlnet,
        image,
        strength,
        start_percent,
        end_percent,
        mask_optional,
        tk_shortcut
):
    controlnet_applier = load_advanced_controlnet_applier()

    positive, negative, _ = controlnet_applier.apply_controlnet(
        positive,
        negative,
        controlnet,
        image,
        strength,
        start_percent,
        end_percent,
        mask_optional,
        timestep_kf = tk_shortcut
    )

    return positive, negative

@torch.inference_mode()
def construct_controlnet_condition(
        cached_model_dict,
        positive,
        negative,
        image,
        mask,
        is_retouch,
        is_first,
        controlnet_requests,
):
    try:
        resizer = load_mask_resizer()
    except:
        resizer = load_mask_resizer()

    for controlnet_request in controlnet_requests:
        if controlnet_request.type == 'canny':
            control_image = controlnet_image_preprocess(image, controlnet_request.preprocessor_type, 'sd15')
            control_mask = mask
        elif controlnet_request.type == 'inpaint':
            control_image = image
            control_mask = mask
        elif controlnet_request.type == 'scribble':
            control_image = load_mask_outliner().run(mask, outline_width=8, tapered_corners=True)[0]
            control_image = torch.stack([control_image]*3, -1)
            control_mask = torch.ones_like(mask)
            controlnet_request.strength = 0.5 if is_first else 0.3
        elif controlnet_request.type == 'depth' :
            control_image = controlnet_image_preprocess(image, controlnet_request.preprocessor_type, 'sd15')
            b, h, w, c = control_image.size()

            mask_ = resizer.resize(mask, w, h, False)[0]
            control_image = control_image * mask_.unsqueeze(-1) + torch.zeros_like(control_image) * (1 - mask_.unsqueeze(-1))
            control_mask = None
            if is_retouch:
                mask_ = mask_blur(mask_, amount = 20)
                control_image = torch.where(mask_.unsqueeze(-1) > 0.5, mask_.unsqueeze(-1), control_image)

        controlnet = cached_model_dict['controlnet']['sd15'][controlnet_request.type][1]


        timestep = load_timestep_kf().load_weights(
            controlnet_request.base_multiplier,
            controlnet_request.flip_weights,
            controlnet_request.uncond_multiplier
        )[1]

        positive, negative = apply_advanced_controlnet(
            positive,
            negative,
            controlnet,
            control_image,
            controlnet_request.strength,
            controlnet_request.start_percent,
            controlnet_request.end_percent,
            control_mask,
            timestep
        )


    return positive, negative


@torch.inference_mode()
def encode_prompt_advance(
        clip,
        is_retouch,
        prompt_positive,
        prompt_negative,
        mask,
        prompt_retouch) :
    from server import PromptServer
    PromptServer.instance = None
    text_encoder = load_text_encoder_advanced()

    args = get_default_args(text_encoder.INPUT_TYPES())
    args['text_g'] = 'CLIP_G'
    args['text_l'] = 'CLIP_L'
    args['clip'] = clip
    args['parser'] = 'A1111'
    args['multi_conditioning'] = False

    args['text'] = prompt_positive
    positive_cond = text_encoder.encode(**args)[0]
    args['text'] = prompt_negative
    negative_cond = text_encoder.encode(**args)[0]

    if is_retouch :
        from ComfyUI.nodes import CLIPSetLastLayer, ConditioningSetMask
        clip = CLIPSetLastLayer().set_last_layer(clip, -2)[0]
        args['text'] = prompt_retouch
        args['clip'] = clip
        retouch_cond = text_encoder.encode(**args)[0]
        retouch_cond = ConditioningSetMask().append(retouch_cond, mask, strength=1, set_cond_area='default')[0]
        try :
            condition_combiner = load_condition_combiner()
        except :
            condition_combiner = load_condition_combiner()
        positive_cond = condition_combiner.combine(2, conditioning_1= positive_cond, conditioning_2=retouch_cond)[0]

    return positive_cond, negative_cond
@torch.inference_mode()
def construct_ipadapter_condition(
        unet,
        cached_model_dict,
        ipadapter_request,
):

    if ipadapter_request is not None:
        clip_vision = load_clip_vision(ipadapter_request.clip_vision)
        ipadapter = cached_model_dict['ipadapter']['sd15'][1]
        ipadapter_images = [convert_base64_to_image_tensor(image) / 255 for image in ipadapter_request.images]
        image_batch = make_image_batch(ipadapter_images)

        unet = apply_ipadapter(
                               unet= unet,
                               ipadapter=ipadapter,
                               clip_vision=clip_vision,
                               image= image_batch,
                               weight= ipadapter_request.weight,
                               start_at = ipadapter_request.start_at,
                               end_at = ipadapter_request.end_at,
                               weight_type = ipadapter_request.weight_type,
                               combine_embeds = ipadapter_request.combine_embeds,
                               embeds_scaling= ipadapter_request.embeds_scaling,)
    return unet

def sned_bg_change_request_to_api(
        image,
        mask,
        mask_retouch,
        prompt,
        prompt_retouch,
        do_retouch,

        ipadapter_enable,
        ipadapter_model_name,
        ipadapter_images,
        ipadapter_weight,
        ipadapter_start,
        ipadapter_end,

        ip_addr
) :
    sd_resolution = 1024
    if isinstance(image, NoneType):
        raise ValueError("image가 None입니다. 올바른 이미지 객체를 전달하세요.")
    if isinstance(mask, NoneType) and isinstance(mask_retouch, NoneType):
        raise ValueError("Mask가 None입니다. 올바른 이미지 객체를 전달하세요.")

    image = resize_image_for_sd(Image.fromarray(image), resolution=sd_resolution)
    if do_retouch == 'False' :
        mask = resize_image_for_sd(Image.fromarray(mask), is_mask=True, resolution=sd_resolution)
    else :
        mask = resize_image_for_sd(Image.fromarray(mask_retouch), is_mask=True, resolution=sd_resolution)

    image = convert_image_to_base64(image)
    mask = convert_image_to_base64(mask)

    if not isinstance(ipadapter_images, NoneType):
        ipadapter_images = [resize_image_for_sd(Image.fromarray(ipadapter_image[0]), resolution=sd_resolution) for ipadapter_image in ipadapter_images]
        ipadapter_images = [convert_image_to_base64(ipadapter_image) for ipadapter_image in ipadapter_images]

    request_body = {
        'checkpoint': 'SD15_LEOSAMMoonFilm2.0.safetensors',
        'vae': 'SD15_vae-ft-mse-840000-ema-pruned.safetensors',
        'init_image': image,
        'mask': mask,
        "prompt_positive": prompt,
        "prompt_retouch": prompt_retouch,
        'is_retouch': do_retouch == 'True',
        'steps': 25,
        'cfg': 7,
        'denoise': 1,
        'seed': -1,
        'controlnet_requests': [],
        'lora_requests': [],
    }

    canny_body = {
        'controlnet': 'SD15_Canny_control_v11p_sd15_lineart.pth',
        'type': 'canny',
        'preprocessor_type': 'lineart',
        'strength': 0.95,
        'start_percent': 0,
        'end_percent': 1,
        'base_multiplier': 0.9,
        'uncond_multiplier': 1,
        'flip_weights': False,
    }

    inpaint_body = {
        'controlnet': 'SD15_Inpaint_control_v11p_sd15_inpaint.pth',
        'type': 'inpaint',
        'preprocessor_type': 'canny',
        'strength': 0.9,
        'start_percent': 0,
        'end_percent': 1,
        'base_multiplier': 0.87,
        'uncond_multiplier': 1,
        'flip_weights': True,
    }

    depth_body = {
        'controlnet': 'SD15_Depth_control_sd15_depth.pth',
        'type': 'depth',
        'preprocessor_type': 'depth_zoe',
        'strength': 0.85,
        'start_percent': 0,
        'end_percent': 1,
        'base_multiplier': 0.9,
        'uncond_multiplier': 1,
        'flip_weights': False,
    }

    scribble_body = {
        'controlnet': 'SD15_Scribble_control_v11p_sd15_scribble.pth',
        'type': 'scribble',
        'preprocessor_type': 'scribble',
        'strength': 0.5,
        'start_percent': 0,
        'end_percent': 0.85,
        'base_multiplier': 0.9,
        'uncond_multiplier': 1,
        'flip_weights': False,
    }

    ipadapter_body = {
        'ipadapter': ipadapter_model_name,
        'clip_vision': 'CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors',
        'images': ipadapter_images,
        'weight': ipadapter_weight,
        'start_at': ipadapter_start,
        'end_at': ipadapter_end,
    }

    lora_requests_sorted = sorted([['SD15_Background_Detail_v3.safetensors', 0.3, 1],
                                   ['SD15_more_details.safetensors', 0.2, 1]])
    lora_body_list = []

    for lora_request_sorted in lora_requests_sorted:
        if lora_request_sorted[0] == 'None': continue
        lora_body = {'lora': lora_request_sorted[0],
                     'strength_model': lora_request_sorted[1],
                     'strength_clip': lora_request_sorted[2], }
        lora_body_list.append(lora_body)

    request_body['controlnet_requests'].append(canny_body)
    request_body['controlnet_requests'].append(inpaint_body)
    request_body['controlnet_requests'].append(depth_body)
    # TODO: 그냥 extend 한줄로 해도 될듯. 위에서 None 걸러내서.
    for lora_body in lora_body_list:
        if lora_body['lora'] != 'None' :
            request_body['lora_requests'].append(lora_body)
    if ipadapter_enable :
        request_body['ipadapter_request'] = ipadapter_body
    if do_retouch == 'True'  :
        request_body['controlnet_requests'].append(scribble_body)

    url = f"http://{ip_addr}/sd15/bg_change"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image_blend_base64 = data['image_blend_base64']
    image = convert_base64_to_image_array(image_base64)
    image_blend = convert_base64_to_image_array(image_blend_base64)
    return [image, image_blend]
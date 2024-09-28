import torch
from cgen_utils.loader import load_face_detailer, load_sam, load_detect_provider
import requests
from cgen_utils.handler import handle_response
from cgen_utils.image_process import (convert_base64_to_image_array,
                                      convert_image_array_to_base64,
                                      convert_image_tensor_to_base64,
                                      resize_image_for_sd,
                                      convert_image_to_base64,
                                      convert_base64_to_image_tensor,
                                      controlnet_image_preprocess
                                      )
from cgen_utils.comfyui import (apply_controlnet,
                                apply_ipadapter,
                                make_image_batch)
from cgen_utils.text_process import gemini_with_prompt_and_image
from cgen_utils.loader import load_clip_vision
from types import NoneType
from PIL import Image
import numpy as np


query = """give a detailed description of a given image.
description will be used as a text-to-image generation prompt.
So it's really important for detailed and quality prompt from the image.
Include human race information (black, white, Asian, Korean, etc.)
Your description should be less than 40 words.
Do not leave any items in the image or details behind.
Do not just list the items in the image.
Describe them like a prompt for image generation.
Your description will be used as image generation prompt.
So give me prompt that would work great when generating image."""
@torch.inference_mode()
def model_patch(unet) :
    from ComfyUI.comfy_extras.nodes_freelunch import FreeU_V2
    model_patcher = FreeU_V2()
    unet = model_patcher.patch(unet, b1=1.9, b2=1.4, s1=0.9, s2=0.2)[0]
    return unet


@torch.inference_mode()
def construct_controlnet_condition(
        cached_model_dict,
        positive,
        negative,
        controlnet_requests,
):

    for controlnet_request in controlnet_requests:
        if controlnet_request.type == 'inpaint':
            control_image = convert_base64_to_image_tensor(controlnet_request.image) / 255
            control_image, control_mask = control_image[:, :, :, :3], control_image[:, :, :, 3]
            control_image = torch.where(control_mask[:, :, :, None] > 0.5, 1, control_image)
        else :
            control_image = convert_base64_to_image_tensor(controlnet_request.image) / 255
            control_image = controlnet_image_preprocess(control_image, controlnet_request.preprocessor_type, 'sd15')
        controlnet = cached_model_dict['controlnet']['sd15'][controlnet_request.type][1]
        positive, negative = apply_controlnet(positive,
                                              negative,
                                              controlnet,
                                              control_image,
                                              controlnet_request.strength,
                                              controlnet_request.start_percent,
                                              controlnet_request.end_percent, )


    return positive, negative


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


@torch.inference_mode()
def face_detailer(image, unet, clip, vae, positive_cond, negative_cond, seed) :
    try:
        face_detail_module = load_face_detailer()
    except:
        face_detail_module = load_face_detailer()
    sam_model_opt = load_sam()
    bbox_detector = load_detect_provider()

    image_face_detailed, _, _, _, _ = face_detail_module.enhance_face(
        image,
        unet,
        clip,
        vae,
        bbox_detector=bbox_detector,
        sam_model_opt = sam_model_opt,
        guide_size = 512,
        guide_size_for_bbox=True,
        max_size=512,
        seed = seed,
        steps = 10,
        cfg = 3,
        sampler_name='euler_ancestral',
        scheduler='simple',
        positive=positive_cond,
        negative=negative_cond,
        denoise=0.3,
        feather=10,
        noise_mask=True,
        force_inpaint=True,
        bbox_threshold=0.8,
        bbox_dilation=8,
        bbox_crop_factor=1.5,
        sam_detection_hint='center-1',
        sam_dilation=0,
        sam_threshold=0.93,
        sam_bbox_expansion=0,
        sam_mask_hint_threshold=0.7,
        sam_mask_hint_use_negative='False',
        drop_size=10,
        noise_mask_feather=2
    )
    del sam_model_opt, bbox_detector
    return image_face_detailed

def sned_i2c_request_to_api(
        image,
        style_type,
        ip_addr
) :
    if not isinstance(image, NoneType):
        image = resize_image_for_sd(Image.fromarray(image))
        image = convert_image_to_base64(image)

    sampler_name = 'euler_ancestral' if style_type == 'type_1' else 'dpmpp_2m_sde_gpu'
    scheduler = 'simple' if style_type == 'type_1' else 'karras'

    request_body = {
        'checkpoint': 'SD15_disneyPixarCartoon_v10.safetensors',
        'vae': 'SD15_vae-ft-mse-840000-ema-pruned.safetensors',
        'prompt_negative': 'worst quality, low quality, normal quality, bad hands,text,bad anatomy',
        'init_image': image,
        'steps': 30,
        'sampler_name': sampler_name,
        'scheduler': scheduler,
        'gen_type': 'inpaint',
        'controlnet_requests': [],
        'lora_requests': [],
    }

    refiner_body = {
        'refiner': 'SD15_realcartoonPixar_v12.safetensors',
    }

    canny_body = {
        'controlnet': 'SD15_Canny_control_v11p_sd15_lineart.pth',
        'type': 'canny',
        'image': image,
        'preprocessor_type': 'lineart',
        'strength': 0.8 if style_type == 'type_1' else 0.65,
        'start_percent': 0,
        'end_percent': 1,
    }

    depth_body = {
        'controlnet': 'SD15_Depth_control_sd15_depth.pth',
        'type': 'depth',
        'image': image,
        'preprocessor_type': 'depth_zoe',
        'strength': 0.6,
        'start_percent': 0,
        'end_percent': 1,
    }

    pose_body = {
        'controlnet': 'SD15_Pose_control_v11p_sd15_openpose.pth',
        'type': 'pose',
        'image': image,
        'preprocessor_type': 'dwpose',
        'strength': 0.6,
        'start_percent': 0,
        'end_percent': 1,
    }

    ipadapter_body = {
        'ipadapter': 'SD15_ip-adapter-plus_sd15.safetensors',
        'clip_vision': 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors',
        'images': [image],
        'weight': 0.7 if style_type == 'type_1' else 0.6,
        'start_at': 0,
        'end_at': 1,
    }

    request_body.update(refiner_body)
    request_body['controlnet_requests'].append(canny_body)
    request_body['controlnet_requests'].append(depth_body)
    if style_type == 'type_2':
        request_body['controlnet_requests'].append(pose_body)
    request_body['ipadapter_request'] = ipadapter_body

    url = f"http://{ip_addr}/i2c/generate"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image_face_detail_base64 = data['image_face_detail_base64']

    image = convert_base64_to_image_array(image_base64)
    image_face_detail = convert_base64_to_image_array(image_face_detail_base64)

    return [image, image_face_detail]


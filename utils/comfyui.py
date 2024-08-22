import torch
from utils.loader import get_function_from_comfyui
from utils import set_comfyui_packages
# set_comfyui_packages()

@torch.inference_mode()
def sample_image(unet,
                 positive_cond,
                 negative_cond,
                 latent_image,
                 seed,
                 steps,
                 cfg,
                 sampler_name,
                 scheduler,
                 denoise,
                 start_at_step=None,
                 end_at_step=None) :
    from ComfyUI.nodes import KSamplerAdvanced

    if start_at_step == None :
        start_at_step = 0
        end_at_step = steps
    ksampler = KSamplerAdvanced()
    image_latent = ksampler.sample(
        model= unet,
        positive= positive_cond,
        negative= negative_cond,
        latent_image= latent_image,
        noise_seed= seed,
        steps= steps,
        start_at_step=start_at_step,
        end_at_step=end_at_step,
        cfg= cfg,
        sampler_name= sampler_name,
        scheduler= scheduler,
        denoise= denoise,

        add_noise='enable',
        return_with_leftover_noise='disable'
    )[0]

    return image_latent

@torch.inference_mode()
def decode_latent(vae, latent) :
    from ComfyUI.nodes import VAEDecode
    vae_decoder = VAEDecode()
    image = vae_decoder.decode(vae, latent)[0]

    return image

@torch.inference_mode()
def encode_image(vae, image) :
    from ComfyUI.nodes import VAEEncode
    vae_encoder = VAEEncode()
    latent = vae_encoder.encode(vae, image)[0]

    return latent

@torch.inference_mode()
def encode_image_for_inpaint(vae, image, mask, grow_mask_by=6) :
    from ComfyUI.nodes import VAEEncodeForInpaint
    vae_encoder = VAEEncodeForInpaint()
    latent = vae_encoder.encode(vae, image, mask, grow_mask_by)[0]

    return latent

@torch.inference_mode()
def encode_prompt(clip, prompt_positive, prompt_negative) :
    from ComfyUI.nodes import CLIPTextEncode
    text_encoder = CLIPTextEncode()
    positive_cond = text_encoder.encode(clip, prompt_positive)[0]
    negative_cond = text_encoder.encode(clip, prompt_negative)[0]

    return positive_cond, negative_cond

@torch.inference_mode()
def apply_controlnet(positive, negative, controlnet, image, strength, start_percent, end_percent,) :
    from ComfyUI.nodes import ControlNetApplyAdvanced
    controlnet_applier = ControlNetApplyAdvanced()
    positive, negative = controlnet_applier.apply_controlnet(positive, negative, controlnet, image, strength, start_percent, end_percent)
    return positive, negative

@torch.inference_mode()
def make_canny(image, low_threshold=0.4, high_threshold=0.8) :
    from ComfyUI.comfy_extras.nodes_canny import Canny
    canny_detector = Canny()
    canny_image = canny_detector.detect_edge(image, low_threshold, high_threshold)[0]
    return canny_image

@torch.inference_mode()
def apply_ipadapter(model,
                    ipadapter,
                    clip_vision,
                    image,
                    weight,
                    start_at,
                    end_at,
                    image_negative=None,
                    weight_type='linear',
                    combine_embeds='concat',
                    embeds_scaling = 'V only') :
    from ComfyUI.custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterAdvanced
    ipadapter_applier = IPAdapterAdvanced()
    unet = ipadapter_applier.apply_ipadapter(
        model=model,
        ipadapter=ipadapter,
        image=image,
        clip_vision=clip_vision,
        weight=weight,
        start_at=start_at,
        end_at=end_at,
        image_negative=image_negative,
        weight_type=weight_type,
        combine_embeds=combine_embeds,
        embeds_scaling=embeds_scaling
    )[0]
    return unet

def get_init_noise(width, height, batch_size=1) :
    from ComfyUI.nodes import EmptyLatentImage
    latent_sampler = EmptyLatentImage()
    init_noise = latent_sampler.generate(width, height, batch_size)[0]

    return init_noise

def make_image_batch(image_list) :
    piv_image = image_list.pop(0)
    image_batch = [piv_image]
    if piv_image is None: return piv_image

    from ComfyUI.comfy.utils import common_upscale
    for image in image_list:
        image = common_upscale(image.movedim(-1, 1), piv_image.shape[2], piv_image.shape[1], "bilinear", "center").movedim(1, -1)
        image_batch.append(image)
    image_batch = torch.cat(image_batch, dim=0)
    return image_batch

def mask_blur(mask, amount=6) :
    from ComfyUI.custom_nodes.ComfyUI_essentials.mask import MaskBlur
    maskblur = MaskBlur()
    mask_blurred = maskblur.execute(mask, amount, 'auto')[0]
    return mask_blurred
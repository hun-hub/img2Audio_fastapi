import torch
from cgen_utils.loader import get_function_from_comfyui
from cgen_utils import set_comfyui_packages
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
                 start_at_step,
                 end_at_step,
                 add_noise='enable',
                 return_with_leftover_noise='disable'
                 ) :
    from ComfyUI.nodes import KSamplerAdvanced

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

        add_noise=add_noise,
        return_with_leftover_noise=return_with_leftover_noise
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



def get_default_args(input_types) :
    args = input_types['required']
    for key, val in args.items():
        if len(val) != 2 : continue
        input_type, info = val
        if 'default' not in info : continue
        args[key] = info['default']

    return args

@torch.inference_mode()
def apply_ipadapter(
        unet,
        ipadapter,
        **kwargs
) :
    from ComfyUI.custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterAdvanced
    ipadapter_applier = IPAdapterAdvanced()

    unet, _ = ipadapter_applier.apply_ipadapter(
        model=unet,
        ipadapter=ipadapter,
        **kwargs
    )
    return unet

@torch.inference_mode()
def apply_lora_to_unet(unet, clip, cached_model_dict, lora_request) :
    from ComfyUI.comfy.sd import load_lora_for_models
    lora_type = lora_request.lora.split('_')[0].lower()
    lora = None
    for i in range(3) :
        lora_tuple = cached_model_dict['lora'][lora_type][f'module_{i+1}']
        lora_name, lora_module = lora_tuple
        if lora_name == lora_request.lora :
            lora = lora_module
            break
    if lora is None :
        raise ValueError(f"Lora {lora_request.lora} not found in the cached_model_dict")
    unet, clip = load_lora_for_models(unet, clip, lora, lora_request.strength_model, strength_clip=lora_request.strength_clip)
    return unet, clip
@torch.inference_mode()
def get_init_noise(width, height, batch_size=1) :
    from ComfyUI.nodes import EmptyLatentImage
    latent_sampler = EmptyLatentImage()
    init_noise = latent_sampler.generate(width, height, batch_size)[0]

    return init_noise
@torch.inference_mode()
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
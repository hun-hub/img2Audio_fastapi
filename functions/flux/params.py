from params import RequestData, ControlNet_RequestData, LoRA_RequestData
from typing import Optional, Literal, List,  Union, Tuple


class FLUX_RequestData(RequestData):
    unet: str = 'FLUX_flux1-dev.safetensors'
    vae: str = 'FLUX_VAE.safetensors'
    clip: Union[str, Tuple[str, str]] = ('t5xxl_fp16.safetensors', 'clip_l.safetensors')
    steps: int = 20
    cfg: float = 3.5
    sampler_name: str = 'euler'
    scheduler: str = 'ddim_uniform'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    lora_requests: Optional[List[LoRA_RequestData]] = []
    gen_type: Literal['t2i', 'i2i', 'inpaint'] = 't2i'

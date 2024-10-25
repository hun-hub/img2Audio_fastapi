from params import RequestData, ControlNet_RequestData, IPAdapter_RequestData, LoRA_RequestData
from typing import Optional, Literal, List

prompt_negative = ""

class SD15_RequestData(RequestData):
    checkpoint: str = 'SD15_realisticVisionV51_v51VAE.safetensors'
    steps: int = 20
    cfg: float = 7
    prompt_negative: str = prompt_negative
    sampler_name: str = 'dpmpp_sde'
    scheduler: str = 'karras'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    ipadapter_request: Optional[IPAdapter_RequestData] = None
    lora_requests: Optional[List[LoRA_RequestData]] = []
    gen_type: Literal['t2i', 'i2i', 'inpaint'] = 't2i'
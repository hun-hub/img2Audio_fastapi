from params import RequestData, ControlNet_RequestData, IPAdapter_RequestData, LoRA_RequestData
from typing import Optional, Literal, List

prompt_negative = "EasyNegative,bad_prompt_version2-neg,NSFW,(worst quality:2),(low quality:2),(normal quality:2),normal quality,((grayscale)), (duplicate:1.331),(mutilated:1.21),blurry,(disfigured:1.331),lowers,extra digit"
class ControlNet_ControlNet_RequestData(ControlNet_RequestData):
    image: str = None
    base_multiplier: float = 1
    uncond_multiplier:float = 1
    flip_weights: bool

class BGChange_RequestData(RequestData):
    checkpoint: str = 'SD15_LEOSAMMoonFilm2.0.safetensors'
    steps: int = 20
    cfg: float = 7
    prompt_retouch: str = ''
    prompt_negative: str = prompt_negative
    sampler_name: str = 'dpmpp_2m'
    scheduler: str = 'karras'
    init_image: str
    mask: str
    is_retouch: bool
    controlnet_requests: Optional[List[ControlNet_ControlNet_RequestData]] = []
    ipadapter_request: Optional[IPAdapter_RequestData] = None
    lora_requests: Optional[List[LoRA_RequestData]] = []

from params import RequestData, ControlNet_RequestData
from typing import Optional, Literal, List

class SD3_RequestData(RequestData):
    checkpoint: str = 'SD3_sd3_medium_incl_clips_t5xxlfp16.safetensors'
    steps: int = 28
    cfg: float = 4.5
    sampler_name: str = 'dpmpp_2m'
    scheduler: str = 'sgm_uniform'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    gen_type: Literal['t2i', 'i2i', 'inpaint'] = 't2i'

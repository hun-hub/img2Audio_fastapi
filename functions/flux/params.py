from params import RequestData, ControlNet_RequestData, LoRA_RequestData
from typing import Optional, Literal, List


class FLUX_RequestData(RequestData):
    steps: int = 20
    cfg: float = 3.5
    # sampler_name: str = 'dpmpp_2m'
    # scheduler: str = 'sgm_uniform'
    sampler_name: str = 'euler'
    scheduler: str = 'simple'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    lora_requests: Optional[List[LoRA_RequestData]] = []
    gen_type: Literal['t2i', 'i2i', 'inpaint'] = 't2i'

from params import RequestData, ControlNet_RequestData, IPAdapter_RequestData, LoRA_RequestData
from typing import Optional, Literal, List


class Half_Inpainting_RequestData(RequestData):
    checkpoint: str = 'SDXL_RealVisXL_V40.safetensors'
    steps: int = 20
    cfg: float = 7
    sampler_name: str = 'dpmpp_2m_sde'
    scheduler: str = 'karras'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    ipadapter_request: Optional[IPAdapter_RequestData] = None


from pydantic import BaseModel
from typing import Optional, List, Literal

class IPAdapter_RequestData(BaseModel):
    ipadapter: str
    clip_vision: str
    images: List[str]
    image_negative: Optional[List[str]] = []
    # Params
    weight: float
    start_at: float = 0
    end_at: float
    weight_type: str = 'linear'
    combine_embeds: Literal['concat', 'add', 'substract', 'average', 'norm average'] = 'concat'
    embeds_scaling: Literal['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'] = 'V only'

class ControlNet_RequestData(BaseModel):
    controlnet: str
    type: Literal['canny', 'inpaint']
    image: Optional[str]
    # Params
    strength: float
    start_percent: float = 0
    end_percent: float

class RequestData(BaseModel):
    basemodel: str = 'SDXL_copaxTimelessxlSDXL1_v12.safetensors'
    init_image: Optional[str] = None
    mask: Optional[str] = None
    prompt_positive: str = 'high quality, 4K, expert.'
    prompt_negative: str = 'Low quality, Blur, artifact'
    seed: int = -1
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    steps: int = 20
    cfg: float = 7
    sampler_name: str = 'dpmpp_2m'
    scheduler: str = 'karras'
    denoise: float= 1.0
    gen_type: Literal['t2i', 'i2i', 'inpaint', 'iclight'] = 't2i'

    # controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    # ipadapter_request: Optional[IPAdapter_RequestData] = None
    # refiner: Optional[str] = None
    # refine_switch: float= 0.4


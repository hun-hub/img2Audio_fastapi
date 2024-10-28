from pydantic import BaseModel
from typing import Optional, List, Literal, Union, Tuple

class IPAdapter_RequestData(BaseModel):
    ipadapter: str
    clip_vision: str
    images: List[str]
    image_negative: Optional[List[str]] = []
    # Params
    weight: float = 0.7
    start_at: float = 0
    end_at: float = 0.4
    weight_type: str = 'linear'
    combine_embeds: Literal['concat', 'add', 'substract', 'average', 'norm average'] = 'concat'
    embeds_scaling: Literal['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'] = 'V only'

class ControlNet_RequestData(BaseModel):
    controlnet: str
    type: Literal['canny', 'inpaint', 'depth', 'normal', 'pose', 'scribble']
    image: Optional[str]
    preprocessor_type: Literal['canny', 'lineart', 'dwpose', 'normalmap_bae', 'normalmap_midas', 'depth_midas', 'depth', 'depth_zoe', 'scribble']
    # Params
    strength: float = 0.7
    start_percent: float = 0
    end_percent: float = 0.4

class LoRA_RequestData(BaseModel):
    lora: str
    strength_model: float
    strength_clip: float

class RequestData(BaseModel):
    checkpoint: str = None
    unet: str = None
    vae: str = None
    clip: Union[str, Tuple[str, str]] = None
    clip_vision: str = None

    refiner: Optional[str] = None
    refine_switch: float= 0.4

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

class Resize_RequestData(BaseModel):
    image: str
    resize_type: Literal['sd15', 'sdxl'] = 'sdxl'
    is_mask: bool = False

class MaskEdit_RequestData(BaseModel) :
    image:str
    mask: str
    edit_mode: Literal['add_white', 'add_black'] = 'add_white'
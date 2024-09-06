from params import RequestData, ControlNet_RequestData, IPAdapter_RequestData
from typing import Optional, Literal, List


class ICLight_RequestData(RequestData):
    checkpoint: str = 'SD15_epicrealism_naturalSinRC1VAE.safetensors'
    steps: int = 30
    cfg: float = 1.5
    sampler_name: str = 'dpmpp_2m_sde'
    scheduler: str = 'karras'

    iclight_model: str
    light_condition: str
    light_strength: float = 0.5
    keep_background: bool = True
    blending_mode_1: str = 'color'
    blending_percentage_1: float = 0.1
    blending_mode_2: str = 'hue'
    blending_percentage_2: float = 0.2
    remap_min_value: float = -0.15
    remap_max_value: float = 1.14

    controlnet_requests: list = [] # 필요없는데 model cache update 때문에 넣음.
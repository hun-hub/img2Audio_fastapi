from params import RequestData, ControlNet_RequestData, IPAdapter_RequestData
from typing import Optional, Literal, List


class Object_Remove_RequestData(RequestData):
    basemodel: str
    inpaint_model_name: str
    init_image: str
    mask: str
    steps: int = 10
    cfg: float = 4
    sampler_name: str = 'dpmpp_2m_sde'
    scheduler: str = 'karras'
    controlnet_requests: list = [] # 필요없는데 model cache update 때문에 넣음.


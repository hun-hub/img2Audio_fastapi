from typing import Optional, Literal, List
from params import RequestData


class Upscale_RequestData(RequestData):
    upscale_model: str = '4x-UltraSharp.pth'
    init_image: str
    method: Literal['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos'] = 'lanczos'
    scale: float = 2
    controlnet_requests: list = [] # 필요없는데 model cache update 때문에 넣음.


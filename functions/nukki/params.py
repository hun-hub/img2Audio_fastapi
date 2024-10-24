from typing import Optional, Literal, List
from params import RequestData

class Nukki_RequestData(RequestData):
    nukki_model: str = 'DIS-TR_TEs.safetensors'
    init_image: str

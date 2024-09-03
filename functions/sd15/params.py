from params import RequestData, ControlNet_RequestData, IPAdapter_RequestData, LoRA_RequestData
from typing import Optional, Literal, List

prompt_negative = """boring_e621_v4, (((text, watermark, logo, phrases, person, face))), 
(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), cropped, out of frame, 
worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, 
poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, 
gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream"""

class SD15_RequestData(RequestData):
    steps: int = 20
    cfg: float = 7
    prompt_negative: str = prompt_negative
    sampler_name: str = 'dpmpp_sde'
    scheduler: str = 'karras'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    ipadapter_request: Optional[IPAdapter_RequestData] = None
    lora_requests: Optional[List[LoRA_RequestData]] = []
    gen_type: Literal['t2i', 'i2i', 'inpaint'] = 't2i'
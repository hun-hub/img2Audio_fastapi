# Connectbrick 자체 API 및 Demo

## 0. Install
[Setting manual](https://www.notion.so/connectbrick/GCP-L4-Server-setting-21b19b610d044d018fcfae14d55f8fde)
* Anaconda
* g-cloud
* huggingface 


## 1. 실행
`.env` 파일 수정
```Bash
COMFYUI_PATH =/home/gkalstn000/cnt_api/ComfyUI
CHECKPOINT_ROOT = /home/gkalstn000/cnt_api/ComfyUI/models
```

```Bash
# API 실행
python main_api.py
# Demo 실행
python main_demo.py
```

| 서버  | IP addr | API Port | Demo Port |
|-----|---------|----------|-----------|
| 개발  | 34.69.14.188     | 7861     | 7860      |
| 테스트 | 130.211.239.93     | 7861     | 7860      |



## 2. SD type 별 기능 요약
| SD type | T2I | I2I | Inpaint  | Controlnet[Canny] | Controlnet[Inpaint] | Controlnet[Depth] | Controlnet[Normal] | Controlnet[Pose] | IP-Adapter | LoRA |
|---------|-----|-----|----------|-------------------|---------------------|-------------------|--------------------|------------------|------------|------|
| FLUX    | O   | O   | O        | O                 | X                   | O                 | X                  | X                | X          | O    |
| SD3     | O   | O   | O        | O                 | X                   | X                 | X                  | X                | X          | X    |
| SDXL    | O   | O   | O        | O                 | O                   | O                 | O                  | O                | O          | O    |
| SD15    | O   | O   | O        | O                 | O                   | O                 | O                  | O                | O          | O    |

* SDXL, SD15 Controlnet Depth는 필요시 작업 수행.

### Patch Node
* (240925) Image to Pixar 추가
* (240924) Controlnet Depth, Normal, Pose Update

### 2.1 Basic Request Format

<details>
<summary>RequestData Class Details</summary>

```python
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
    type: Literal['canny', 'inpaint', 'depth']
    image: Optional[str]
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
```

</details>

Request Overview
```python
request_body = {
    'checkpoint': CHECKPOINT_NAME,
    'init_image': IMAGE_BASE64,
    'mask': MASK_BASE64,
    "prompt_positive": POSITIVE_PROMPT,
    "prompt_negative": NEGATIVE_PROMPT,
    'width': WIDTH,
    'height': HEIGHT,
    'steps': 20,
    'cfg': 7,
    'denoise': 1,
    'gen_type': 't2i', # ['t2i', 'i2i', 'inpain'] 
    'controlnet_requests': [],
    'lora_requests': [],
}

canny_body = {
    'controlnet': CANNY_MODEL_NAME,
    'type': 'canny',
    'image': IMAGE_BASE64,
    'preprocessor_type': Literal['canny', 'lineart', 'dwpose', 'normalmap_bae', 'normalmap_midas', 'depth_midas', 'depth', 'depth_zoe'],
    'strength': 1,
    'start_percent': 0,
    'end_percent': 1,
}

inpaint_body = {
    'controlnet': INPAINT_MODEL_NAME,
    'type': 'inpaint',
    'image': IMAGE_BASE64,
    'strength': 1,
    'start_percent': 0,
    'end_percent': 1,
}

ipadapter_body = {
    'ipadapter': IPADAPTER_MODEL_NAME,
    'clip_vision': 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors', # SD15: 'CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors'
    'images': [IMAGE_BASE64, IMAGE_BASE64],
    'weight': 1,
    'start_at': 0,
    'end_at': 1,
}

lora_body = {'lora': LoRA_MODEL_NAME,
             'strength_model': 1,
             'strength_clip': 1}

if canny_enable :
    request_body['controlnet_requests'].append(canny_body)
if inpaint_enable :
    request_body['controlnet_requests'].append(inpaint_body)
if ipadapter_enable :
    request_body['ipadapter_request'] = ipadapter_body
if lora_enable : # 최대 3개
    request_body['lora_requests'].append(lora_body)

flux_url = f"http://{IP_Addr}:{Port}/flux/generate"
sd3_url = f"http://{IP_Addr}:{Port}/sd3/generate"
sdxl_url = f"http://{IP_Addr}:{Port}/sdxl/generate"
sd15_url = f"http://{IP_Addr}:{Port}/sd15/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64']
image = convert_base64_to_image_array(image_base64)
```
Model 종류는 Demo에서 확인 가능.  



### 2.2 Service Request Examples
* [Jector](docs/jector.md)
* [Proposal](docs/proposal.md)
* [IC-Light](docs/iclight.md)

### 2.3 Base function Examples
* [Flux](#flux)
* [SD3](#sd3)
* [SDXL](#sdxl)
* [SD15](#sd15)
* [Object Removal](#object-removal)
* [Up-scale](#up-scale)
* [IC-Light](#ic-light)
* [Segment Anything (coming soon)](#segment-anything)
* [Gemini](#gemini)
* [Image-to-Cartoon](#image-to-cartoon-)
---
#### FLUX
Parameter format
```python
class FLUX_RequestData(RequestData):
    unet: str = 'FLUX_flux1-dev.safetensors'
    vae: str = 'FLUX_VAE.safetensors'
    clip: Union[str, Tuple[str, str]] = ('t5xxl_fp16.safetensors', 'clip_l.safetensors')
    steps: int = 20
    cfg: float = 3.5
    sampler_name: str = 'euler'
    scheduler: str = 'simple'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    lora_requests: Optional[List[LoRA_RequestData]] = []
    gen_type: Literal['t2i', 'i2i', 'inpaint'] = 't2i'
```
FLUX Image Generation Example

```python
request_body = {
    'unet': 'FLUX_flux1-dev.safetensors',
    'vae': 'FLUX_VAE.safetensors',
    'clip': ('t5xxl_fp16.safetensors', 'clip_l.safetensors'),    
    
    'init_image': IMAGE_BASE64,
    'mask': MASK_BASE64,
    "prompt_positive": POSITIVE_PROMPT,
    "prompt_negative": NEGATIVE_PROMPT,
    'width': WIDTH,
    'height': HEIGHT,
    'steps': 20,
    'cfg': 3.5,
    'denoise': 1,
    'gen_type': 't2i', # ['t2i', 'i2i', 'inpain'] 
    'controlnet_requests': [],
    'lora_requests': [],
}

canny_body = {
    'controlnet': CANNY_MODEL_NAME,
    'type': 'canny',
    'image': IMAGE_BASE64,
    'strength': 1,
    'start_percent': 0,
    'end_percent': 1,
}

lora_body = {'lora': LoRA_MODEL_NAME,
             'strength_model': 1,
             'strength_clip': 1}

if canny_enable :
    request_body['controlnet_requests'].append(canny_body)
if lora_enable : # 최대 3개
    request_body['lora_requests'].append(lora_body)

url = f"http://{IP_Addr}:{Port}/flux/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64']
image = convert_base64_to_image_array(image_base64)
```
---
#### SD3
Parameter format
```python
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
```

SD3 Image Generation Example
```python
request_body = {
    'checkpoint': 'SD3_sd3_medium_incl_clips_t5xxlfp16.safetensors',
    'init_image': IMAGE_BASE64,
    'mask': MASK_BASE64,
    "prompt_positive": POSITIVE_PROMPT,
    "prompt_negative": NEGATIVE_PROMPT,
    'width': WIDTH,
    'height': HEIGHT,
    'steps': 20,
    'cfg': 4,
    'denoise': 1,
    'gen_type': 't2i', # ['t2i', 'i2i', 'inpain'] 
    'controlnet_requests': [],
    'lora_requests': [],
}

canny_body = {
    'controlnet': CANNY_MODEL_NAME,
    'type': 'canny',
    'image': IMAGE_BASE64,
    'strength': 1,
    'start_percent': 0,
    'end_percent': 1,
}

if canny_enable :
    request_body['controlnet_requests'].append(canny_body)

url = f"http://{IP_Addr}:{Port}/sd3/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64']
image = convert_base64_to_image_array(image_base64)
```
---
#### SDXL
Parameter format
```python
class SDXL_RequestData(RequestData):
    checkpoint: str = 'SDXL_RealVisXL_V40.safetensors'
    steps: int = 20
    cfg: float = 7
    sampler_name: str = 'dpmpp_2m_sde'
    scheduler: str = 'karras'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    ipadapter_request: Optional[IPAdapter_RequestData] = None
    lora_requests: Optional[List[LoRA_RequestData]] = []
    gen_type: Literal['t2i', 'i2i', 'inpaint'] = 't2i'
    refiner: Optional[str] = None
    refine_switch: float= 0.4
```

SDXL Image Generation Example
```python
request_body = {
    'checkpoint': 'SDXL_RealVisXL_V40.safetensors',
    'init_image': IMAGE_BASE64,
    'mask': MASK_BASE64,
    "prompt_positive": POSITIVE_PROMPT,
    "prompt_negative": NEGATIVE_PROMPT,
    'width': WIDTH,
    'height': HEIGHT,
    'steps': 20,
    'cfg': 7,
    'denoise': 1,
    'gen_type': 't2i', # ['t2i', 'i2i', 'inpain'] 
    'controlnet_requests': [],
    'lora_requests': [],
}

refiner_body = {
    'refiner': 'SDXL_copaxPhotoxl_v2.safetensors',
    'refine_switch': 0.45,
}

canny_body = {
    'controlnet': CANNY_MODEL_NAME,
    'type': 'canny',
    'image': IMAGE_BASE64,
    'strength': 1,
    'start_percent': 0,
    'end_percent': 1,
}

inpaint_body = {
    'controlnet': INPAINT_MODEL_NAME,
    'type': 'inpaint',
    'image': IMAGE_BASE64,
    'strength': 1,
    'start_percent': 0,
    'end_percent': 1,
}

ipadapter_body = {
    'ipadapter': IPADAPTER_MODEL_NAME,
    'clip_vision': 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors', # SD15: 'CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors'
    'images': [IMAGE_BASE64, IMAGE_BASE64],
    'weight': 1,
    'start_at': 0,
    'end_at': 1,
}

lora_body = {'lora': LoRA_MODEL_NAME,
             'strength_model': 1,
             'strength_clip': 1}

if refiner_enable:
    request_body.update(refiner_body)
if canny_enable :
    request_body['controlnet_requests'].append(canny_body)
if inpaint_enable :
    request_body['controlnet_requests'].append(inpaint_body)
if ipadapter_enable :
    request_body['ipadapter_request'] = ipadapter_body
if lora_enable : # 최대 3개
    request_body['lora_requests'].append(lora_body)

url = f"http://{IP_Addr}:{Port}/sdxl/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64']
image = convert_base64_to_image_array(image_base64)
```
---
#### SD15
Parameter format
```python
class SD15_RequestData(RequestData):
    checkpoint: str = 'SD15_realisticVisionV51_v51VAE.safetensors'
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
```

SD15 Image Generation Example
```python
request_body = {
    'checkpoint': 'SD15_epicrealism_naturalSinRC1VAE.safetensors',
    'init_image': IMAGE_BASE64,
    'mask': MASK_BASE64,
    "prompt_positive": POSITIVE_PROMPT,
    "prompt_negative": NEGATIVE_PROMPT,
    'width': WIDTH,
    'height': HEIGHT,
    'steps': 20,
    'cfg': 7,
    'denoise': 1,
    'gen_type': 't2i', # ['t2i', 'i2i', 'inpain'] 
    'controlnet_requests': [],
    'lora_requests': [],
}

canny_body = {
    'controlnet': CANNY_MODEL_NAME,
    'type': 'canny',
    'image': IMAGE_BASE64,
    'strength': 1,
    'start_percent': 0,
    'end_percent': 1,
}

inpaint_body = {
    'controlnet': INPAINT_MODEL_NAME,
    'type': 'inpaint',
    'image': IMAGE_BASE64,
    'strength': 1,
    'start_percent': 0,
    'end_percent': 1,
}

ipadapter_body = {
    'ipadapter': IPADAPTER_MODEL_NAME,
    'clip_vision': 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors', # SD15: 'CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors'
    'images': [IMAGE_BASE64, IMAGE_BASE64],
    'weight': 1,
    'start_at': 0,
    'end_at': 1,
}

lora_body = {'lora': LoRA_MODEL_NAME,
             'strength_model': 1,
             'strength_clip': 1}

if canny_enable :
    request_body['controlnet_requests'].append(canny_body)
if inpaint_enable :
    request_body['controlnet_requests'].append(inpaint_body)
if ipadapter_enable :
    request_body['ipadapter_request'] = ipadapter_body
if lora_enable : # 최대 3개
    request_body['lora_requests'].append(lora_body)

url = f"http://{IP_Addr}:{Port}/sd15/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64']
image = convert_base64_to_image_array(image_base64)
```
---
#### Object Removal
Parameter format
```python
class Object_Remove_RequestData(RequestData):
    checkpoint: str = 'SDXL_RealVisXL_V40.safetensors'
    inpaint_model_name: str = 'SDXL_inpaint_v26.fooocus.patch'
    init_image: str
    mask: str
    steps: int = 10
    cfg: float = 4
    sampler_name: str = 'dpmpp_2m_sde'
    scheduler: str = 'karras'
```
Object Removal Example
```python
request_body = {
    'checkpoint': 'SDXL_copaxTimelessxlSDXL1_v12.safetensors', # 입력 고정
    'inpaint_model_name': 'SDXL_inpaint_v25.fooocus.patch', # 입력 고정
    'init_image': IMAGE_BASE64, # RGB Image 
    'mask': MASK_BASE64, # RGB Image 
    "prompt_positive": "",
    'steps': 10,
    'cfg': 3,
    'denoise': 1,
}
        
url = f"http://{IP_Addr}:{Port}/object_remove"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
---
#### Up-scale
Parameter format
```python
class Upscale_RequestData(RequestData):
    upscale_model: str = '4x-UltraSharp.pth'
    init_image: str
    method: Literal['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos'] = 'lanczos'
    scale: float = 2 # 최대 4배
```
Up-scale Example
```python
request_body = {
    'upscale_model': '4x-UltraSharp.pth',
    'init_image': IMAGE_BASE64, # RGB Image 
    'method': 'lanczos',
    "scale": 2, # 최대 4배 
}

url = f"http://{IP_Addr}:{Port}/upscale"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
---
#### IC-Light
Parameter format
```python
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
```

SD15 Image Generation Example
```python
request_body = {
    'checkpoint': 'SD15_epicrealism_naturalSinRC1VAE.safetensors',
    'init_image': IMAGE_BASE64, # RGB Image 
    'mask': MASK_BASE64, # RGB Image
    "prompt_positive": POSITIVE_PROMPT, # 입력 필요
    "prompt_negative": NEGATIVE_PROMPT, # 입력 선택
    'steps': 30,
    'cfg': 4,
    'denoise': 0.7,

    'iclight_model': 'SD15_iclight_sd15_fc.safetensors', # 입력 고정
    'light_condition': IMAGE_BASE64, # RGB Image 
    'light_strength': 0.5, # 0 ~ 1 사이 float
    'keep_background': True if keep_background == 'True' else False,
    'blending_mode_1': blending_mode_1,
    'blending_percentage_1': blending_percentage_1,
    'blending_mode_2': blending_mode_2,
    'blending_percentage_2': blending_percentage_2,
    'remap_min_value': remap_min_value,
    'remap_max_value': remap_max_value,
}

url = f"http://{IP_Addr}:{Port}/iclight/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
---
#### Segment Anything
작업중...

---
#### Gemini
Parameter format
```python
class Gemini_RequestData(BaseModel) :
    user_prompt: Optional[str] = ''
    object_description: Optional[str] = ''
    background_description: Optional[str] = ''
    query_type: str
    image: Optional[str] = None
```

Gemini Example

[Gemini Query Document](https://www.notion.so/connectbrick/240830-Gemini-Query-e3b55161ef934083a6a2a9909a0dfc63)

```python
'''
Input
image: str or None
user_prompt: str
object_description: str
backgroun_description: str

query_type: 
    'product_description' : BLIP 대체 (object category 추출)
    'image_description'
    'prompt_refine': Keyword 기반 유저 입력 refine
    'prompt_refine_with_image': referenct 이미지 + 키워드 유저 입력 prompt refine
    'synthesized_image_description': 합성(concat) 이미지 + 유저 입력 prompt refine 추출
    'decompose_background_and_product': backgorund와 object category 각각 추출 (iclight용)
    'iclight_keep_background'
    'iclight_gen_background'
'''
request_body = {
    'user_prompt': user_prompt,
    'object_description': object_description,
    'background_description': background_description,
    'query_type': query_type,
    'image': image # base64
}

url = f"http://{IP_Addr}:{Port}/gemini"

response = requests.post(url, json=request_body)
data = response.json()
prompt = data['prompt']
return prompt
```
--- 
#### Image-to-Cartoon 
Parameter format (SD15와 동일)
```python
class SD15_RequestData(RequestData):
    checkpoint: str = 'SD15_realisticVisionV51_v51VAE.safetensors'
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
```

Image-to-Cartoon Generation Example
```python
'''
Input 인자
image: str, base64 형식 이미지
style_type: str, ['type_1', 'type_2'] 둘 중 하나
'''
import io, base64, os
from PIL import Image
import requests

def _crop_image(image: Image):
    w, h = image.size

    h_ = h - h % 64
    w_ = w - w % 64

    # 중앙을 기준으로 크롭할 영역 계산
    left = (w - w_) // 2
    top = (h - h_) // 2
    right = left + w_
    bottom = top + h_
    image = image.crop((left, top, right, bottom))
    return image

def resize_image_for_sd(image: Image, is_mask=False, resolution = 1024) :
    w, h = image.size
    scale = (resolution ** 2 / (w * h)) ** 0.5

    scale = 1 if scale > 1 else scale
    w_scaled = w * scale
    h_scaled = h * scale
    interpolation = Image.NEAREST if is_mask else Image.BICUBIC
    image_resized = image.resize((int(w_scaled), int(h_scaled)), interpolation)
    image_resized_cropped = _crop_image(image_resized)
    return image_resized_cropped

def convert_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return image_base64

def convert_base64_to_image(image_base64):
    image_data = base64.b64decode(image_base64)
    image_rgb = Image.open(io.BytesIO(image_data))
    return image_rgb

image = Image.open(IMAGE_PATH)
image = resize_image_for_sd(image)
image = convert_image_to_base64(image)

style_type = 'type_X'

sampler_name = 'euler_ancestral' if style_type == 'type_1' else 'dpmpp_2m_sde_gpu'
scheduler = 'simple' if style_type == 'type_1' else 'karras'

request_body = {
    'checkpoint': 'SD15_disneyPixarCartoon_v10.safetensors',
    'vae': 'SD15_vae-ft-mse-840000-ema-pruned.safetensors',
    'prompt_negative': 'worst quality, low quality, normal quality, bad hands,text,bad anatomy',
    'init_image': image,
    'steps': 30,
    'sampler_name': sampler_name,
    'scheduler': scheduler,
    'gen_type': 'inpaint',
    'controlnet_requests': [],
    'lora_requests': [],
}

refiner_body = {
    'refiner': 'SD15_realcartoonPixar_v12.safetensors',
}

canny_body = {
    'controlnet': 'SD15_Canny_control_v11p_sd15_lineart.pth',
    'type': 'canny',
    'image': image,
    'preprocessor_type': 'lineart',
    'strength': 0.8 if style_type == 'type_1' else 0.65,
    'start_percent': 0,
    'end_percent': 1,
}

depth_body = {
    'controlnet': 'SD15_Depth_control_sd15_depth.pth',
    'type': 'depth',
    'image': image,
    'preprocessor_type': 'depth_zoe',
    'strength': 0.6,
    'start_percent': 0,
    'end_percent': 1,
}

pose_body = {
    'controlnet': 'SD15_Pose_control_v11p_sd15_openpose.pth',
    'type': 'pose',
    'image': image,
    'preprocessor_type': 'dwpose',
    'strength': 0.6,
    'start_percent': 0,
    'end_percent': 1,
}

ipadapter_body = {
    'ipadapter': 'SD15_ip-adapter-plus_sd15.safetensors',
    'clip_vision': 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors',
    'images': [image],
    'weight': 0.7 if style_type == 'type_1' else 0.6,
    'start_at': 0,
    'end_at': 1,
}

request_body.update(refiner_body)
request_body['controlnet_requests'].append(canny_body)
request_body['controlnet_requests'].append(depth_body)
if style_type == 'type_2':
    request_body['controlnet_requests'].append(pose_body)
request_body['ipadapter_request'] = ipadapter_body

ip_addr = '130.211.239.93:7861'
url = f"http://{ip_addr}/i2c/generate"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64']
image_face_detail_base64 = data['image_face_detail_base64']
```
---

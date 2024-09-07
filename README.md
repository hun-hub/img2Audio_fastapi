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
| SD type | T2I | I2I | Inpaint | Controlnet[Canny] | Controlnet[Inpaint] | Controlnet[Depth] | IP-Adapter | LoRA |
|---------|-----|-----|--------------------------|-------------------|---------------------|-------------------|------------|------|
| FLUX    | O   | O   | O                        | O                 | X                   | 작업중               | X          | O    |
| SD3     | O   | O   | O                        | O                 | X                   | X                 | X          | X    |
| SDXL    | O   | O   | O                        | O                 | O                   | X        | O          | O    |
| SD15    | O   | O   | O                        | O                 | O                   | X                 | O          | O    |

* SDXL, SD15 Controlnet Depth는 필요시 작업 수행.

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
---
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
        
url = f"http://{ip_addr}:7861/object_remove"
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

url = f"http://{ip_addr}:7861/upscale"

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

url = f"http://{ip_addr}:7861/iclight/generate"

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
    query: str
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

query_type: [
'product_description', 
'image_description',
'prompt_refine',
'prompt_refine_with_image',
'synthesized_image_description',
'decompose_background_and_product',
'iclight_keep_background',
'iclight_gen_background'
]
'''
query = query_dict[query_type]
user_prompt = ', '.join(user_prompt.split(' '))
query = query.format(user_prompt = user_prompt,
                     object_description = object_description,
                     background_description = background_description)

request_body = {
    'query': query,
    'image': image # str or None
}

url = f"http://{ip_addr}:7861/gemini"

response = requests.post(url, json=request_body)
data = response.json()
prompt = data['prompt']
return prompt
```



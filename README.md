# Connectbrick 자체 API 및 Demo

## 0. Install
```Bash
pip install -r requirements.txt
sh setup_comfy.sh

# ComfyUI 한번 실행 했다가 끄기
cd ComfyUI
python main.py --listen
```

## 1. 실행
```Bash
# API 실행
python main_api.py

# Demo 실행
python main_demo.py
```

[Demo 링크](http://34.69.14.188:7860/).  
Develop 서버 IP 주소: `34.69.14.188`
* 상시 켜져 있지는 않음
## 2. API 기능

**Request Data Format**
```python
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

```


### 2.1 Service Request Examples
* [Jector](docs/jector.md)
* [Proposal](docs/proposal.md)
* [IC-Light](docs/iclight.md)

### 2.2 Base function Examples
* [SD3](#221-sd3)
* [SDXL](#222-sdxl)
* [SD15](#223-sd15)
* [Object Removal](#224-object-removal)
* [Up-scale](#225-up-scale)
* [IC-Light](#226-ic-light)
* [Segment Anything (coming soon)](#227-segment-anything)
* [Gemini](#228-gemini)
* [Flux (coming soon)](#229-flux)

#### 2.2.1 SD3

Parameter format
```python
class SD3_RequestData(RequestData):
    basemodel: str
    steps: int = 28
    cfg: float = 4.5
    sampler_name: str = 'dpmpp_2m'
    scheduler: str = 'sgm_uniform'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    gen_type: Literal['t2i', 'i2i', 'inpaint'] = 't2i'
```

**SD3 Image Generation Example**
```python
request_body = {
    'basemodel': 'SD3_sd3_medium_incl_clips_t5xxlfp16.safetensors', # 입력 고정
    'init_image': IMAGE_BASE64, # RGB Image / i2i, inpaint 일 경우 필요
    'mask': MASK_BASE64, # RGB Image / inpaint 일 경우 필요
    "prompt_positive": POSITIVE_PROMPT, # 입력 필요
    "prompt_negative": NEGATIVE_PROMPT, # 입력 선택
    'width': WIDTH, # t2i는 입력 필요
    'height': HEIGHT, # t2i는 입력 필요
    'steps': 28, # 입력 선택
    'cfg': 4, # 입력 선택
    'denoise': 1.0, 
    'gen_type': 't2i', # 't2i' or 'i2i' or 'inpaint'
}

canny_request_body = {
    'controlnet_requests':
        [
            {
                'controlnet': 'SD3_Canny.safetensors', # 입력 고정
                'type': 'canny', # 입력 고정
                'image': IMAGE_BASE64, # RGB Image
                'strength': 0.7, # 입력 선택
                'start_percent': 0, # 입력 선택
                'end_percent': 0.4, # 입력 선택
            }
        ]
}

if canny_enable :
    request_body.update(canny_request_body)


url = f"http://{ip_addr}:7861/sd3/generate"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```

#### 2.2.2 SDXL
Parameter format
```python
class SDXL_RequestData(RequestData):
    basemodel: str
    steps: int = 20
    cfg: float = 7
    sampler_name: str = 'dpmpp_2m_sde'
    scheduler: str = 'karras'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    ipadapter_request: Optional[IPAdapter_RequestData] = None
    gen_type: Literal['t2i', 'i2i', 'inpaint'] = 't2i'
    refiner: Optional[str] = None
    refine_switch: float= 0.4
```

**SDXL Image Generation Example**
```python
request_body = {
    'basemodel': 'SDXL_copaxTimelessxlSDXL1_v12.safetensors', # 입력 고정
    'init_image': IMAGE_BASE64, # RGB Image / i2i, inpaint 일 경우 필요
    'mask': MASK_BASE64, # RGB Image / inpaint 일 경우 필요
    "prompt_positive": POSITIVE_PROMPT, # 입력 필요
    "prompt_negative": NEGATIVE_PROMPT, # 입력 선택
    'width': WIDTH, # t2i는 입력 필요
    'height': HEIGHT, # t2i는 입력 필요
    'steps': 20, # 입력 선택
    'cfg': 7, # 입력 선택
    'denoise': 1.0, 
    'gen_type': 't2i', # 't2i' or 'i2i' or 'inpaint'
    'controlnet_requests': [],
}

refiner_body = {
    'refiner': 'SDXL_RealVisXL_V40.safetensors', # 입력 고정
    'refine_switch': 0.4, # 입력 선택
}

canny_body = {
    'controlnet': 'SDXL_Canny_jector.safetensors',  # 입력 고정
    'type': 'canny',
    'image': IMAGE_BASE64, # RGB Image
    'strength': 0.7, # 입력 선택
    'start_percent': 0, # 입력 선택
    'end_percent': 0.4, # 입력 선택
}

inpaint_body = { # Controlnet Inpaint body
    'controlnet': 'SDXL_Inpaint_dreamerfp16.safetensors',
    'type': 'inpaint',
    'image': IMAGE_BASE64 # RGB Image, 채워질 부분 하얀색(255) 값으로
    'strength': 0.7, # 입력 선택
    'start_percent': 0, # 입력 선택
    'end_percent': 0.4, # 입력 선택
}

ipadapter_body = {
    'ipadapter': 'SDXL_ip-adapter-plus_sdxl_vit-h.safetensors', # 입력 고정
    'clip_vision': 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors', # 입력 고정
    'images': [IMAGE_BASE64, IMAGE_BASE64], # RGB Image list
    'weight': 0.7, # 입력 선택
    'start_at': 0, # 입력 선택
    'end_at': 0.4, # 입력 선택
}

if refiner_enable:
    request_body.update(refiner_body)
if canny_enable :
    request_body['controlnet_requests'].append(canny_body)
if inpaint_enable :
    request_body['controlnet_requests'].append(inpaint_body)
if ipadapter_enable :
    request_body['ipadapter_request'] = ipadapter_body
        
url = f"http://{ip_addr}:7861/sdxl/generate"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```

#### 2.2.3 SD15
Parameter format
```python
class SD15_RequestData(RequestData):
    basemodel: str
    steps: int = 20
    cfg: float = 7
    prompt_negative: str = prompt_negative
    sampler_name: str = 'dpmpp_sde'
    scheduler: str = 'karras'
    init_image: Optional[str]= None
    mask: Optional[str]= None
    controlnet_requests: Optional[List[ControlNet_RequestData]] = []
    ipadapter_request: Optional[IPAdapter_RequestData] = None
    gen_type: Literal['t2i', 'i2i', 'inpaint'] = 't2i'
```

**SD15 Image Generation Example**
```python
request_body = {
    'basemodel': 'SD15_realisticVisionV51_v51VAE.safetensors', # 입력 고정
    'init_image': IMAGE_BASE64, # RGB Image / i2i, inpaint 일 경우 필요
    'mask': MASK_BASE64, # RGB Image / inpaint 일 경우 필요
    "prompt_positive": POSITIVE_PROMPT, # 입력 필요
    "prompt_negative": NEGATIVE_PROMPT, # 입력 선택
    'width': WIDTH, # t2i는 입력 필요
    'height': HEIGHT, # t2i는 입력 필요
    'steps': 20, # 입력 선택
    'cfg': 7, # 입력 선택
    'denoise': 1.0, 
    'gen_type': 't2i', # 't2i' or 'i2i' or 'inpaint'
    'controlnet_requests': [],
}

canny_body = {
    'controlnet': 'SD15_Canny_control_v11p_sd15_canny.pth',  # 입력 고정
    'type': 'canny',
    'image': IMAGE_BASE64, # RGB Image
    'strength': 1., # 입력 필수
    'start_percent': 0.03, # 입력 필수
    'end_percent': 1, # 입력 필수
}

inpaint_body = { # Controlnet Inpaint body
    'controlnet': 'SD15_Inpaint_control_v11p_sd15_inpaint.pth.safetensors',
    'type': 'inpaint',
    'image': IMAGE_BASE64 # RGB Image, 채워질 부분 검은색(0) 값으로
    'strength': 1.3, # 입력 필수
    'start_percent': 0.31, # 입력 필수
    'end_percent': 0.86, # 입력 필수
}

ipadapter_body = {
    'ipadapter': 'SD15_ip-adapter-plus_sd15.safetensors', # 입력 고정
    'clip_vision': 'CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors', # 입력 고정
    'images': [IMAGE_BASE64, IMAGE_BASE64], # RGB Image list
    'weight': 1, # 입력 필수
    'start_at': 0.27, # 입력 필수
    'end_at': 1, # 입력 필수
}

if canny_enable :
    request_body['controlnet_requests'].append(canny_body)
if inpaint_enable :
    request_body['controlnet_requests'].append(inpaint_body)
if ipadapter_enable :
    request_body['ipadapter_request'] = ipadapter_body
        
url = f"http://{ip_addr}:7861/sd15/generate"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```



#### 2.2.4 Object Removal
Parameter format
```python
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
```
**Object Removal Example**
```python
request_body = {
    'basemodel': 'SDXL_copaxTimelessxlSDXL1_v12.safetensors', # 입력 고정
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



#### 2.2.5 Up-scale
Parameter format
```python
class Upscale_RequestData(RequestData):
    upscale_model: str
    init_image: str
    method: Literal['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos'] = 'lanczos'
    scale: float = 2
    controlnet_requests: list = [] # 필요없는데 model cache update 때문에 넣음.
```

**Up-scale Example**
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

#### 2.2.6 IC-Light
Parameter format
```python
class ICLight_RequestData(RequestData):
    basemodel: str
    init_image: str
    mask: str
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
```

**SD15 Image Generation Example**
```python
request_body = {
    'basemodel': 'SD15_epicrealism_naturalSinRC1VAE.safetensors',
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

    'controlnet_requests': [],
}

url = f"http://{ip_addr}:7861/iclight/generate"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```



#### 2.2.7 Segment Anything
작업중...
#### 2.2.8 Gemini
Parameter format
```python
class Gemini_RequestData(BaseModel) :
    user_prompt: str
    query: Optional[str] = ''
    user_image: Optional[str] = None
```

**Gemini Example**
```python
request_body = {
    'user_prompt': user_prompt, # 입력 필수
    'query': query, # 입력 필수
    'user_image': user_image # 입력 선택
}

url = f"http://{ip_addr}:7861/gemini"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
#### 2.2.9 Flux
작업중...


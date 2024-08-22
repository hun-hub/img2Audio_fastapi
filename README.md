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
* SDXL
* SD15
* Object Removal
* Up-scale
* IC-Light
* Segment Anything (coming soon)
* Gemini
* Flux (coming soon)

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

**Text-to-Image Example**
```python
request_body = {
    'basemodel': 'SD3_sd3_medium_incl_clips_t5xxlfp16.safetensors', # 입력 고정
    "prompt_positive": POSITIVE_PROMPT, # 입력 필요
    "prompt_negative": NEGATIVE_PROMPT, # 입력 선택
    'width': WIDTH, # 입력 필요
    'height': HEIGHT, # 입력 필요
    'steps': 28, # 입력 선택
    'cfg': 4, # 입력 선택
    'gen_type': 't2i' # 입력 고정
}
url = f"http://{ip_addr}:7861/sd3/generate"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
**Text-to-Image + Canny Controlnet Example**
```python
request_body = {
    'basemodel': 'SD3_sd3_medium_incl_clips_t5xxlfp16.safetensors', # 입력 고정
    "prompt_positive": POSITIVE_PROMPT, # 입력 필요
    "prompt_negative": NEGATIVE_PROMPT, # 입력 선택
    'width': WIDTH, # 입력 필요
    'height': HEIGHT, # 입력 필요
    'steps': 28, # 입력 선택
    'cfg': 4, # 입력 선택
    'gen_type': 't2i', # 입력 고정
    'controlnet_requests' : [
        {
            'controlnet': 'SD3_Canny.safetensors', # 입력 고정
            'type': 'canny', # 입력 고정
            'image': IMAGE_BASE64, # 입력 필요
            'strength': 0.7, # 입력 선택
            'start_percent': 0.0, # 입력 선택
            'end_percent': 0.4, # 입력 선택
        }
    ]
}
url = f"http://{ip_addr}:7861/sd3/generate"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
# Jector Service Functions

**Board**
* [Text to image Generation (키워드로 시작)](#text-to-image-generation-키워드로-시작)
* [Text to image Generation + IPAdapter (레퍼런스로 시작)](#text-to-image-generation--ipadapter-레퍼런스로-시작)
* [Inpaint + ControlNet-Canny (Synthesis 1, 2 단계)](#inpaint--controlnet-canny-synthesis-1-2-단계)
* [Inpaint + ControlNet-Canny + IPAdapter (Synthesis 3, 4 단계)](#inpaint--controlnet-canny--ipadapter-synthesis-3-4-단계)
* [Inpaint + ControlNet-Canny (합성하기 이후 변형 process)](#inpaint--controlnet-canny-합성하기-이후-변형-process)
* [Object Removal](#object-removal)
* [Upscale](#upscale)

**Gallary**
* [theme](#theme)
* [theme + reference](#theme--reference)


## Text to image Generation (키워드로 시작)
```python
request_body = {
    'checkpoint': 'SDXL_copaxTimelessxlSDXL1_v12.safetensors', 
    "prompt_positive": POSITIVE_PROMPT,
    "prompt_negative": 'worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch', # 입력 선택
    'width': 1024,
    'height': 1024, 
    'steps': 20, 
    'cfg': 7, 
    'denoise': 1.0, 
    'gen_type': 't2i',
    'controlnet_requests': [],
}

url = f"http://{ip_addr}:{port}/sdxl/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```

## Text to image Generation + IPAdapter (레퍼런스로 시작)
```python
request_body = {
    'checkpoint': 'SDXL_copaxTimelessxlSDXL1_v12.safetensors', 
    "prompt_positive": POSITIVE_PROMPT,
    "prompt_negative": 'worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch', # 입력 선택
    'width': 1024, 
    'height': 1024, 
    'steps': 20, 
    'cfg': 7, 
    'denoise': 1.0, 
    'gen_type': 't2i', 
    'controlnet_requests': [],
}

ipadapter_body = {
    'ipadapter': 'SDXL_ip-adapter-plus_sdxl_vit-h.safetensors',
    'clip_vision': 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors', 
    'images': [IMAGE_BASE64], 
    'weight': 0.4, 
    'start_at': 0, 
    'end_at': 0.4, 
}
request_body['ipadapter_request'] = ipadapter_body
        
url = f"http://{ip_addr}:{port}/sdxl/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```

## Inpaint + ControlNet-Canny (Synthesis 1, 2 단계)
```python
request_body = {
    'checkpoint': 'SDXL_copaxTimelessxlSDXL1_v12.safetensors',
    'init_image': IMAGE_BASE64,
    'mask': MASK_BASE64, 
    "prompt_positive": POSITIVE_PROMPT, 
    "prompt_negative": 'worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch', # 입력 선택
    'steps': 20, 
    'cfg': 7,
    'denoise': 0.6, # 1단계는 0.6, 2단계는 0.9
    'gen_type': 'inpaint', 
    'controlnet_requests': [],
}

canny_body = {
    'controlnet': 'SDXL_Canny_jector.safetensors',  
    'type': 'canny',
    'image': IMAGE_BASE64,
    'strength': 0.75, 
    'start_percent': 0, 
    'end_percent': 0.5, 
}
request_body['controlnet_requests'].append(canny_body)

url = f"http://{ip_addr}:{port}/sdxl/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```

## Inpaint + ControlNet-Canny + IPAdapter (Synthesis 3, 4 단계)
```python
request_body = {
    'checkpoint': 'SDXL_copaxTimelessxlSDXL1_v12.safetensors', 
    'init_image': IMAGE_BASE64, 
    'mask': MASK_BASE64, 
    "prompt_positive": POSITIVE_PROMPT, 
    "prompt_negative": 'worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch', 
    'steps': 20, 
    'cfg': 7, 
    'denoise': 1.0, 
    'gen_type': 't2i', 
    'controlnet_requests': [],
}

canny_body = {
    'controlnet': 'SDXL_Canny_jector.safetensors', 
    'type': 'canny',
    'image': IMAGE_BASE64, 
    'strength': 0.75, 
    'start_percent': 0, 
    'end_percent': 0.5, 
}
ipadapter_body = {
    'ipadapter': 'SDXL_ip-adapter-plus_sdxl_vit-h.safetensors', 
    'clip_vision': 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors', 
    'images': [IMAGE_BASE64], 
    'weight': 0.8, # 3단계 0.8, 4단계 0.4
    'start_at': 0, 
    'end_at': 0.4, 
}
request_body['controlnet_requests'].append(canny_body)
request_body['ipadapter_request'] = ipadapter_body
        
url = f"http://{ip_addr}:{port}/sdxl/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
## Inpaint + ControlNet-Canny (합성하기 이후 변형 process)
```python
'''
denoising range [0.4, 1] -> [1, 0.5] 로 linear 하게 mapping
f(x) = -5/6 * denoise + 4/3
'''
denoise = -5/6 * denoise + 4/3
request_body = {
    'checkpoint': 'SDXL_copaxTimelessxlSDXL1_v12.safetensors',
    'init_image': IMAGE_BASE64,
    'mask': MASK_BASE64, 
    "prompt_positive": POSITIVE_PROMPT, 
    "prompt_negative": 'worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch', # 입력 선택
    'steps': 20, 
    'cfg': 7,
    'denoise': denoise,
    'gen_type': 'inpaint', 
    'controlnet_requests': [],
}

canny_body = {
    'controlnet': 'SDXL_Canny_jector.safetensors',  
    'type': 'canny',
    'image': IMAGE_BASE64,
    'strength': 0.75, 
    'start_percent': 0, 
    'end_percent': 0.5, 
}
request_body['controlnet_requests'].append(canny_body)

url = f"http://{ip_addr}:{port}/sdxl/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```

## Object Removal
```python
request_body = {
    'checkpoint': 'SDXL_copaxTimelessxlSDXL1_v12.safetensors', 
    'inpaint_model_name': 'SDXL_inpaint_v26.fooocus.patch', 
    'init_image': IMAGE_BASE64, 
    'mask': MASK_BASE64, 
    "prompt_positive": "",
    'steps': 10,
    'cfg': 3,
    'denoise': 1,
}
        
url = f"http://{ip_addr}:{port}/object_remove"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
## Upscale
```python
request_body = {
    'upscale_model': '4x-UltraSharp.pth',
    'init_image': IMAGE_BASE64, # RGB Image 
    'method': 'lanczos',
    "scale": 2, # range: [2, 4]
}

url = f"http://{ip_addr}:{port}/upscale"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
## theme
```python
request_body = {
    'checkpoint': 'SD15_realisticVisionV51_v51VAE.safetensors', # 입력 고정
    'init_image': IMAGE_BASE64, 
    'mask': MASK_BASE64, 
    "prompt_positive": POSITIVE_PROMPT, 
    "prompt_negative": NEGATIVE_PROMPT,
    'steps': 20, 
    'cfg': 7, 
    'denoise': 1.0, 
    'gen_type': 'inpaint', 
    'controlnet_requests': [],
}

canny_body = {
    'controlnet': 'SD15_Canny_control_v11p_sd15_canny.pth',  # 입력 고정
    'type': 'canny',
    'image': IMAGE_BASE64, 
    'strength': 1.3, 
    'start_percent': 0., 
    'end_percent': 1, 
}

inpaint_body = { 
    'controlnet': 'SD15_Inpaint_control_v11p_sd15_inpaint.pth.safetensors',
    'type': 'inpaint',
    'image': IMAGE_BASE64, # inpaint 할 부분 검은색 (0) 값으로. 
    'strength': 1.3, 
    'start_percent': 0., 
    'end_percent': 1, 
}

request_body['controlnet_requests'].append(canny_body)
request_body['controlnet_requests'].append(inpaint_body)

        
url = f"http://{ip_addr}:{port}/sd15/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
## theme + reference
```python
request_body = {
    'checkpoint': 'SD15_realisticVisionV51_v51VAE.safetensors', # 입력 고정
    'init_image': IMAGE_BASE64, 
    'mask': MASK_BASE64, 
    "prompt_positive": POSITIVE_PROMPT, 
    "prompt_negative": NEGATIVE_PROMPT,
    'steps': 20, 
    'cfg': 7, 
    'denoise': 1.0, 
    'gen_type': 'inpaint', 
    'controlnet_requests': [],
}

canny_body = {
    'controlnet': 'SD15_Canny_control_v11p_sd15_canny.pth',  # 입력 고정
    'type': 'canny',
    'image': IMAGE_BASE64, 
    'strength': 1., 
    'start_percent': 0.03, 
    'end_percent': 1, 
}

inpaint_body = { 
    'controlnet': 'SD15_Inpaint_control_v11p_sd15_inpaint.pth.safetensors',
    'type': 'inpaint',
    'image': IMAGE_BASE64, # inpaint 할 부분 검은색 (0) 값으로. 
    'strength': 1.3, 
    'start_percent': 0.31, 
    'end_percent': 0.86, 
}

ipadapter_body = {
    'ipadapter': 'SD15_ip-adapter-plus_sd15.safetensors', # 입력 고정
    'clip_vision': 'CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors', # 입력 고정
    'images': [IMAGE_BASE64, IMAGE_BASE64, IMAGE_BASE64], # RGB Image list
    'weight': 1, # 입력 필수
    'start_at': 0.27, # 입력 필수
    'end_at': 1, # 입력 필수
}
request_body['controlnet_requests'].append(canny_body)
request_body['controlnet_requests'].append(inpaint_body)
request_body['ipadapter_request'] = ipadapter_body

        
url = f"http://{ip_addr}:{port}/sd15/generate"

response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
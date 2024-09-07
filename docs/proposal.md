# Proposal Service Functions

## Model list
FLUX도 추가 가능.

| SD type | Model Name  | 
|---------|-------------|
| FLUX    | FLUX_flux1-dev.safetensors |
| SD3     | SD3_sd3_medium_incl_clips_t5xxlfp16 |
| SDXL    | SDXL_RealVisXL_V40.safetensors |
| SDXL    | SDXL_jector_core_Dtype3_Ptype3_lr4e-7_bs48_Tddp_Fv-param_step00590000.safetensors |


### Text to Image Generation Example

SD3, SDXL case
```python
request_body = {
    'checkpoint': MODEL_NAME, 
    "prompt_positive": POSITIVE_PROMPT,
    "prompt_negative": 'worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch', 
    'width': WIDTH, 
    'height': HEIGHT, 
    'steps': 20, 
    'cfg': 7, # SD3 는 5
    'denoise': 1.0, 
    'gen_type': 't2i', 
}

url = f"http://{IP_Addr}:{Port}/sd3/generate"
url = f"http://{IP_Addr}:{Port}/sdxl/generate"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```

FLUX case
```python
# Model name default 값으로 되어있음.
request_body = {
    "prompt_positive": POSITIVE_PROMPT,
    "prompt_negative": 'worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch', 
    'width': WIDTH, 
    'height': HEIGHT, 
    'steps': 20, 
    'cfg': 3.5,
    'denoise': 1.0, 
    'gen_type': 't2i', 
}

url = f"http://{IP_Addr}:{Port}/flux/generate"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
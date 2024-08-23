# IC-Light

```python
cfg = 4
denoise = 0.85
if keep_background == False :
    cfg = 1.5
    denoise = 0.9

request_body = {
    'basemodel': 'SD15_epicrealism_naturalSinRC1VAE.safetensors',
    'init_image': IMAGE_BASE64, 
    'mask': MASK_BASE64, 
    "prompt_positive": POSITIVE_PROMPT, 
    "prompt_negative": NEGATIVE_PROMPT, 
    'steps': 30,
    'cfg': cfg,
    'denoise': denoise,

    'iclight_model': 'SD15_iclight_sd15_fc.safetensors', 
    'light_condition': IMAGE_BASE64, 
    'light_strength': 0.5, # 0 ~ 1 사이 float
    'keep_background': keep_background,
    'blending_mode_1': 'color',
    'blending_percentage_1': 0.1,
    'blending_mode_2': 'hue',
    'blending_percentage_2': 0.2,
    'remap_min_value': -0.15,
    'remap_max_value': 1.14,

    'controlnet_requests': [],
}

url = f"http://{ip_addr}:7861/iclight/generate"
response = requests.post(url, json=request_body)
data = response.json()
image_base64 = data['image_base64'] 
```
# Connectbrick 자체 API 및 Demo
* 학습/개발한 모델 test 및 호출 하는 것을 목표로함.
* 새로운 프로젝트에 대해서도 코드 업데이트를 통해 Inference 환경 구축.
* Stable Diffusion 기반 최신 모델 쉽게 실행 가능하도록.

## 1. Base Function
* T2I
* I2I
* Inpainting
* T2V
* I2V

### 1.1 Adapter
* ControlNet
* IC-Light
* Textural Inversion



## 2. Gemini API
### 2.1 Request Format & Example
```python
from pydantic import BaseModel
from typing import Optional, List, Literal

class Gemini_RequestData(BaseModel) :
    user_prompt: str
    query: Optional[str]
    user_image: Optional[str] = None
```

**Request Example**
### Image Refinement
```python
import requests

ip_addr = '34.134.200.251'
url = f"http://{ip_addr}:7861"
gemini_url =  f'{url}/gemini'

query = """You need to refine user's input prompt. 
Users inputs are set of simple keywords for text to image generation.
Keywords are {user_prompt}.
Here are some examples of refined prompts:
1. flower, water, cyan perfumes bottle, still life, reflection, scenery, cosmetics, reality, 4K, nobody, creek, product photography, forest background
2. Product photography, two bottles of perfume and one bottle on the table surrounded by flowers, with soft lighting and a warm color tone. The background is a beige wall decorated with green plants, a table topped with bottles of perfume next to flowers and greenery on a table cloth covered tablecloth,
3. Product photography, a perfume bottle, in the style of floral art, horizontal composition, dreamy pastel color palette, serene floral theme, beautiful sunlight and shadow
Harmoniously combine all provided keywords into a detailed and visually appealing final description.
Refined prompt's length should be less than 60 characters.
"""

# Prompt Refinement
prompt_refinement_body = {
    'user_prompt': 'flower, space, moon, sun, blackhole',
    'query': query
}
response = requests.post(gemini_url, json=prompt_refinement_body)
data = response.json()
prompt = data['prompt']
```
### Image Caption
```python
# Image Caption
from PIL import Image
import io, base64
import requests

ip_addr = '34.134.200.251'
url = f"http://{ip_addr}:7861"
gemini_url =  f'{url}/gemini'

def convert_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return image_base64

image = Image.open('path/to/the/image.png')
image_base64 = convert_image_to_base64(image)
image_caption_body = {
    'user_prompt': '',
    'query': query,
    'image': image_base64
}
response = requests.post(gemini_url, json=image_caption_body)
data = response.json()
prompt = data['prompt']
```
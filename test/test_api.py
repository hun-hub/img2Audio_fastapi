import asyncio
import time


async def sleep():
    await time.sleep(1)


async def sum(name, numbers):
    start = time.time()
    total = 0
    for number in numbers:
        await sleep()
        total += number
        print(f'작업중={name}, number={number}, total={total}')
    end = time.time()
    print(f'작업명={name}, 걸린시간={end-start}')
    return total


async def main():
    start = time.time()

    task1 = asyncio.create_task(sum("A", [1, 2]))
    task2 = asyncio.create_task(sum("B", [1, 2, 3]))

    await task1
    await task2

    result1 = task1.result()
    result2 = task2.result()

    end = time.time()
    print(f'총합={result1+result2}, 총시간={end-start}')

import requests
from utils.handler import handle_response
if __name__ == "__main__":
    ip_addr = '117.52.72.82'

    request_body = {
        'checkpoint': 'SDXL_forrealxl_v05.safetensors',
        "prompt_positive": "cute tiny stribul monster in floating village lake africa , by Jon Klassen, Chris LaBrooy, Guy Denning, Wes Anderson, Surprising, natural light, Orton effect, Nikon Z9, F/8, yellow orange ultramarine blue brown hues, film grain cinematic lighting, 35mm film, atmospheric mo’ai",  # 입력 필요
        "prompt_negative": 'worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch',
        'width': 1344,
        'height': 768,
        'steps': 20,
        'cfg': 2,
        'denoise': 1.0,
        'gen_type': 't2i',
        'lora_requests': [],
    }

    lora_body_list = [
        {'lora': 'SDXL_MJ52.safetensors',
         'strength_model': 0.5,
         'strength_clip': 1, },
        {'lora': 'SDXL_add-detail-xl.safetensors',
         'strength_model': 1,
         'strength_clip': 1, },
    ]

    for lora_body in lora_body_list:
        request_body['lora_requests'].append(lora_body)

    url = f"http://{ip_addr}:7863/sdxl/generate"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
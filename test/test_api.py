from cgen_utils import set_comfyui_packages
set_comfyui_packages()

import asyncio
import time
from cgen_utils.image_process import convert_image_to_base64
from PIL import Image
import httpx
import random
import requests
from cgen_utils.handler import handle_response



async def sdxl_request(style_type, image, ip_addr='localhost:7861') :

    request_body = {
        "prompt_positive": 'cat',
        "prompt_negative": "",
        'controlnet_requests': [],
        'lora_requests': [],
    }

    url = f"http://{ip_addr}/sdxl/generate"

    async with httpx.AsyncClient(timeout=1000.0) as client:
        response = await client.post(url, json=request_body)
        data = handle_response(response)
        image_base64 = data['image_base64']
        return image_base64

async def sd15_request(style_type, image, ip_addr='localhost:7861') :

    request_body = {
        "prompt_positive": 'cat',
        "prompt_negative": "",
        'controlnet_requests': [],
        'lora_requests': [],
    }

    url = f"http://{ip_addr}/sd15/generate"

    async with httpx.AsyncClient(timeout=1000.0) as client:
        response = await client.post(url, json=request_body)
        data = handle_response(response)
        image_base64 = data['image_base64']
        return image_base64

async def upscale_request(image, scale=2, ip_addr='localhost:7861'):
    request_body = {
        'init_image': image,
        "scale": scale,
    }

    url = f"http://{ip_addr}/upscale"
    async with httpx.AsyncClient(timeout=1000.0) as client:
        response = await client.post(url, json=request_body)
        data = handle_response(response)
        image_base64 = data['image_base64']
        return image_base64



async def resize_request(image, ip_addr='localhost:7861') :
    request_body = {
        'image': image,
    }
    url = f"http://{ip_addr}/functions/resize_image_for_sd"
    async with httpx.AsyncClient(timeout=1000.0) as client:
        response = await client.post(url, json=request_body)
        data = handle_response(response)
        image_base64 = data['image_base64']
        return image_base64

async def test_process(style_type, image_base64, idx) :
    image_base64 = await sdxl_request(style_type, image_base64)
    print(f'success for {idx} task sdxl_request')

    image_base64 = await upscale_request(image_base64)
    print(f'success for {idx} task style_type upscale')

    image_base64 = await sd15_request(style_type, image_base64)
    print(f'success for {idx} task sd15_request')


async def main():

    image_path = '/home/gkalstn000/test.jpg'
    image = Image.open(image_path)
    image_base64 = convert_image_to_base64(image)

    # 10명의 사용자가 랜덤으로 요청을 보내는 것을 시뮬레이션
    tasks = []
    for i in range(2):  # 여기서 10은 사용자 수를 의미 (원하는 대로 조절 가능)
        style_type = random.choice(['type_1', 'type_2'])  # 랜덤으로 스타일 선택
        tasks.append(test_process(style_type, image_base64, i))

    # 비동기적으로 모든 요청을 실행
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

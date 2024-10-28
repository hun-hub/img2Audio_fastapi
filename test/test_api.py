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



async def i2c_request(style_type, image, ip_addr='localhost:7861') :

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

    url = f"http://{ip_addr}/i2c/generate"

    async with httpx.AsyncClient(timeout=1000.0) as client:
        response = await client.post(url, json=request_body)
        data = handle_response(response)
        image_base64 = data['image_base64']
        image_face_detail_base64 = data['image_face_detail_base64']
        return image_base64, image_face_detail_base64

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
    image_base64 = await upscale_request(image_base64)
    print(f'success for {idx} task style_type upscale')
    image_base64 = await resize_request(image_base64)
    print(f'success for {idx} task style_type resize')
    image_base64, image_detailed_base64 = await i2c_request(style_type, image_base64)
    print(f'success for {idx} task i2c {style_type}')

async def main():

    image_path = '/home/gkalstn000/test.jpg'
    image = Image.open(image_path)
    image_base64 = convert_image_to_base64(image)

    # 10명의 사용자가 랜덤으로 요청을 보내는 것을 시뮬레이션
    tasks = []
    for i in range(10):  # 여기서 10은 사용자 수를 의미 (원하는 대로 조절 가능)
        style_type = random.choice(['type_1', 'type_2'])  # 랜덤으로 스타일 선택
        tasks.append(test_process(style_type, image_base64, i))

    # 비동기적으로 모든 요청을 실행
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

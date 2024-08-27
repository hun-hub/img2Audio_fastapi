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
    ip_addr = '117.52.72.83'

    request_body = {
        'basemodel': 'SD3_sd3_medium_incl_clips_t5xxlfp16.safetensors',
        "prompt_positive": 'a dog',
        "prompt_negative": "",
        'width': 1344,
        'height': 768,
        'steps': 20,
        'cfg': 4,
        'denoise': 1,
        'gen_type': 't2i'
    }

    url = f"http://{ip_addr}:7863/sd3/generate"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
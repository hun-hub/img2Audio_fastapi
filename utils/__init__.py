import os
import sys
from dotenv import load_dotenv
from tabulate import tabulate
from typing import List, Dict, Any
import pandas as pd

blue_print = {'sd_checkpoint':
                  {'basemodel': {},
                   'refiner': {}},
              'controlnet': {f'module_{i}': {} for i in range(3)},
              'ipadapter': {f'module_{i}': {} for i in range(3)}
              }

resolution_list = [
    "(512, 512)",
    "(704, 1408)",
    "(704, 1344)",
    "(768, 1344)",
    "(768, 1280)",
    "(832, 1216)",
    "(832, 1152)",
    "(896, 1152)",
    "(896, 1088)",
    "(960, 1088)",
    "(960, 1024)",
    "(1024, 1024)",
    "(1024, 960)",
    "(1088, 960)",
    "(1088, 896)",
    "(1152, 896)",
    "(1152, 832)",
    "(1216, 832)",
    "(1280, 768)",
    "(1344, 768)",
    "(1344, 704)",
    "(1408, 704)",
    "(1472, 704)",
    "(1536, 640)",
    "(1600, 640)",
    "(1664, 576)",
    "(1728, 576)",
]

def set_comfyui_packages() :
    load_dotenv()
    COMFYUI_PATH = os.getenv('COMFYUI_PATH')
    sys.path.insert(0, os.path.abspath(COMFYUI_PATH))
    print(f'Setting ComfyUI Packages PATH to {COMFYUI_PATH}')


def compare_cache_and_request_difference(cached, request):
    differences = {'sd_checkpoint': {}, 'controlnet': {}, 'ipadapter': {}}
    table = []

    # Compare sd_checkpoint
    for key in cached['sd_checkpoint']:
        cache_value = cached['sd_checkpoint'][key][0]
        request_value = request['sd_checkpoint'].get(key, None)
        table.append([f'sd_checkpoint/{key}', cache_value, request_value])
        if request_value and cache_value != request_value:
            differences['sd_checkpoint'][key] = {'cached': cache_value, 'requested': request_value}

    # Compare controlnet
    for key in cached['controlnet']:
        cache_value = cached['controlnet'][key][0]
        request_value = request['controlnet'].get(key, None)
        table.append([f'controlnet/{key}', cache_value, request_value])
        if request_value and cache_value != request_value :
            differences['controlnet'][key] = {'cached': cache_value, 'requested': request_value}

    # Compare ipadapter
    for key in cached['ipadapter']:
        cache_value = cached['ipadapter'][key][0]
        request_value = request['ipadapter'].get(key, None)
        table.append([f'ipadapter/{key}', cache_value, request_value])
        if request_value and cache_value != request_value:
            differences['ipadapter'][key] = {'cached': cache_value, 'requested': request_value}

    print(tabulate(table, headers=['Module', 'Cached', 'Requested'], tablefmt='pretty'))
    return differences

import os
import sys
from dotenv import load_dotenv
from tabulate import tabulate
from typing import List, Dict, Any
import pandas as pd
import gc
from prettytable import PrettyTable


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

def update_model_cache_from_blueprint(model_cache, model_cache_blueprint):
    # 테이블을 생성하여 변경된 내용을 기록
    changes_table = PrettyTable()
    changes_table.field_names = ["Model Type", "SD Type", "Subcategory", "Old Value", "New Value"]

    def recursive_update(cache, blueprint, model_type, category=None):
        for key_request, value_request in blueprint.items():
            if isinstance(value_request, dict):
                new_category = key_request if category is None else category
                recursive_update(cache[key_request], value_request, model_type, new_category)
            elif value_request :
                value_cached = cache.get(key_request)

                category = key_request if category is None else category
                sub_category = None if category == key_request else key_request

                if isinstance(value_request, tuple) and value_cached is None:
                    model_name_request, module_request = value_request
                    changes_table.add_row([model_type, category, sub_category, None, model_name_request])

                    cache[key_request] = value_request
                elif isinstance(value_request, tuple) and isinstance(value_cached, tuple):
                    model_name_cached, module_cached = value_cached
                    model_name_request, module_request = value_request
                    if model_name_cached != model_name_request:
                        changes_table.add_row([model_type, category, sub_category, model_name_cached, model_name_request])

                    del cache[key_request]
                    cache[key_request] = value_request

    # 각 모델 타입에 대해 업데이트 및 변경 사항 추적
    for model_type in model_cache.keys():
        recursive_update(model_cache[model_type], model_cache_blueprint[model_type], model_type)
    print(changes_table)

def cache_checkpoint(model_cache, checkpoint_name, is_refiner=False) :
    from utils.loader import load_checkpoint
    checkpoint_type = checkpoint_name.split('_')[0].lower()
    unet, vae, clip, _ = load_checkpoint(checkpoint_name)

    if checkpoint_type == 'sdxl' :
        sdxl_type = 'refiner' if is_refiner else 'base'
        model_cache['unet'][checkpoint_type][sdxl_type] = (checkpoint_name, unet)
        model_cache['vae'][checkpoint_type][sdxl_type] = (checkpoint_name, vae)
        model_cache['clip'][checkpoint_type][sdxl_type] = (checkpoint_name, clip)
    else :
        model_cache['unet'][checkpoint_type] = (checkpoint_name, unet)
        model_cache['vae'][checkpoint_type] = (checkpoint_name, vae)
        model_cache['clip'][checkpoint_type] = (checkpoint_name, clip)

def cache_unet(model_cache, unet) :
    pass
def cache_vae(model_cache, vae) :
    pass
def cache_clip(model_cache, clip) :
    pass
def cache_clip_vision(model_cache, clip_vision) :
    pass
def cache_controlnet(model_cache, controlnet_requests) :
    from utils.loader import load_controlnet
    for controlnet_request in controlnet_requests :
        control_model = controlnet_request['controlnet']
        control_type = controlnet_request['type']
        checkpoint_type = control_model.split('_')[0].lower()
        controlnet = load_controlnet(control_model)
        model_cache['controlnet'][control_type][checkpoint_type] = (control_model, controlnet)
def cache_ipadapter(model_cache, ipadapter_request) :
    from utils.loader import load_ipadapter
    ipadapter_model = ipadapter_request['ipadapter']
    ipadapter_type = ipadapter_model.split('_')[0].lower()
    ipadapter = load_ipadapter(ipadapter_model)
    model_cache['ipadapter'][ipadapter_type] = (ipadapter_model, ipadapter)

def cache_lora(model_cache, lora_requests) :
    from utils.loader import load_lora
    lora_type = lora_requests[0]['lora'].split('_')[0].lower()
    for i, lora_request in enumerate(lora_requests) :
        lora_model = lora_request['lora']
        lora = load_lora(lora_model)
        model_cache['lora'][f'module_{i+1}'][lora_type] = (lora_model, lora)

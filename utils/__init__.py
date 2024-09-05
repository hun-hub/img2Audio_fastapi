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

def cache_checkpoint(model_cache, model_cache_blueprint, checkpoint_name, is_refiner=False) :
    from utils.loader import load_checkpoint
    checkpoint_type = checkpoint_name.split('_')[0].lower()

    if checkpoint_type == 'sdxl' :
        sdxl_type = 'refiner' if is_refiner else 'base'
        cached_unet = model_cache['unet'][checkpoint_type][sdxl_type]
    else :
        cached_unet = model_cache['unet'][checkpoint_type]

    if cached_unet is not None and cached_unet[0] == checkpoint_name :
        return None

    unet, vae, clip, _ = load_checkpoint(checkpoint_name)

    if checkpoint_type == 'sdxl' :
        sdxl_type = 'refiner' if is_refiner else 'base'
        model_cache_blueprint['unet'][checkpoint_type][sdxl_type] = (checkpoint_name, unet)
        model_cache_blueprint['vae'][checkpoint_type][sdxl_type] = (checkpoint_name, vae)
        model_cache_blueprint['clip'][checkpoint_type][sdxl_type] = (checkpoint_name, clip)
    else :
        model_cache_blueprint['unet'][checkpoint_type] = (checkpoint_name, unet)
        model_cache_blueprint['vae'][checkpoint_type] = (checkpoint_name, vae)
        model_cache_blueprint['clip'][checkpoint_type] = (checkpoint_name, clip)

def cache_unet(model_cache, model_cache_blueprint, unet_name) :
    from ComfyUI.nodes import UNETLoader
    unet_type = unet_name.split('_')[0].lower()

    cached_unet = model_cache['unet'][unet_type]
    if cached_unet is not None and cached_unet[0] == unet_name :
        return None

    unet_loader = UNETLoader()
    unet = unet_loader.load_unet(unet_name, 'fp8_e4m3fn')[0]
    model_cache_blueprint['unet'][unet_type] = (unet_name, unet)

def cache_vae(model_cache, model_cache_blueprint, vae_name) :
    from ComfyUI.nodes import VAELoader
    vae_type = vae_name.split('_')[0].lower()

    cached_vae = model_cache['vae'][vae_type]
    if cached_vae is not None and cached_vae[0] == vae_name:
        return None

    vae_loader = VAELoader()
    vae = vae_loader.load_vae(vae_name)[0]
    model_cache_blueprint['vae'][vae_type] = (vae_name, vae)

# TODO: 일단 CLIP 따로 load하는 case는 FLUX로 가정.
def cache_clip(model_cache, model_cache_blueprint, clip_name) :
    from ComfyUI.nodes import DualCLIPLoader, CLIPLoader
    if isinstance(clip_name, tuple) :
        clip_type = 'flux'

        cached_clip = model_cache['clip'][clip_type]
        if cached_clip is not None and cached_clip[0] == clip_name:
            return None

        clip_loader = DualCLIPLoader()
        clip = clip_loader.load_clip(clip_name[0], clip_name[1], clip_type)[0]
        model_cache_blueprint['clip'][clip_type] = (clip_name, clip)
def cache_clip_vision(model_cache, clip_vision) :
    pass
def cache_controlnet(model_cache, model_cache_blueprint, controlnet_requests) :
    from utils.loader import load_controlnet
    for controlnet_request in controlnet_requests :
        control_model = controlnet_request['controlnet']
        control_type = controlnet_request['type']
        checkpoint_type = control_model.split('_')[0].lower()

        cached_controlnet = model_cache['controlnet'][control_type][checkpoint_type]
        if cached_controlnet is not None and cached_controlnet[0] == control_model:
            continue

        controlnet = load_controlnet(control_model)
        model_cache_blueprint['controlnet'][control_type][checkpoint_type] = (control_model, controlnet)
def cache_ipadapter(model_cache, model_cache_blueprint, ipadapter_request) :
    from utils.loader import load_ipadapter
    ipadapter_model = ipadapter_request['ipadapter']
    ipadapter_type = ipadapter_model.split('_')[0].lower()

    cached_ipadapter = model_cache['ipadapter'][ipadapter_type]
    if cached_ipadapter is not None and cached_ipadapter[0] == ipadapter_model:
        return None

    ipadapter = load_ipadapter(ipadapter_model)
    model_cache_blueprint['ipadapter'][ipadapter_type] = (ipadapter_model, ipadapter)

def cache_lora(model_cache, model_cache_blueprint, lora_requests) :
    from utils.loader import load_lora
    lora_type = lora_requests[0]['lora'].split('_')[0].lower()
    for i, lora_request in enumerate(lora_requests) :
        lora_model = lora_request['lora']

        cached_lora = model_cache['lora'][f'module_{i+1}'][lora_type]
        if cached_lora is not None and cached_lora[0] == lora_model:
            continue

        lora = load_lora(lora_model)
        model_cache_blueprint['lora'][f'module_{i+1}'][lora_type] = (lora_model, lora)

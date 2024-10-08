from cgen_utils import set_comfyui_packages
set_comfyui_packages()

import os, json
from PIL import Image
from cgen_utils.loader import load_extra_path_config, load_checkpoint, load_lllite, load_controlnet
from functions.sdxl.generate import *
from functions.sdxl.utils import *
from tqdm import tqdm
import numpy as np
import torch
import random
import gc
import comfy.model_management

load_extra_path_config('ComfyUI/extra_model_paths.yaml')

checkpoint_name = 'SDXL_sd_xl_base_1.0_light.safetensors'
controlnet_name = 'SDXL_Depth_sai_xl_depth_256lora.safetensors'
lllite_name = 'ITS_lllite_Dtype4_Ptype3_lr2e-4_bs50_Fddp_Fv-param_step00030000.safetensors'

image_root = '/home/gkalstn000/type_4'
save_root = '/home/gkalstn000/controlnet_results'
os.makedirs(save_root, exist_ok=True)

with open(os.path.join(image_root, 'metafile.json'), 'r') as f:
    metafile = json.load(f)

filenames = os.listdir(os.path.join(save_root, 'gt_512'))
metas = {}
for filename in filenames:
    infos = metafile[filename]
    metas[filename] = infos
metafile = metas


# random_keys = random.sample(list(metafile.keys()), 1000)
# metafile = {key: metafile[key] for key in random_keys}
# with open(os.path.join(save_root, 'metafile_for_eval.json'), 'w') as f:
#     json.dump(metafile, f)

def generate_base_controlnet() :
    unet, vae, clip = load_checkpoint(checkpoint_name)
    controlnet = load_controlnet(controlnet_name)
    save_path = os.path.join(save_root,'basic_controlnet')
    os.makedirs(save_path, exist_ok=True)
    for filename, infos in tqdm(metafile.items()) :
        depth_map = Image.open(os.path.join(image_root, 'depth_map', filename)).convert('RGB')
        width, height = depth_map.size

        init_noise = get_init_noise(width,
                                    height,
                                    1)

        positive_prompt = infos['caption']
        positive_negative = ''

        positive_cond, negative_cond = encode_prompt(clip,
                                                     positive_prompt,
                                                     positive_negative)
        # ControlNet
        depth_map = torch.Tensor(np.array(depth_map)) / 255
        depth_map = depth_map.unsqueeze(0)

        positive_cond, negative_cond = apply_controlnet(
            positive_cond,
            negative_cond,
            controlnet,
            depth_map,
            1,
            0,
            1

        )
        seed = random.randint(0, 1e9)
        latent_image = sample_image(
            unet=unet,
            positive_cond=positive_cond,
            negative_cond=negative_cond,
            latent_image=init_noise,
            seed=seed,
            steps=20,
            cfg=5,
            sampler_name='dpmpp_2m_sde',
            scheduler='karras',
            start_at_step=0,
            end_at_step=20,
        )
        image_tensor = decode_latent(vae, latent_image).squeeze() * 255
        image_array = image_tensor.numpy().astype(np.uint8)
        image = Image.fromarray(image_array)
        image.save(os.path.join(save_path, filename))

        comfy.model_management.unload_all_models()
        comfy.model_management.cleanup_models()
        gc.collect()
        comfy.model_management.soft_empty_cache()

def generate_lllite_controlnet() :
    unet, vae, clip = load_checkpoint(checkpoint_name)

    save_path = os.path.join(save_root,'lllite_controlnet')
    os.makedirs(save_path, exist_ok=True)
    for filename, infos in tqdm(metafile.items()) :
        depth_map = Image.open(os.path.join(image_root, 'depth_map', filename)).convert('RGB')
        width, height = depth_map.size

        init_noise = get_init_noise(width,
                                    height,
                                    1)

        positive_prompt = infos['caption']
        positive_negative = ''

        positive_cond, negative_cond = encode_prompt(clip,
                                                     positive_prompt,
                                                     positive_negative)
        # ControlNet
        depth_map = torch.Tensor(np.array(depth_map)) / 255
        depth_map = depth_map.unsqueeze(0)

        unet_llite = load_lllite(unet, lllite_name, depth_map)

        seed = random.randint(0, 1e9)
        latent_image = sample_image(
            unet=unet_llite,
            positive_cond=positive_cond,
            negative_cond=negative_cond,
            latent_image=init_noise,
            seed=seed,
            steps=20,
            cfg=5,
            sampler_name='euler_ancestral',
            scheduler='simple',
            start_at_step=0,
            end_at_step=20,
        )
        image_tensor = decode_latent(vae, latent_image).squeeze() * 255
        image_array = image_tensor.numpy().astype(np.uint8)
        image = Image.fromarray(image_array)
        image.save(os.path.join(save_path, filename))

        comfy.model_management.unload_all_models()
        comfy.model_management.cleanup_models()
        gc.collect()
        comfy.model_management.soft_empty_cache()
generate_lllite_controlnet()
# generate_base_controlnet()
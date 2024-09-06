import importlib, sys, os
import torch
import numpy as np
import yaml, os
from utils import set_comfyui_packages
set_comfyui_packages()
import folder_paths
CHECKPOINT_ROOT = os.getenv('CHECKPOINT_ROOT')

def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = CHECKPOINT_ROOT
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                folder_paths.add_model_folder_path(x, full_path)

def import_module_from_path(module_name, module_path):
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        else:
            module_spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_path, "__init__.py"))

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)

    return module

def get_class_from_module(module, name):
    try:
        return getattr(module, name)
    except AttributeError:
        raise ImportError(f"Module '{module.__name__}' does not define a '{name}' class or function")

def get_function_from_comfyui(module_path, func_name = None) :
    module_name = os.path.basename(module_path)
    module = import_module_from_path(module_name, module_path)
    if func_name:
        for m in func_name.split('.') :
            module = get_class_from_module(module, m)
    return module

# TODO: clip_vision 가지고 다니면 IP-adapter 할 때 굳이 안불러와도됨.
@torch.inference_mode()
def load_checkpoint(model_name) :
    from ComfyUI.comfy.sd import load_checkpoint_guess_config
    model_path = os.path.join(CHECKPOINT_ROOT, 'checkpoints', model_name)
    if not os.path.exists(model_path):
        raise Exception(f"SD model path wrong: {model_path}")
    model_patcher, clip, vae, clipvision = load_checkpoint_guess_config(model_path)
    return model_patcher, vae, clip, clipvision

@torch.inference_mode()
def load_controlnet(model_name) :
    from ComfyUI.comfy.controlnet import load_controlnet
    model_path = os.path.join(CHECKPOINT_ROOT, 'controlnet', model_name)
    if not os.path.exists(model_path):
        raise Exception(f"ControlNet model path wrong: {model_path}")
    controlnet = load_controlnet(model_path)
    return controlnet

@torch.inference_mode()
def load_fooocus(model_name) :
    head = 'SDXL_fooocus_inpaint_head.pth'
    patch = model_name
    module_path = 'ComfyUI/custom_nodes/comfyui-inpaint-nodes'
    func_name = 'nodes.LoadFooocusInpaint'
    fooocus_loader = get_function_from_comfyui(module_path, func_name)
    fooocus = fooocus_loader().load(head, patch)[0]
    return fooocus

@torch.inference_mode()
def load_upscaler(model_name) :
    from ComfyUI.comfy_extras.nodes_upscale_model import UpscaleModelLoader
    loader = UpscaleModelLoader()
    upsclae_model = loader.load_model(model_name)[0]
    return upsclae_model

@torch.inference_mode()
def load_clip_vision(model_name) :
    from ComfyUI.comfy.clip_vision import load
    model_path = os.path.join(CHECKPOINT_ROOT, 'clip_vision', model_name)
    if not os.path.exists(model_path):
        raise Exception(f"CLIP_vision model path wrong: {model_path}")
    clip_vision = load(model_path)
    return clip_vision

@torch.inference_mode()
def load_ipadapter(model_name) :
    from ComfyUI.custom_nodes.ComfyUI_IPAdapter_plus.utils import ipadapter_model_loader
    model_path = os.path.join(CHECKPOINT_ROOT, 'ipadapter', model_name)
    if not os.path.exists(model_path):
        raise Exception(f"IP-Adapter model path wrong: {model_path}")
    ipadapter = ipadapter_model_loader(model_path)
    return ipadapter

@torch.inference_mode()
def load_lora(model_name) :
    from ComfyUI.comfy.utils import load_torch_file
    model_path = os.path.join(CHECKPOINT_ROOT, 'loras', model_name)
    if not os.path.exists(model_path):
        raise Exception(f"LoRA model path wrong: {model_path}")
    lora = load_torch_file(model_path, safe_load=True)
    return lora
@torch.inference_mode()
def encode_decode(vae, image, scale = 0.18215) :
    image_scaled = image * scale
    encoded = vae.encode(image_scaled.float().cuda()).latent_dist.sample()
    encoded_scaled = 1 / scale * encoded
    decoded = vae.decode(encoded_scaled).sample
    image = (decoded / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().squeeze().numpy() * 255
    return image.astype(np.uint8)

def resize_image_with_pad(image, resolution) :
    module_path = 'ComfyUI/custom_nodes/ComfyUI-LaMA-Preprocessor'
    func_name = 'inpaint_Lama.resize_image_with_pad'
    _resize_image_with_pad = get_function_from_comfyui(module_path, func_name)

    return _resize_image_with_pad(image, resolution, skip_hwc3=True)

def load_lamaInpainting() :
    module_path = 'ComfyUI/custom_nodes/ComfyUI-LaMA-Preprocessor'
    func_name = 'inpaint_Lama.LamaInpainting'
    LamaInpainting = get_function_from_comfyui(module_path, func_name)

    return LamaInpainting()

if __name__ == '__main__':
    # module_path = 'ComfyUI/custom_nodes/ComfyUI-LaMA-Preprocessor'
    # func_name = 'annotator.lama.LamaInpainting'
    #
    # module_path = 'ComfyUI/custom_nodes/ComfyUI-LaMA-Preprocessor'
    # func_name = 'inpaint_Lama.LamaInpainting'

    module_path = 'ComfyUI/custom_nodes/ComfyUI-LaMA-Preprocessor'
    func_name = 'inpaint_Lama.resize_image_with_pad'

    module = get_function_from_comfyui(module_path, func_name)
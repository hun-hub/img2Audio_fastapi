import torch
from .utils import tensor_to_pil, normalize_mask
from cgen_utils.loader import load_birefnet, load_bg_remover
from cgen_utils.image_process import convert_image_tensor_to_base64, convert_base64_to_image_tensor, convert_image_array_to_base64
from torchvision import transforms
from comfy import model_management
from comfy.utils import common_upscale

proc_img = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
deviceType = model_management.get_torch_device().type


@torch.inference_mode()
def generate(cached_model_dict, request_data):
    biref_model, _ = load_birefnet(request_data.nukki_model)

    init_image = convert_base64_to_image_tensor(request_data.init_image) / 255
    b, h, w, c = init_image.shape

    pil_image = tensor_to_pil(init_image)
    im_tensor = proc_img(pil_image).unsqueeze(0)

    with torch.no_grad():
        mask = biref_model(im_tensor.to(deviceType))[-1].sigmoid().cpu()
    mask = common_upscale(mask, w, h, 'bilinear', "disabled")
    mask = normalize_mask(mask)

    image_base64 = convert_image_tensor_to_base64((1 - mask) * 255)

    return image_base64

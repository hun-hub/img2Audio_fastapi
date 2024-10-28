import logging
from .base_architecture import CntGenAPI
from params import Resize_RequestData, MaskEdit_RequestData
from cgen_utils.image_process import *

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseFunction_API(CntGenAPI):
    def __init__(self, args):
        super().__init__(args)
        self.app.post('/functions/resize_image_for_sd')(self.resize_image_for_sd)
        self.app.post('/functions/mask_edit')(self.mask_edit)

    def resize_image_for_sd(self, request_data: Resize_RequestData):
        image = convert_base64_to_image(request_data.image)
        resolution = 512 if request_data.resize_type == 'sd15' else 1024
        image_resized = resize_image_for_sd(image, request_data.is_mask, resolution)
        image_base64 = convert_image_to_base64(image_resized)
        return {"image_base64": image_base64}

    def mask_edit(self, request_data: MaskEdit_RequestData):
        image = convert_base64_to_image(request_data.image)
        mask = convert_base64_to_image(request_data.mask)
        image_array = np.array(image)
        mask_array = np.array(mask)

        edit_value = 255 if request_data.edit_mode == 'add_white' else 0

        image_array = np.where(mask_array == 255, edit_value, image_array)
        image = Image.fromarray(image_array)
        image_base64 = convert_image_to_base64(image)
        return {"image_base64": image_base64}
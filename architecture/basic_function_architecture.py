import logging
from .base_architecture import CntGenAPI
from params import BaseFunctionRequestData
from cgen_utils.image_process import *

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseFunction_API(CntGenAPI):
    def __init__(self, args):
        super().__init__(args)
        self.app.post('/functions/resize_image_for_sd')(self.resize_image_for_sd)

    def resize_image_for_sd(self, request_data: BaseFunctionRequestData):
        image = convert_base64_to_image(request_data.image)
        resolution = 512 if request_data.resize_type == 'sd15' else 1024
        image_resized = resize_image_for_sd(image, request_data.is_mask, resolution)
        image_base64 = convert_image_to_base64(image_resized)
        return {"image_base64": image_base64}
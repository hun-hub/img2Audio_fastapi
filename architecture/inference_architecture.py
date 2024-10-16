import logging
from functions.flux.params import FLUX_RequestData
import functions.flux.generate
from functions.sd3.params import SD3_RequestData
import functions.sd3.generate
from functions.sdxl.params import SDXL_RequestData
import functions.sdxl.generate
from functions.sd15.params import SD15_RequestData
import functions.sd15.generate
from functions.object_remove.params import Object_Remove_RequestData
import functions.object_remove.generate
from functions.upscale.params import Upscale_RequestData
import functions.upscale.generate
from functions.iclight.params import ICLight_RequestData
import functions.iclight.generate
import functions.i2c.generate
from functions.half_inpainting.params import Half_Inpainting_RequestData
import functions.half_inpainting.generate
from functions.gemini.params import Gemini_RequestData
import functions.gemini.generate
from functions.sam.params import SAM_RequestData
import functions.sam.generate

from .basic_function_architecture import BaseFunction_API

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Inference_API(BaseFunction_API):
    def __init__(self, args):
        super().__init__(args)
        # SD3
        self.app.post('/flux/generate')(self.flux_generate)
        self.app.post('/sd3/generate')(self.sd3_generate)
        self.app.post('/sdxl/generate')(self.sdxl_generate)
        self.app.post('/sd15/generate')(self.sd15_generate)
        self.app.post('/object_remove')(self.object_remove)
        self.app.post('/upscale')(self.upscale)
        self.app.post('/iclight/generate')(self.iclight_generate)
        self.app.post('/i2c/generate')(self.i2c_generate)
        self.app.post('/sdxl/half_inpainting')(self.half_inpainting_generate)
        self.app.post('/gemini')(self.gemini_generate)
        self.app.post('/sam')(self.sam)

    def flux_generate(self, request_data: FLUX_RequestData):
        image_base64 = self.generate_blueprint(functions.flux.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    def sd3_generate(self, request_data: SD3_RequestData):
        image_base64 = self.generate_blueprint(functions.sd3.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    def sdxl_generate(self, request_data: SDXL_RequestData):
        image_base64 = self.generate_blueprint(functions.sdxl.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    def sd15_generate(self, request_data: SD15_RequestData):
        image_base64 = self.generate_blueprint(functions.sd15.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    def object_remove(self, request_data: Object_Remove_RequestData):
        image_base64 = self.generate_blueprint(functions.object_remove.generate.remove, request_data)
        return {"image_base64": image_base64}

    def upscale(self, request_data: Upscale_RequestData):
        image_base64 = self.generate_blueprint(functions.upscale.generate.upscale, request_data)
        return {"image_base64": image_base64}

    def iclight_generate(self, request_data: ICLight_RequestData):
        image_base64 = self.generate_blueprint(functions.iclight.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    def i2c_generate(self, request_data: SD15_RequestData):
        image_base64, image_base64_print, image_face_detailed_base64, image_face_detailed_base64_print = self.generate_blueprint(functions.i2c.generate.generate_image, request_data)
        return {"image_base64": image_base64,
                "image_face_detail_base64": image_face_detailed_base64,
                'image_base64_print': image_base64_print,
                'image_face_detail_base64_print': image_face_detailed_base64_print,}

    def sam(self, request_data: SAM_RequestData):
        image_base64, mask_base64, mask_inv_base64 = self.generate_blueprint(functions.sam.generate.predict, request_data)
        return {'image_base64': image_base64, 'mask_base64': mask_base64, 'mask_inv_base64': mask_inv_base64}

    def half_inpainting_generate(self, request_data: Half_Inpainting_RequestData):
        image_base64 = self.generate_blueprint(functions.half_inpainting.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    def gemini_generate(self, request_data: Gemini_RequestData):
        prompt = self.gemini(functions.gemini.generate.generate_prompt, request_data)
        return {"prompt": prompt}
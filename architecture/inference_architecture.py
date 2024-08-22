import torch
from fastapi import FastAPI, HTTPException
import time
import random
import logging
from params import RequestData
from sd3.params import SD3_RequestData
from sdxl.params import SDXL_RequestData
from sd15.params import SD15_RequestData
from object_remove.params import Object_Remove_RequestData
from sub_iclight.params import ICLight_RequestData
from gemini.params import Gemini_RequestData
import sd3.generate
import sdxl.generate
import sd15.generate
import object_remove.generate
import sub_iclight.generate
import gemini.generate

from .base_architecture import CntGenAPI

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Inference_API(CntGenAPI):
    def __init__(self, args):
        super().__init__(args)
        # SD3
        self.app.post('/sd3/generate')(self.sd3_generate)
        self.app.post('/sdxl/generate')(self.sdxl_generate)
        self.app.post('/sd15/generate')(self.sd15_generate)
        self.app.post('/object_remove')(self.object_remove)
        self.app.post('/iclight/generate')(self.iclight_generate)
        self.app.post('/gemini')(self.gemini_generate)


    def sd3_generate(self, request_data: SD3_RequestData):
        image_base64 = self.generate_blueprint(sd3.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    def sdxl_generate(self, request_data: SDXL_RequestData):
        image_base64 = self.generate_blueprint(sdxl.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    def sd15_generate(self, request_data: SD15_RequestData):
        image_base64 = self.generate_blueprint(sd15.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    def object_remove(self, request_data: Object_Remove_RequestData):
        image_base64 = self.generate_blueprint(object_remove.generate.remove, request_data)
        return {"image_base64": image_base64}

    def iclight_generate(self, request_data: ICLight_RequestData):
        image_base64 = self.generate_blueprint(sub_iclight.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    def gemini_generate(self, request_data: Gemini_RequestData):
        prompt = self.gemini(gemini.generate.generate_prompt, request_data)
        return {"prompt": prompt}
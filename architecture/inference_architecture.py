import logging
import asyncio

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
from functions.nukki.params import Nukki_RequestData
import functions.nukki.generate
from functions.bg_change.params import BGChange_RequestData
import functions.bg_change.generate


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
        # self.app.post('/sdxl/half_inpainting')(self.half_inpainting_generate)
        self.app.post('/nukki')(self.nukki_generate)
        self.app.post('/sd15/bg_change')(self.bg_change_generate)
        self.app.post('/sdxl/bg_change')(self.bg_change_sdxl_generate)
        self.app.post('/gemini')(self.gemini_generate)
        self.app.post('/sam')(self.sam)

    async def flux_generate(self, request_data: FLUX_RequestData):
        logging.info("\nflux_generate API called\n")
        image_base64 = await self.add_to_queue(functions.flux.generate.generate_image, request_data)
        return {"image_base64": image_base64}

    async def sd3_generate(self, request_data: SD3_RequestData):
        logging.info("\nsd3_generate API called\n")
        image_base64 = await self.add_to_queue( # 동기적 함수 호출
            functions.sd3.generate.generate_image,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {"image_base64": image_base64}

    async def sdxl_generate(self, request_data: SDXL_RequestData):
        logging.info("\nsdxl_generate API called\n")
        image_base64 = await self.add_to_queue(# 동기적 함수 호출
            functions.sdxl.generate.generate_image,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {"image_base64": image_base64}

    async def sd15_generate(self, request_data: SD15_RequestData):
        logging.info("\nsd15_generate API called\n")
        image_base64 = await self.add_to_queue(
            functions.sd15.generate.generate_image,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {"image_base64": image_base64}

    async def object_remove(self, request_data: Object_Remove_RequestData):
        logging.info("\nobj_remove_generate API called\n")
        image_base64 = await self.add_to_queue(
            functions.object_remove.generate.remove,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {"image_base64": image_base64}

    async def upscale(self, request_data: Upscale_RequestData):
        logging.info("\nupscale_generate API called\n")
        image_base64 = await self.add_to_queue(
            functions.upscale.generate.upscale,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {"image_base64": image_base64}

    async def iclight_generate(self, request_data: ICLight_RequestData):
        logging.info("\niclight_generate API called\n")
        image_base64 = await self.add_to_queue(
            functions.iclight.generate.generate_image,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {"image_base64": image_base64}

    async def i2c_generate(self, request_data: SD15_RequestData):
        logging.info("\ni2c_generate API called\n")
        image_base64, image_base64_print, image_face_detailed_base64, image_face_detailed_base64_print =await self.add_to_queue(
            functions.i2c.generate.generate_image,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {"image_base64": image_base64,
                "image_face_detail_base64": image_face_detailed_base64,
                'image_base64_print': image_base64_print,
                'image_face_detail_base64_print': image_face_detailed_base64_print,}

    async def sam(self, request_data: SAM_RequestData):
        logging.info("\nsam_generate API called\n")
        image_base64, mask_base64, mask_inv_base64 = await self.add_to_queue(
            functions.sam.generate.predict,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {'image_base64': image_base64, 'mask_base64': mask_base64, 'mask_inv_base64': mask_inv_base64}

    # async def half_inpainting_generate(self, request_data: Half_Inpainting_RequestData):
    #     loop = asyncio.get_event_loop()
    #
    #     image_base64 = await loop.run_in_executor(
    #         self._executor,  # ThreadPoolExecutor 사용
    #         self.generate_blueprint,  # 동기적 함수 호출
    #         functions.half_inpainting.generate.generate_image,  # 첫 번째 인자 (gen_function)
    #         request_data  # 두 번째 인자 (request_data)
    #     )
    #
    #     return {"image_base64": image_base64}

    async def nukki_generate(self, request_data:Nukki_RequestData):
        logging.info("\nnukki_generate API called\n")
        image_base64 = await self.add_to_queue(
            functions.nukki.generate.generate,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {"image_base64": image_base64}

    async def bg_change_generate(self, request_data:BGChange_RequestData):
        logging.info("\nbg_change_sd15_generate API called\n")
        image_base64, image_blend_base64 = await self.add_to_queue(
            functions.bg_change.generate.generate_image,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {"image_base64": image_base64, "image_blend_base64": image_blend_base64}

    async def bg_change_sdxl_generate(self, request_data:BGChange_RequestData):
        logging.info("\nbg_change_sdxl_generate API called\n")
        image_base64, image_blend_base64 = await self.add_to_queue(
            functions.bg_change.generate.sdxl_generate_image,  # 첫 번째 인자 (gen_function)
            request_data  # 두 번째 인자 (request_data)
        )

        return {"image_base64": image_base64, "image_blend_base64": image_blend_base64}

    async def gemini_generate(self, request_data: Gemini_RequestData):
        logging.info("\ngemini API called\n")
        prompt = await self.gemini(functions.gemini.generate.generate_prompt, request_data)
        return {"prompt": prompt}
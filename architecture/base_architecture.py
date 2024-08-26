import torch
from fastapi import FastAPI, HTTPException
from logs.log_middleware import LogRequestMiddleware
from fastapi.middleware.cors import CORSMiddleware
import time
import random
import logging
from queue import Queue
from utils.loader import load_clip_vision, load_stable_diffusion, load_controlnet, load_ipadapter, load_extra_path_config
from utils import compare_cache_and_request_difference
from utils.image_process import convert_base64_to_image
from utils.text_process import prompt_refine, image_caption
import gc
import psutil

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class CntGenAPI:
    def __init__(self, args):
        self.args = args
        self.queue = Queue()
        self.app = FastAPI()
        self.app.add_middleware(LogRequestMiddleware)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        '''
        Load model 관리
        unet_base
        vae_base
        clip_base
        clip_vision_base
        unet_refine
        vae_refine
        clip_refine
        clip_vision_refine
        controlnet
            'canny':
            'inpaint': 
            'depth'
        ipadapter
        lora
        '''
        self.model_cache = {'sd_checkpoint': {'basemodel': (None, None),
                                              'refiner': (None, None)},
                            'controlnet': {'canny': (None, None),
                                           'inpaint': (None, None)},
                            'ipadapter': {'module': (None, None)}
                            } # 최대 3개 adapter 저장.

        self.app.on_event("startup")(self.startup_event)

    def check_memory_usage(self, threshold=50):
        # 시스템의 메모리 사용량을 퍼센트로 반환
        memory = psutil.virtual_memory()
        psutil
        return memory.percent > threshold

    def startup_event(self):
        logger.info(" \n============ 초기 모델 로드 중 ============")
        self.model_cache['sd_checkpoint']['basemodel'] = (self.args.default_ckpt, load_stable_diffusion(self.args.default_ckpt))
        load_extra_path_config('ComfyUI/extra_model_paths.yaml')
        logger.info("\n============ 초기 모델 로드 완료 ============")

    def _cached_model_update(self, request_data):
        # log 출력
        model_request = {'sd_checkpoint': {'basemodel': request_data.basemodel},
                         'controlnet': {controlnet_request.type : controlnet_request.controlnet for i, controlnet_request in enumerate(request_data.controlnet_requests)},
                         'ipadapter': {'module': None}
                         }
        if 'refiner' in request_data.dict() and request_data.refiner :
            model_request['sd_checkpoint']['refiner'] = request_data.refiner
        if 'ipadapter_request' in request_data.dict() and request_data.ipadapter_request :
            model_request['ipadapter']['module'] = request_data.ipadapter_request.ipadapter

        difference_dict = compare_cache_and_request_difference(self.model_cache, model_request)
        self._update_cache_module(difference_dict)


    def _update_cache_module(self, difference_dict):
        # SD checkpoint
        sd_difference = difference_dict['sd_checkpoint']
        for sd_type, info in sd_difference.items() :
            model_name = info['requested']
            self.model_cache['sd_checkpoint'][sd_type] = (model_name, load_stable_diffusion(model_name))

        controlnet_difference = difference_dict['controlnet']
        for control_type, info in controlnet_difference.items() :
            model_name = info['requested']
            self.model_cache['controlnet'][control_type] = (model_name, load_controlnet(model_name))

        ipadapter_difference = difference_dict['ipadapter']
        for ip_type, info in ipadapter_difference.items():
            model_name = info['requested']
            self.model_cache['ipadapter'][ip_type] = (model_name, load_ipadapter(model_name))

    def generate_blueprint(self, gen_function, request_data):
        # while self.check_memory_usage():
        #     print("Memory usage is high, waiting...")
        #     time.sleep(1)  # 대기

        try:
            # cache check & load model
            self._cached_model_update(request_data)
            image_base64 = gen_function(self.model_cache, request_data)

            # self.queue.get()
            torch.cuda.empty_cache()
            gc.collect()
            return image_base64

        except Exception as e:
            # self.queue.get()
            torch.cuda.empty_cache()
            gc.collect()
            raise HTTPException(status_code=500, detail=str(e))

    def gemini(self, gen_function, request_data):
        # 큐에 요청을 넣고 대기
        # self.queue.put(request_data)
        # while self.queue.queue[0] is not request_data:
        #     time.sleep(0.1)

        try:
            prompt = gen_function(request_data)

            # self.queue.get()
            torch.cuda.empty_cache()
            gc.collect()
            return prompt

        except Exception as e:
            # self.queue.get()
            torch.cuda.empty_cache()
            gc.collect()
            raise HTTPException(status_code=500, detail=str(e))

    def get_app(self):
        return self.app
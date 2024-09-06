import torch
from fastapi import FastAPI, HTTPException
from logs.log_middleware import LogRequestMiddleware
from fastapi.middleware.cors import CORSMiddleware
import time
import random
import logging
from queue import Queue
from utils.loader import load_clip_vision, load_controlnet, load_ipadapter, load_extra_path_config
from utils import (update_model_cache_from_blueprint,
                   cache_checkpoint,
                   cache_unet,
                   cache_vae,
                   cache_clip,
                   cache_controlnet,
                   cache_ipadapter,
                   cache_lora
                   )
import gc
import psutil
import json

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
        ===== Load model 관리 =====
        unet : 
            sd15: {MODEL_NAME: MODULE}
            sdxl:
                base: {MODEL_NAME: MODULE}
                refiner: {MODEL_NAME: MODULE}
            sd3: {MODEL_NAME: MODULE}
            flux: {MODEL_NAME: MODULE}
        vae :
            sd15: {MODEL_NAME: MODULE}
            sdxl:
                base: {MODEL_NAME: MODULE}
                refiner: {MODEL_NAME: MODULE}
            sd3: {MODEL_NAME: MODULE}
            flux: {MODEL_NAME: MODULE}
        clip :
            sd15: {MODEL_NAME: MODULE}
            sdxl: {MODEL_NAME: MODULE}
                base: {MODEL_NAME: MODULE}
                refiner: {MODEL_NAME: MODULE}
            sd3: {MODEL_NAME: MODULE}
            flux: {MODEL_NAME: MODULE}
        clip_vision :
            sd15: {MODEL_NAME: MODULE}
            sdxl:
                base: {MODEL_NAME: MODULE}
                refiner: {MODEL_NAME: MODULE}
            sd3: {MODEL_NAME: MODULE}
            flux {MODEL_NAME: MODULE}
        controlnet
            'sd15' :
                'canny' : {MODEL_NAME: MODULE}
                'inpaint' : {MODEL_NAME: MODULE}
                'depth' : {MODEL_NAME: MODULE}
            'sdxl'
                'canny' : {MODEL_NAME: MODULE}
                'inpaint' : {MODEL_NAME: MODULE}
                'depth' : {MODEL_NAME: MODULE}
            'sd3'
                'canny' : {MODEL_NAME: MODULE}
                'inpaint' : {MODEL_NAME: MODULE}
                'depth' : {MODEL_NAME: MODULE}
            'flux'
                'canny' : {MODEL_NAME: MODULE}
                'inpaint' : {MODEL_NAME: MODULE}
                'depth' : {MODEL_NAME: MODULE}
        ipadapter:
            sd15: {MODEL_NAME: MODULE}
            sdxl: {MODEL_NAME: MODULE}
            sd3: {MODEL_NAME: MODULE}
            flux: {MODEL_NAME: MODULE}
        lora:
            sd15: 
                'module_1':{MODEL_NAME: MODULE}
                'module_2':{MODEL_NAME: MODULE}
                'module_3':{MODEL_NAME: MODULE}
            sdxl: 
                'module_1':{MODEL_NAME: MODULE}
                'module_2':{MODEL_NAME: MODULE}
                'module_3':{MODEL_NAME: MODULE}
            sd3:
                'module_1':{MODEL_NAME: MODULE}
                'module_2':{MODEL_NAME: MODULE}
                'module_3':{MODEL_NAME: MODULE}
            flux:
                'module_1':{MODEL_NAME: MODULE}
                'module_2':{MODEL_NAME: MODULE}
                'module_3':{MODEL_NAME: MODULE}
        '''
        with open('model_cache.json', 'r') as file:
            self.model_cache = json.load(file)

        self.app.on_event("startup")(self.startup_event)

    def check_memory_usage(self, threshold=50):
        # 시스템의 메모리 사용량을 퍼센트로 반환
        memory = psutil.virtual_memory()
        psutil
        return memory.percent > threshold

    def startup_event(self):
        logger.info(" \n============ 초기 모델 로드 중 ============")
        load_extra_path_config('ComfyUI/extra_model_paths.yaml')
        with open('model_cache.json', 'r') as file:
            model_cache_blueprint = json.load(file)
        cache_checkpoint(self.model_cache, model_cache_blueprint, self.args.default_ckpt)
        update_model_cache_from_blueprint(self.model_cache, model_cache_blueprint)
        logger.info("\n============ 초기 모델 로드 완료 ============")

        del model_cache_blueprint
        gc.collect()

    def _cached_model_update(self, request_data):
        # log 출력
        with open('model_cache.json', 'r') as file:
            model_cache_blueprint = json.load(file)
        request_dict = request_data.dict()

        if request_dict['checkpoint'] :
            cache_checkpoint(self.model_cache, model_cache_blueprint, request_dict['checkpoint'])
        if request_dict['unet'] : # FLUX의 경우 unet, vae, clip 따로
            cache_unet(self.model_cache, model_cache_blueprint, request_dict['unet'])
        if request_dict['vae'] : # FLUX의 경우 unet, vae, clip 따로
            cache_vae(self.model_cache, model_cache_blueprint, request_dict['vae'])
        if request_dict['clip'] : # FLUX의 경우 unet, vae, clip 따로
            cache_clip(self.model_cache, model_cache_blueprint, request_dict['clip'])
        if 'refiner' in request_dict and request_dict['refiner'] :
            cache_checkpoint(self.model_cache, model_cache_blueprint, request_dict['refiner'], True)
        if 'controlnet_requests' in request_dict and request_dict['controlnet_requests'] :
            cache_controlnet(self.model_cache, model_cache_blueprint, request_dict['controlnet_requests'])
        if 'ipadapter_request' in request_dict and request_dict['ipadapter_request'] :
            cache_ipadapter(self.model_cache, model_cache_blueprint, request_dict['ipadapter_request'])
        if 'lora_requests' in request_dict and request_dict['lora_requests'] :
            cache_lora(self.model_cache, model_cache_blueprint, request_dict['lora_requests'])
        start = time.time()
        update_model_cache_from_blueprint(self.model_cache, model_cache_blueprint)
        print('Model check time: ', time.time() - start)
        del model_cache_blueprint
        gc.collect()

    def generate_blueprint(self, gen_function, request_data):
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
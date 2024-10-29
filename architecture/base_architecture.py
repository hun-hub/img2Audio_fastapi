import torch
import threading
from fastapi import FastAPI, HTTPException
from logs.log_middleware import LogRequestMiddleware
from fastapi.middleware.cors import CORSMiddleware
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from cgen_utils.loader import load_extra_path_config
from cgen_utils import (update_model_cache_from_blueprint,
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
import asyncio
import json
import os, sys, signal
import comfy.model_management

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class CntGenAPI:
    def __init__(self, args):
        self.args = args

        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._queue = asyncio.Queue()

        self.app = FastAPI()
        self.app.add_middleware(LogRequestMiddleware)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        with open('model_cache.json', 'r') as file:
            self.model_cache = json.load(file)

        self.app.on_event("startup")(self.startup_event)
        self.app.on_event("startup")(self.start_queue_processing)  # 큐 처리를 위한 이벤트 추가
        self.app.post('/restart')(self.restart)

    # 작업 큐에서 작업 처리하기 위한 소비자 함수
    async def process_queue(self):
        while True:
            gen_function, request_data, future = await self._queue.get()
            try:
                result = await self.generate_blueprint(gen_function, request_data)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self._queue.task_done()
                logging.info(f"Task completed: {gen_function.__name__}")

    async def start_queue_processing(self):
        asyncio.create_task(self.process_queue())

    async def generate_blueprint(self, gen_function, request_data):
        try:
            # 모델 업데이트 (락 필요)
            async with self._lock:
                self._cached_model_update(request_data)

            # 모델 생성 (락 없이 실행)
            generate_output = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                gen_function,
                self.model_cache,
                request_data
            )

            # 메모리 관리 (락 필요)
            async with self._lock:
                comfy.model_management.unload_all_models()
                comfy.model_management.cleanup_models()
                gc.collect()
                comfy.model_management.soft_empty_cache()
        except Exception as e:
            # 에러 발생 시 메모리 관리 (락 필요)
            async with self._lock:
                comfy.model_management.unload_all_models()
                comfy.model_management.cleanup_models()
                gc.collect()
                comfy.model_management.soft_empty_cache()
            raise HTTPException(status_code=500, detail=str(e))
        return generate_output

    async def add_to_queue(self, gen_function, request_data):
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((gen_function, request_data, future))
        return await future

    async def gemini(self, gen_function, request_data):
        try:
            loop = asyncio.get_event_loop()
            gemini_output = await loop.run_in_executor(
                None,  # 기본 ThreadPoolExecutor 사용
                gen_function,
                request_data
            )

            torch.cuda.empty_cache()
            gc.collect()
            return gemini_output

        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            raise HTTPException(status_code=500, detail=str(e))


    def check_memory_usage(self, threshold=50):
        # 시스템의 메모리 사용량을 퍼센트로 반환
        memory = psutil.virtual_memory()
        return memory.percent > threshold

    def startup_event(self):
        logger.info(" \n============ 초기 모델 로드 중 ============")
        load_extra_path_config('ComfyUI/extra_model_paths.yaml')
        # with open('model_cache.json', 'r') as file:
        #     model_cache_blueprint = json.load(file)
        # cache_checkpoint(self.model_cache, model_cache_blueprint, self.args.default_ckpt)
        # update_model_cache_from_blueprint(self.model_cache, model_cache_blueprint)
        logger.info("\n============ 초기 모델 로드 완료 ============")
        #
        # del model_cache_blueprint
        gc.collect()

    def restart(self):
        try:
            sys.stdout.close_log()
        except Exception as e:
            pass
        print(f"\nRestarting... [Legacy Mode]\n\n")
        os.kill(os.getpid(), signal.SIGTERM)
        os.execv(sys.executable, [sys.executable] + sys.argv)

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

    def get_app(self):
        return self.app
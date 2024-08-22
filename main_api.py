import argparse
from architecture.inference_architecture import Inference_API


parser = argparse.ArgumentParser(description="Run FastAPI server with specified model")
parser.add_argument("--port", type=int, default=7861, help="Port to serve the server on")
parser.add_argument("--default_ckpt", type=str, default='SDXL_copaxTimelessxlSDXL1_v12.safetensors', help="Default checkpoint name")
args = parser.parse_args()

app_instance = Inference_API(args)
app = app_instance.get_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_api:app",
                host="0.0.0.0",
                port=args.port,
                reload=True,
                reload_excludes=['test/*',
                                 'ComfyUI/*',
                                 'architecture/gradio_architecture.py',
                                 'architecture/ui/*']
                )

from cgen_utils import set_comfyui_packages
from dotenv import load_dotenv
load_dotenv()
set_comfyui_packages()

from architecture.gradio_architecture import GradioApp
import argparse



parser = argparse.ArgumentParser(description="Run Gradio")
parser.add_argument("--inference_addr", type=str, default='localhost:7861', help="Inference API IP_ADDR:PORT (default: localhost)")
parser.add_argument("--port", type=int, default=7860, help="Inference API port num (default: 7860)")

args = parser.parse_args()

if __name__ == "__main__":
    app = GradioApp(args)
    app.launch()



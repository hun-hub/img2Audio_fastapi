from dotenv import load_dotenv
load_dotenv()
from architecture.gradio_architecture import GradioApp
import argparse


parser = argparse.ArgumentParser(description="Run Gradio")
parser.add_argument("--ip_addr", type=str, default='localhost')
parser.add_argument("--port", type=int, default='7860')
parser.add_argument("--checkpoint_root", type=str, default='/checkpoint')

args = parser.parse_args()

if __name__ == "__main__":
    app = GradioApp(args)
    app.launch()



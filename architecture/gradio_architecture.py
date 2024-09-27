import gradio as gr
import requests

from functions.flux.utils import sned_flux_request_to_api
from .ui.flux_ui import build_flux_ui
from functions.sd3.utils import sned_sd3_request_to_api
from .ui.sd3_ui import build_sd3_ui
from functions.sdxl.utils import sned_sdxl_request_to_api
from .ui.sdxl_ui import build_sdxl_ui
from functions.sd15.utils import sned_sd15_request_to_api
from .ui.sd15_ui import build_sd15_ui
from functions.object_remove.utils import sned_object_remove_request_to_api
from .ui.obj_remove_ui import build_object_remove_ui
from functions.upscale.utils import sned_upscale_request_to_api
from .ui.upscale_ui import build_upscale_ui
from functions.iclight.utils import sned_iclight_request_to_api
from .ui.iclight_ui import build_iclight_ui
from functions.i2c.utils import sned_i2c_request_to_api
from .ui.i2c_ui import build_i2c_ui


from .ui.gemini_ui import build_gemini_ui
from functions.gemini.utils import send_gemini_request_to_api


class GradioApp:
    def __init__(self, args):
        self.args = args
        self.inference_addr = f'{args.inference_addr}'
        self.block = gr.Blocks()
        self.build_ui()  # build_ui 메서드를 호출하여 UI를 빌드합니다.

    def build_ui(self):
        with self.block as demo:
            with gr.Row():
                gr.Markdown("# Connectbrick API demo")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            image = gr.Image(sources='upload', type="numpy", label="Image")
                        with gr.Column():
                            mask = gr.Image(sources='upload', type="numpy", label="Mask")
                    with gr.Row() :
                        prompt = gr.Textbox(label="User Input Prompt")
                    with gr.Row():
                        gemini_prompt = gr.Textbox(label="Gemini Recommand Prompt (Edit Here!)")
                    with gr.Row():
                        gemini_refinement = gr.Button("Gemini Refine Prompt")

                    with gr.Tab("FLUX"):
                        flux_inputs, flux_generate = build_flux_ui(image, mask, gemini_prompt, self.inference_addr)
                    with gr.Tab("Stable Diffusion 3"):
                        sd3_inputs, sd3_generate = build_sd3_ui(image, mask, gemini_prompt, self.inference_addr)
                    with gr.Tab("Stable Diffusion XL"):
                        sdxl_inputs, sdxl_generate = build_sdxl_ui(image, mask, gemini_prompt, self.inference_addr)
                    with gr.Tab("Stable Diffusion 1.5"):
                        sd15_inputs, sd15_generate = build_sd15_ui(image, mask, gemini_prompt, self.inference_addr)
                    with gr.Tab("Object Removal"):
                        obj_remove_inputs, obj_remove_generate = build_object_remove_ui(gemini_prompt, self.inference_addr)
                    with gr.Tab("UP-scale"):
                        upscale_inputs, upscale_generate = build_upscale_ui(self.inference_addr)
                    with gr.Tab("IC-Light"):
                        iclight_inputs, iclight_generate = build_iclight_ui(image, mask, gemini_prompt, self.inference_addr)
                    with gr.Tab("Image-to-Cartoon"):
                        i2c_inputs, i2c_generate = build_i2c_ui(self.inference_addr)
                    with gr.Tab("Segment Anything"):
                        gr.Markdown("SAM Functions")
                    with gr.Tab("Gemini"):
                        gemini_inputs, (gemini_result, gemini_button) = build_gemini_ui()

                with gr.Column() :
                    generated_image = gr.Image(sources='upload', type="numpy", label="Generated Image", interactive=False)
                    # api_restart_button = gr.Button("API Restart")
            gemini_inputs_for_imagen = [gr.Text('prompt_refine', visible=False),
                                        gr.Image(sources='upload', type="numpy", visible=False),
                                        prompt,
                                        gr.Text('', visible=False),
                                        gr.Text('', visible=False),
                                        ]
            gemini_refinement.click(fn=send_gemini_request_to_api, inputs=gemini_inputs_for_imagen, outputs=gemini_prompt)
            # prompt.change(fn=send_gemini_request_to_api, inputs=gemini_inputs_for_imagen, outputs=gemini_prompt)

            # Gemini Tab
            gemini_button.click(fn=send_gemini_request_to_api, inputs=gemini_inputs, outputs=gemini_result)
            # Flux Tab
            flux_generate.click(fn=sned_flux_request_to_api, inputs = flux_inputs, outputs=generated_image)
            # SD3 Tab
            sd3_generate.click(fn=sned_sd3_request_to_api, inputs=sd3_inputs, outputs=generated_image)
            # SDXL Tab
            sdxl_generate.click(fn=sned_sdxl_request_to_api, inputs=sdxl_inputs, outputs=generated_image)
            # SD15 Tab
            sd15_generate.click(fn=sned_sd15_request_to_api, inputs=sd15_inputs, outputs=generated_image)
            # Object Removal Tab
            obj_remove_generate.click(fn=sned_object_remove_request_to_api, inputs=obj_remove_inputs, outputs=generated_image)
            # Upscale
            upscale_generate.click(fn= sned_upscale_request_to_api, inputs=upscale_inputs, outputs=generated_image)
            # IC-Light
            iclight_generate.click(fn=sned_iclight_request_to_api, inputs=iclight_inputs, outputs=generated_image)
            # I2C
            i2c_generate.click(fn=sned_i2c_request_to_api, inputs=i2c_inputs, outputs=generated_image)
            # API Restart Button
            # api_restart_button.click(fn=self.restart)
    def launch(self):
        self.block.launch(server_name='0.0.0.0', server_port=self.args.port)

    def restart(self):
        url = f"http://{self.inference_addr}/restart"
        requests.post(url)

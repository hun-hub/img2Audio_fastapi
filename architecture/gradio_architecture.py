import gradio as gr

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

from .ui.gemini_ui import build_gemini_ui
from functions.gemini.utils import send_gemini_request_to_api
from functions.gemini.params import query_dict


class GradioApp:
    def __init__(self, args):
        self.args = args
        self.ip_addr = args.ip_addr
        self.port = args.port
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

                    # with gr.Tab("FLUX"):
                    #     flux_inputs, flux_generate = build_flux_ui(image, mask, gemini_prompt, self.ip_addr)
                    with gr.Tab("Stable Diffusion 3"):
                        sd3_inputs, sd3_generate = build_sd3_ui(image, mask, gemini_prompt, self.ip_addr)
                    with gr.Tab("Stable Diffusion XL"):
                        sdxl_inputs, sdxl_generate = build_sdxl_ui(image, mask, gemini_prompt, self.ip_addr)
                    with gr.Tab("Stable Diffusion 1.5"):
                        sd15_inputs, sd15_generate = build_sd15_ui(image, mask, gemini_prompt, self.ip_addr)
                    with gr.Tab("Object Removal"):
                        obj_remove_inputs, obj_remove_generate = build_object_remove_ui(gemini_prompt, self.ip_addr)
                    with gr.Tab("UP-scale"):
                        upscale_inputs, upscale_generate = build_upscale_ui(self.ip_addr)
                    with gr.Tab("IC-Light"):
                        iclight_inputs, iclight_generate = build_iclight_ui(image, mask, gemini_prompt, self.ip_addr)
                    with gr.Tab("Segment Anything"):
                        gr.Markdown("SAM Functions")
                    with gr.Tab("Gemini"):
                        gemini_inputs, (gemini_result, gemini_button) = build_gemini_ui(self.ip_addr)

                with gr.Column() :
                    generated_image = gr.Image(sources='upload', type="numpy", label="Generated Image", interactive=False)

            gemini_inputs_for_imagen = [gr.Image(sources='upload', type="numpy", visible=False),
                                        gr.Text(query_dict['prompt_refine_query'], visible=False),
                                        prompt,
                                        gr.Text(self.ip_addr, visible=False)]
            gemini_refinement.click(fn=send_gemini_request_to_api, inputs=gemini_inputs_for_imagen, outputs=gemini_prompt)
            prompt.change(fn=send_gemini_request_to_api, inputs=gemini_inputs_for_imagen, outputs=gemini_prompt)

            # Gemini Tab
            gemini_button.click(fn=send_gemini_request_to_api, inputs=gemini_inputs, outputs=gemini_result)
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

    def launch(self):
        self.block.launch(server_name='0.0.0.0', server_port=self.port)



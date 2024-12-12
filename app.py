import gradio as gr 
import numpy as np 
import soundfile as sf
from stableaudio import get_prompt, generate_audio_from_image
from PIL import Image
import numpy as np

def tmp(image):
    image.size
    return gr.update(value="abc")

# Gradio 인터페이스
with gr.Blocks(title="lg_audio") as demo:
    gr.Markdown("""
    <h1 style="font-family: 'Comic Sans MS', cursive; font-size: 48px; text-align: center;">lg_audio</h1>
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input image here()", type="pil")
            text_input = gr.Textbox(
                label="Input text here()",
                value="""You are a musician AI whose job is to help users create their own music which its genre will reflect the character or scene from an image described by users. In particular, you need to respond succintly with few musical words, in a friendly tone, write a musical prompt for a music generation model, you MUST include chords progression. For example, if a user says, "a painting of three old women having tea party", provide immediately a musical prompt corresponding to the image description. Immediately STOP after that. It should be EXACTLY in this format: "The song is an instrumental. The song is in medium tempo with a classical guitar playing a lilting melody in accompaniment style. The song is emotional and romantic. The song is a romantic instrumental song. The chord sequence is Gm, F6, Ebm. The time signature is 4/4. This song is in Adagio. The key of this song is G minor.
                """,
                visible=False,
                interactive=False,
                
            )
            seed = gr.Slider(1, 100, value=1, step = 1, label="Seed", interactive= True)
            audio_end_in_s = gr.Slider(1, 47, value=10, step = 1, label="Video length", interactive= True)
            num_waveforms_per_prompt = gr.Slider(1, 4, value=1, step = 1, label="Waveforms_per_prompt", interactive= True)
            num_inference_steps = gr.Slider(100, 300, value=200, step = 1, label="inference_steps", interactive= True)
        with gr.Column():
            text_output = gr.Textbox(label="Output prompt here()", visible=True)
            audio_output = gr.Audio(label="Generated Audio", type="filepath", visible=True)  # 타입 변경


    text_button = gr.Button(value="Generate Music")

    text_button.click(
        fn=generate_audio_from_image,  # 오디오 생성 함수 호출
        inputs=[input_image,seed,num_inference_steps,audio_end_in_s,num_waveforms_per_prompt],                     
        outputs=[audio_output]         # 출력: 오디오 파일 경로
    )
# 인터페이스 실행
demo.launch()
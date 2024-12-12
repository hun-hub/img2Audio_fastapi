import gradio as gr 
import numpy as np 
import soundfile as sf

# 오디오 파일 생성 함수
def generate_audio_file():
    output_path = "output.wav"
    # 사인파 오디오 생성
    duration = 2  # 초
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz 사인파
    sf.write(output_path, audio, sample_rate)  # WAV 파일로 저장
    return output_path  # 파일 경로 반환

with gr.Blocks(title="lg_audio") as demo:
    gr.Markdown("""
    <h1 style="font-family: 'Comic Sans MS', cursive; font-size: 48px; text-align: center;">lg_audio</h1>
    """)

    with gr.Column():
        text_button = gr.Button("Submit")
        audio_output = gr.Audio(type="numpy")  # 오디오 출력

    text_button.click(
        generate_audio_file,  # 오디오 생성 함수 호출
        inputs=[],  # 입력값 없음 (빈 리스트로 수정)
        outputs=audio_output,  # 오디오 출력
    )

demo.launch(server_port=45698, server_name="0.0.0.0")
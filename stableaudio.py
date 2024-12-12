import torch
import soundfile as sf
from diffusers import StableAudioPipeline
from diffusers import StableAudioPipeline
from IPython.display import Audio
from PIL import Image
from functions.gemini.utils import send_gemini_request_to_api
import gradio as gr
import numpy as np 

def get_prompt(image) :
    print(type(image))

    # Define the query
    query = """You are a musician AI whose job is to help users create their own music which its genre will reflect the character or scene from an image described by users. In particular, you need to respond succintly with few musical words, in a friendly tone, write a musical prompt for a music generation model, you MUST include chords progression. For example, if a user says, "a painting of three old women having tea party", provide immediately a musical prompt corresponding to the image description. Immediately STOP after that. It should be EXACTLY in this format: "The song is an instrumental. The song is in medium tempo with a classical guitar playing a lilting melody in accompaniment style. The song is emotional and romantic. The song is a romantic instrumental song. The chord sequence is Gm, F6, Ebm. The time signature is 4/4. This song is in Adagio. The key of this song is G minor."
    """

    # Send the query to the API
    prompt = send_gemini_request_to_api(
        query_type=query,
        image=image,
        user_prompt='',
        object_description='',
        background_description='',
    )

    return prompt 


def generate_audio_from_image(image,seed =1, num_inference_steps = 200, audio_end_in_s = 47.0, num_waveforms_per_prompt = 1,  output_filename = "output.wav", negative_prompt = "Low quality."):
    """
    Combines get_prompt and generate_audio to create audio from an image.

    Args:
        image: Input image for generating the prompt and audio.
        output_filename (str): The name of the output audio file.
        negative_prompt (str): The negative prompt for audio generation.
        seed (int): Seed for reproducibility.
        num_inference_steps (int): Number of inference steps for the generation.
        audio_end_in_s (float): The duration of the generated audio in seconds.
        num_waveforms_per_prompt (int): Number of waveforms to generate.

    Returns:
        tuple: A tuple containing the sampling rate and generated audio data as a numpy array.
    """
    # Get the musical prompt from the image
    prompt = get_prompt(image)

    # Load the pipeline
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Set the seed for generator
    generator = torch.Generator("cuda").manual_seed(seed)

    # Run the audio generation
    audio = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        audio_end_in_s=audio_end_in_s,
        num_waveforms_per_prompt=num_waveforms_per_prompt,
        generator=generator,
    ).audios

    # Save the audio to a file
    output = audio[0].T.float().cpu().numpy()
    sampling_rate = pipe.vae.sampling_rate
    sf.write(output_filename, output, sampling_rate)
    
    # Return the path to the output file
    return output_filename

import vertexai
from vertexai.generative_models import GenerativeModel, Image
import numpy as np
import base64

PROJECT_ID = "mystical-nimbus-408605"
REGION = "us-central1"  # e.g. us-central1
vertexai.init(project=PROJECT_ID, location=REGION)

prompt_refiner = GenerativeModel('gemini-1.5-pro')
image_captioner = GenerativeModel('gemini-pro-vision')

def prompt_refine(query, user_prompt) :
    input_text = query.format(user_prompt = user_prompt)
    response = prompt_refiner.generate_content(input_text)
    return response.text

def image_caption(query, user_prompt, image_base64) :
    input_text = query.format(user_prompt = user_prompt)
    image = Image.from_bytes(base64.b64decode(image_base64))
    response = image_captioner.generate_content([input_text, image])
    return response.text
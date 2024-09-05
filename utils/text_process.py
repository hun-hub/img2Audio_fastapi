import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Image,
    HarmCategory,
    HarmBlockThreshold,
    SafetySetting,
)
import numpy as np
import base64

PROJECT_ID = "mystical-nimbus-408605"
REGION = "us-central1"  # e.g. us-central1
vertexai.init(project=PROJECT_ID, location=REGION)

safety_config = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
]

prompt_refiner = GenerativeModel('gemini-1.5-pro')
image_captioner = GenerativeModel('gemini-pro-vision')

def gemini_with_prompt(query) :
    response = prompt_refiner.generate_content([query], safety_settings=safety_config)
    return response.text

def gemini_with_prompt_and_image(query, image_base64) :
    image = Image.from_bytes(base64.b64decode(image_base64))
    response = image_captioner.generate_content([query, image], safety_settings=safety_config)
    return response.text
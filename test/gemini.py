import vertexai
from vertexai.generative_models import GenerativeModel, Image

PROJECT_ID = "mystical-nimbus-408605"
REGION = "us-central1"  # e.g. us-central1
vertexai.init(project=PROJECT_ID, location=REGION)

model = GenerativeModel('gemini-1.0-pro')
response = model.generate_content('The opposite of hot is')
print(response.text) #  The opposite of hot is cold.
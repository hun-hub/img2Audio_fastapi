from pydantic import BaseModel
from typing import Optional, List, Literal

class Gemini_RequestData(BaseModel) :
    user_prompt: str
    query: Optional[str] = ''
    user_image: Optional[str] = None

# TODO 지저분해서 정리해야함
query_dict = {'prompt_refine_query': """You need to refine user's input prompt. 
Users_inputs are set of simple keywords for text to image generation.
Here are some examples of refined prompts:
1. flower, water, cyan perfumes bottle, still life, reflection, scenery, cosmetics, reality, 4K, nobody, creek, product photography, forest background
2. Product photography, two bottles of perfume and one bottle on the table surrounded by flowers, with soft lighting and a warm color tone. The background is a beige wall decorated with green plants, a table topped with bottles of perfume next to flowers and greenery on a table cloth covered tablecloth,
3. Product photography, a perfume bottle, in the style of floral art, horizontal composition, dreamy pastel color palette, serene floral theme, beautiful sunlight and shadow
Harmoniously combine all provided keywords into a detailed and visually appealing final description.

Given Users_inputs are {user_prompt}. 
Refined prompt's length should be less than 60 characters.
""",
              'image_caption_query': ''}
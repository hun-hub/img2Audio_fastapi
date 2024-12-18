import torch
from cgen_utils.text_process import gemini_with_prompt, gemini_with_prompt_and_image
from . import *

query_dict = {'product_description': product_description,
              'image_description': image_description,
              'prompt_refine': prompt_refine,
              'prompt_refine_with_image': prompt_refine_with_image,
              'synthesized_image_description': synthesized_image_description,
              'decompose_background_and_product': decompose_background_and_product,
              'iclight_keep_background': iclight_keep_background,
              'iclight_gen_background': iclight_gen_background,
              'i2c_description': i2c_description,
              'lg_base': lg_base,
              'lg_aniv': lg_aniv,
              'lg_artwork': lg_artwork,
              'sentence_to_keywords': sentence_to_keywords,
              'combine_sentence_keywords': combine_sentence_keywords,
              'translate_keywords': translate_keywords,
              'lg_audio': lg_audio}

@torch.inference_mode()
def generate_prompt(request_data):
    print(request_data.query_type)
    if request_data.query_type not in query_dict: # 자유 형식의 query문
        query = request_data.query_type
    else :
        query = query_dict[request_data.query_type]

    # user_prompt = ', '.join(request_data.user_prompt.split(' '))
    user_prompt = request_data.user_prompt

    object_description = request_data.object_description
    background_description = request_data.background_description
    query = query.format(user_prompt=user_prompt,
                         object_description=object_description,
                         background_description=background_description)

    if request_data.image :
        prompt = gemini_with_prompt_and_image(
            query,
            request_data.image)
    else :
        prompt = gemini_with_prompt(query)

    return prompt


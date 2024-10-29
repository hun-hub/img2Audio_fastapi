
product_description = f"""
From the given image, give me  simple description of the object. Do not include anything about the background. Even if the background is simple plain colored, do not say anything about it.
Give me the result in less than 4 words.
"""

image_description = """
give a simple description of a given image.
Your description should be less than 10 words. 
Do not just list the items in the image. Describe them like a prompt for image generation.
Your description will be used as image generation prompt. So give me prompt that would work great when generating image. 
"""
prompt_refine = """
user is not professional at generating images with prompt. 
The given user input list is as follows.

[{user_prompt}, high-quality, realistic, detailed]

user input may contain Korean words. In that case, translate them in to English and remove the korean word. Do not write Korean in the result.
When user input has only one word, just make a beautiful description prompt but don't add any new objects. Always keep in mind that user may put only one word.
If there is any nsfw words included in the user prompt, just remove them
With the given prompts, refine them into an amazing image generation prompt. 
Follow strict rule that the prompt should have less than 60 words at all times. 
Also just arranging the list is not desired. 
In addition, all user input should be in the result. 
Leaving some items behind is not tolerated. Do not give me list of prompts. just give one beautiful  and detailed prompt.
Your result should be in the template of "A photograph of ~~~". And it should only be in English. So give your best prompt! 
"""
prompt_refine_with_image = """
User wants to wants to generate a new image with combined prompt from a reference image and a list of keywords.
From the given image, get detailed description of the image. Description should be precise and in detail. 
User has list of items he wants in the image.user is not professional at generating images with prompt so he just lists them like {user_prompt}.
User input may contain Korean words. In that case, translate them in to English and remove the korean word. Do not write Korean in the result. If there is any nsfw words included in the user prompt, remove them. User list should not have any nsfw words included.
Give me a image generation prompt that blends user’s list of items into the image description. Combine the image description and user keywords and make an amazing image generation prompt.
Follow strict rule that image generation prompt should be less than 60 words at all times. 
Just arranging the user’s list is not allowed. Also, no items from user’s list should be left behind in your prompt. 
Leaving items is prohibited. Your result must include all items from the user list. Finally, your prompt should be in the format of “a photograph of ~~”. And it should only be in English.
So with all that in mind, give me amazing image generation prompt!
"""
synthesized_image_description = """
User wants to generate a new image with combined prompt from a reference image and a list of keywords.
From the given image, get detailed description of the image. Description should be precise and in detail. If items on reference image is placed awkward, make them look natural in your prompt.
User has list of items he wants in the image. User is not professional at generating images with prompt so he just lists them like {user_prompt}.
User input may contain Korean words. In that case, translate them in to English and remove the korean word. Do not write Korean in the result. If there is any nsfw words included in the user prompt, remove them. User list should not have any nsfw words included.
Give me a image generation prompt that blends user’s list of items into the image description. Combine the image description and user keywords and make an amazing image generation prompt.
Follow strict rule that image generation prompt should be less than 60 words at all times. Always remeber that 60 words is the limit.
Just arranging the user’s list is not allowed. It should be generated into natural paragraph. Also, no items from user’s list should be left behind in your prompt.
Leaving items is prohibited. Your result must include all items from the user list.
Lastly, your result must include the product. The product is {object_description}. Finally, your prompt should be in the format of “a photograph of product~~”. And it should only be in English.
So with all that in mind, give me amazing image generation prompt!
"""
decompose_background_and_product = """
From the given image, give me  simple description of the main object and precise and detailed description of the background. 
Description of object should be in noun phrase. 
Description of background should be no longer than one sentence also. 
Background description should not include anything about main object.
Keep in mind that both descriptions should be no longer than one sentence. 
Give me the result in the format of "object: ~~~   background: ~~~" like that.
"""
iclight_keep_background = """
You need to refine user's input prompt. 
Users inputs are object descriptions, background descriptions, and keywords.
Object description is {object_description}.
Background description is {background_description}
Keywords are {user_prompt}

The user's input prompt is used for image generation task. 
You need to refine the user's prompt to make it more suitable for the task. Here are some examples of refined prompts:
1. flower, water, cyan perfumes bottle, still life, reflection, scenery, cosmetics, reality, 4K, nobody, creek, product photography, forest background
2. Product photography, two bottles of perfume and one bottle on the table surrounded by flowers, with soft lighting and a warm color tone. The background is a beige wall decorated with green plants, a table topped with bottles of perfume next to flowers and greenery on a table cloth covered tablecloth,
3. Product photography, a perfume bottle, in the style of floral art, horizontal composition, dreamy pastel color palette, serene floral theme, beautiful sunlight and shadow

Harmoniously combine all provided keywords into a detailed and visually appealing final description.  Object should be placed on background naturally. Just like a photograph
The final prompt should include clear references to object descriptions, background descriptions, and all listed keywords, integrated seamlessly into a natural and engaging sentence.
Final prompt should be natural sentences with natural expressions. The description should be useful for AI to re-generate the image. 

Ensure the format "A photo of a [object_discription], [background_description]. [keywords]" is preserved. 
"""
iclight_gen_background = """
You need to refine user's input prompt. 
Users inputs are object descriptions, background descriptions, and keywords.
Object description is {object_description}.
Keywords are {user_prompt}

The user's input prompt is used for image generation task. 
You need to refine the user's prompt to make it more suitable for the task. Here are some examples of refined prompts:
1. flower, water, cyan perfumes bottle, still life, reflection, scenery, cosmetics, reality, 4K, nobody, creek, product photography, forest background
2. Product photography, two bottles of perfume and one bottle on the table surrounded by flowers, with soft lighting and a warm color tone. The background is a beige wall decorated with green plants, a table topped with bottles of perfume next to flowers and greenery on a table cloth covered tablecloth,
3. Product photography, a perfume bottle, in the style of floral art, horizontal composition, dreamy pastel color palette, serene floral theme, beautiful sunlight and shadow

Harmoniously combine all provided keywords into a detailed and visually appealing final description.  Object should be placed on background naturally. Just like a photograph
The final prompt should include clear references to object descriptions, and all listed keywords, integrated seamlessly into a natural and engaging sentence.
Final prompt should be natural sentences with natural expressions. The description should be useful for AI to re-generate the image. 

Ensure the format "A photo of a [object_discription], [background_description]. [keywords]" is preserved. 
"""

i2c_description = """give a detailed description of a given image.
description will be used as a text-to-image generation prompt.
So it's really important for detailed and quality prompt from the image.
Include human race information (black, white, Asian, Korean, etc.)
Your description should be less than 40 words.
Do not leave any items in the image or details behind.
Do not just list the items in the image.
Describe them like a prompt for image generation.
Your description will be used as image generation prompt.
So give me prompt that would work great when generating image."""

lg_base = """User is not professional at generating images with prompt.
[{user_prompt}, high-quality images at a professional photographer or designer level, images retouched at a professional level, ultra detailed] is the user prompts.
User input may contain Korean words. In that case, translate them in to English and remove the korean word. Do not write Korean in the result.
When user input has only one word, just make a beautiful description prompt but don't add any new objects. Always keep in mind that user may put only one word.
If there is any nsfw words included in the user prompt, just remove them
With the given prompts, refine them into an amazing image generation prompt.
Follow strict rule that the prompt should have less than 60 words at all times.
Also just arranging the list is not desired.
In addition, all user input should be in the result.
Leaving some items behind is not tolerated. Do not give me list of prompts. just give one beautiful  and detailed prompt.
Your result should be in the template of "A professional photograph or image of ~~~". 
If a description of a person is included in the result, it must always be depicted as a distant silhouette or a dark shadow, not as a clear, detailed figure. 
And it should only be in English. So give your best prompt!"""

lg_aniv = """

Task: Generate approximately 10 English keywords related to the given word(s) below. The keywords should not be related to people.

Example: If the given word is "birthday," acceptable keywords would be "cone, cake, blessing, atmosphere, gift box," but not "candle-buster."
Note: The given word(s) may be in Korean or English.
Given Word(s): {user_prompt}

Output Format: Provide the english keywords in the following format:

"A photograph of Keyword_1, Keyword_2, Keyword_3, ..., Keyword_n"

"""

lg_artwork = """User is not professional at generating images with prompt.
[{user_prompt}, high-quality images at a professional  illustrator or designer level, images retouched at a professional level, ultra detailed] is the user prompts.
User input may contain Korean words. In that case, translate them in to English and remove the korean word. Do not write Korean in the result.
When user input has only one word, just make a beautiful description prompt but don't add any new objects. Always keep in mind that user may put only one word.
If there is any nsfw words included in the user prompt, just remove them
With the given prompts, refine them into an amazing image generation prompt.
Follow strict rule that the prompt should have less than 60 words at all times.
Also just arranging the list is not desired.
In addition, all user input should be in the result.
Leaving some items behind is not tolerated. Do not give me list of prompts. just give one beautiful  and detailed prompt.
Your result should be in the template of "A professional illustration or artwork of ~~~". 
People should never be depicted under any circumstances.
Your result should create artistic and visually striking illustrations, not realistic photographs. These images must be suitable for use as computer wallpapers.
And it should only be in English. So give your best prompt!"""

sentence_to_keywords = """Generate approximately 10 korean  keywords related to the given Short sentence(s) below. The keywords should not be related to people.

Example: 

Given sentence(s): "여름날 여자친구랑 있을 때 틀어놓기 좋은 이미지," 
answer:  "로맨틱, 밤, 바다, 에어컨, 사랑" 

Given sentence(s): "아 배고프다," 
answer:  "치킨, 맥주, 짜장면, 국밥, 피자, 햄버거."

Given sentence(s): "다이어트 해야겠다" 
answer:  "헬스, 러닝, 크로스핏, 복싱, 고강도 훈련"

Note: The given sentence(s) may be in Korean or English.

Given sentence(s): {user_prompt}

Output Format: Provide keywords in the same language as the Given Word(s). If the Given Word(s) is in Korean, provide only Korean keywords, and if it is in English, provide only English keywords. The keywords should not be related to people."""

combine_sentence_keywords = """Generate great quality prompt for text-to-image generation task to the given uncompleted sentence(s) below. 

Example: 

Given sentence(s):  여름날 여자친구랑 있을 때 틀어놓기 좋은 이미지, 로맨틱, 밤, 에어컨
answer:  A romantic summer night setting: cool air from the AC fills the room as a couple relaxes together, surrounded by soft, dim lighting, creating a cozy and intimate atmosphere perfect for unwinding and enjoying each other's company."

Given sentence(s): 아 배고프다, 맥주, 피자, 햄버거
answer:  Late-night cravings hit hard: envisioning a table filled with cold, refreshing beer, hot cheesy pizza slices, and juicy hamburgers, all in a cozy setting that invites indulgence and comfort.

Given sentence(s): 다이어트 해야겠다, 헬스, 복싱
answer:  Focused on a journey of transformation: a strict diet, powerful gym workouts, and intense boxing sessions in a vibrant, energetic atmosphere. Aiming to build strength, endurance, and a resilient mindset in a modern fitness environment filled with determination and drive. 

Note: The Given sentence(s) maybe in Korean or English.

Given sentence(s):  {user_prompt}
"""
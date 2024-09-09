from pydantic import BaseModel
from typing import Optional, List, Literal

class Gemini_RequestData(BaseModel) :
    user_prompt: Optional[str] = ''
    object_description: Optional[str] = ''
    background_description: Optional[str] = ''
    query_type: str
    image: Optional[str] = None

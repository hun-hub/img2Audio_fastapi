from pydantic import BaseModel
from typing import Optional, List, Literal

class Gemini_RequestData(BaseModel) :
    query: str
    image: Optional[str] = None

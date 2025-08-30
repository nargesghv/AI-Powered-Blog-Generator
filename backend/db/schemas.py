from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from typing import Literal, Dict, Any

class ExtendedTopicRequest(BaseModel):
    topic: str
    model: Literal["gpt", "llama"] = "gpt"

class ReferenceCreate(BaseModel):
    title: str
    url: str

class ImageCreate(BaseModel):
    url: str
    alt: str
    license: str

class BlogCreate(BaseModel):
    topic: str
    markdown: str
    references: List[ReferenceCreate] = []
    images: List[ImageCreate] = []

class ReferenceOut(ReferenceCreate):
    id: int

class ImageOut(ImageCreate):
    id: int

class BlogOut(BaseModel):
    id: int
    topic: str
    markdown: str
    created_at: datetime
    references: List[ReferenceOut]
    images: List[ImageOut]

    class Config:
        orm_mode = True
class TopicRequest(BaseModel):
    topic: str

class EditRequest(BaseModel):
    state: dict
    edit_request: str

class BlogResponse(BaseModel):
    final_post: str
    state: dict
class RegenerateImageRequest(BaseModel):
    state: Dict[str, Any]
    final_post: str

# db/schemas.py (snippets)
from typing import Literal
from pydantic import BaseModel

class ExtendedTopicRequest(BaseModel):
    topic: str
    model: Literal["ollama", "croq"] = "ollama"

class EditRequest(BaseModel):
    state: dict
    edit_request: str

class ExtendedEditRequest(EditRequest):
    model: Literal["ollama", "croq"] = "ollama"

class RegenerateImageRequest(BaseModel):
    state: dict
    model: Literal["ollama", "croq"] = "ollama"


# backend/langchain_runner.py
from copy import deepcopy
from typing import Dict, Any
from agents.blog_agents import blog_chain, BlogState

def run_blog_chain(topic: str) -> BlogState:
    initial_state: BlogState = {"topic": topic}
    return blog_chain.invoke(initial_state)

def edit_blog_chain(state: BlogState, edit_request: str) -> BlogState:
    if "topic" not in state:
        raise ValueError("Missing 'topic' in state. Cannot edit blog without it.")
    new_state: BlogState = deepcopy(state)
    new_state["edit_request"] = edit_request
    return blog_chain.invoke(new_state)

def regenerate_images_only(state: Dict[str, Any]) -> Dict[str, Any]:
    new_state = deepcopy(state)
    new_state["regenerate_images"] = True
    return blog_chain.invoke(new_state)

# backend/langchain_runner2.py
from copy import deepcopy
from typing import Dict, Any
from agents.blog_agents2 import blog_chain, BlogState

def run_blog_chain(topic: str) -> BlogState:
    return blog_chain.invoke({"topic": topic})

def edit_blog_chain(state: BlogState, edit_request: str) -> BlogState:
    new_state: BlogState = deepcopy(state)
    new_state["edit_request"] = edit_request
    return blog_chain.invoke(new_state)

def regenerate_images_only(state: Dict[str, Any]) -> Dict[str, Any]:
    new_state = deepcopy(state)
    new_state["regenerate_images"] = True
    return blog_chain.invoke(new_state)


import sys
import os
from typing import List, Literal
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.langchain_runner import run_blog_chain as run_ollama_chain, edit_blog_chain as edit_ollama_chain, regenerate_images_only as regen_ollama_images
from backend.langchain_runner2 import run_blog_chain as run_croq_chain, edit_blog_chain as edit_croq_chain, regenerate_images_only as regen_croq_images
from backend.db import crud, schemas, models
from backend.database import SessionLocal
from db.schemas import EditRequest, BlogResponse, RegenerateImageRequest, ExtendedTopicRequest

app = FastAPI(title="Blog Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "Backend is running!"}

@app.post("/generate", response_model=schemas.BlogResponse)
def generate_blog(req: ExtendedTopicRequest, db: Session = Depends(get_db)):
    model = (getattr(req, "model", "ollama") or "ollama").lower()
    if model not in {"ollama", "croq"}:
        raise HTTPException(status_code=400, detail="model must be 'ollama' or 'croq'")
    state = run_croq_chain(req.topic) if model == "croq" else run_ollama_chain(req.topic)
    blog = models.Blog(topic=req.topic, markdown=state.get("final_post", ""))
    db.add(blog)
    db.commit()
    db.refresh(blog)
    for ref in state.get("citations", []):
        db.add(models.Reference(blog_id=blog.id, title=ref.get("title", ""), url=ref.get("url", "")))
    for img in state.get("images", []):
        db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
    db.commit()
    return {"final_post": blog.markdown, "state": state}

class ExtendedEditRequest(EditRequest):
    model: Literal["ollama", "croq"] = "ollama"

@app.post("/edit", response_model=BlogResponse)
def edit_blog(req: ExtendedEditRequest, db: Session = Depends(get_db)):
    if "topic" not in req.state:
        raise HTTPException(status_code=400, detail="Missing 'topic' in state.")
    model = (getattr(req, "model", "ollama") or "ollama").lower()
    if model not in {"ollama", "croq"}:
        raise HTTPException(status_code=400, detail="model must be 'ollama' or 'croq'")
    updated_state = edit_croq_chain(req.state, req.edit_request) if model == "croq" else edit_ollama_chain(req.state, req.edit_request)
    blog = models.Blog(topic=req.state["topic"], markdown=updated_state.get("final_post", ""))
    db.add(blog)
    db.commit()
    db.refresh(blog)
    for ref in updated_state.get("citations", []):
        db.add(models.Reference(blog_id=blog.id, title=ref.get("title", ""), url=ref.get("url", "")))
    for img in updated_state.get("images", []):
        db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
    db.commit()
    return {"final_post": updated_state.get("final_post", ""), "state": updated_state}

@app.post("/regenerate-images", response_model=BlogResponse)
def regenerate_images(req: RegenerateImageRequest, db: Session = Depends(get_db)):
    model = (getattr(req, "model", "ollama") or "ollama").lower()
    if model not in {"ollama", "croq"}:
        raise HTTPException(status_code=400, detail="model must be 'ollama' or 'croq'")
    updated_state = regen_croq_images(req.state) if model == "croq" else regen_ollama_images(req.state)
    blog = models.Blog(topic=updated_state.get("topic", ""), markdown=updated_state.get("final_post", ""))
    db.add(blog)
    db.commit()
    db.refresh(blog)
    for img in updated_state.get("images", []):
        db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
    db.commit()
    return {"final_post": updated_state.get("final_post", ""), "state": updated_state}

@app.get("/blogs", response_model=List[schemas.BlogOut])
def list_blogs(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return crud.list_blogs(db, skip=skip, limit=limit)

@app.get("/blogs/{blog_id}", response_model=List[schemas.BlogOut])
def get_blog(blog_id: int, db: Session = Depends(get_db)):
    blog = crud.get_blog(db, blog_id)
    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    return blog


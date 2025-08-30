# backend/main.py
import sys
import os
from typing import List, Literal
from dotenv import load_dotenv

load_dotenv()
MCP_CONFIG_FILE = os.getenv("MCP_CONFIG_FILE", "multiserver_setup_config.json")
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both runners and alias them to providers
from backend.langchain_runner import (
    run_blog_chain as run_ollama_chain,
    edit_blog_chain as edit_ollama_chain,
    regenerate_images_only as regen_ollama_images,
)
from backend.langchain_runner2 import (
    run_blog_chain as run_croq_chain,
    edit_blog_chain as edit_croq_chain,
    regenerate_images_only as regen_croq_images,
)

from backend.db import crud, schemas, models
from backend.database import SessionLocal
from db.schemas import EditRequest, BlogResponse, RegenerateImageRequest, ExtendedTopicRequest

app = FastAPI(title="Blog Backend", version="1.0.0")

# CORS (wide-open for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "Backend is running!"}

# -----------------------------
# Generate (Ollama or Croq)
# -----------------------------
@app.post("/generate", response_model=schemas.BlogResponse)
def generate_blog(req: ExtendedTopicRequest, db: Session = Depends(get_db)):
    model = (getattr(req, "model", "ollama") or "ollama").lower()
    if model not in {"ollama", "croq"}:
        raise HTTPException(status_code=400, detail="model must be 'ollama' or 'croq'")

    state = run_croq_chain(req.topic) if model == "croq" else run_ollama_chain(req.topic)

    blog = models.Blog(topic=req.topic, markdown=state.get("final_post", ""))
    db.add(blog); db.commit(); db.refresh(blog)

    for ref in state.get("citations", []):
        db.add(models.Reference(blog_id=blog.id, title=ref.get("title", ""), url=ref.get("url", "")))
    for img in state.get("images", []):
        db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
    db.commit()

    return {"final_post": blog.markdown, "state": state}

# Extend EditRequest to include model selection
class ExtendedEditRequest(EditRequest):
    model: Literal["ollama", "croq"] = "ollama"

@app.post("/edit", response_model=BlogResponse)
def edit_blog(req: ExtendedEditRequest, db: Session = Depends(get_db)):
    if "topic" not in req.state:
        raise HTTPException(status_code=400, detail="Missing 'topic' in state.")

    model = (getattr(req, "model", "ollama") or "ollama").lower()
    if model not in {"ollama", "croq"}:
        raise HTTPException(status_code=400, detail="model must be 'ollama' or 'croq'")

    updated_state = edit_croq_chain(req.state, req.edit_request) if model == "croq" \
                   else edit_ollama_chain(req.state, req.edit_request)

    blog = models.Blog(topic=req.state["topic"], markdown=updated_state.get("final_post", ""))
    db.add(blog); db.commit(); db.refresh(blog)

    for ref in updated_state.get("citations", []):
        db.add(models.Reference(blog_id=blog.id, title=ref.get("title", ""), url=ref.get("url", "")))
    for img in updated_state.get("images", []):
        db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
    db.commit()

    return {"final_post": updated_state.get("final_post", ""), "state": updated_state}

# -----------------------------
# Regenerate Images (Ollama or Croq)
# -----------------------------
@app.post("/regenerate-images", response_model=BlogResponse)
def regenerate_images(req: RegenerateImageRequest, db: Session = Depends(get_db)):
    model = (getattr(req, "model", "ollama") or "ollama").lower()
    if model not in {"ollama", "croq"}:
        raise HTTPException(status_code=400, detail="model must be 'ollama' or 'croq'")

    updated_state = regen_croq_images(req.state) if model == "croq" else regen_ollama_images(req.state)

    blog = models.Blog(topic=updated_state.get("topic", ""), markdown=updated_state.get("final_post", ""))
    db.add(blog); db.commit(); db.refresh(blog)

    for img in updated_state.get("images", []):
        db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
    db.commit()

    return {"final_post": updated_state.get("final_post", ""), "state": updated_state}

# -----------------------------
# Blog listing endpoints
# -----------------------------
@app.get("/blogs", response_model=List[schemas.BlogOut])
def list_blogs(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return crud.list_blogs(db, skip=skip, limit=limit)

@app.get("/blogs/{blog_id}", response_model=schemas.BlogOut)
def get_blog(blog_id: int, db: Session = Depends(get_db)):
    blog = crud.get_blog(db, blog_id)
    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    return blog

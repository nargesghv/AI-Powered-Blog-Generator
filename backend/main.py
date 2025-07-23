# backend/main.py

import sys, os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from typing import Literal



# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.langchain_runner import run_blog_chain, edit_blog_chain, regenerate_images_only
from backend.db import crud, schemas, models
from backend.database import SessionLocal
from db.schemas import TopicRequest, EditRequest, BlogResponse, RegenerateImageRequest, ExtendedTopicRequest
from typing import Literal, Dict, Any, List


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
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

@app.post("/generate", response_model=schemas.BlogResponse)
def generate_blog(req: TopicRequest, db: Session = Depends(get_db)):
    state = run_blog_chain(req.topic)

    blog = models.Blog(topic=req.topic, markdown=state.get("final_post", ""))
    db.add(blog)
    db.commit()
    db.refresh(blog)

    for ref in state.get("references", []):
        db.add(models.Reference(blog_id=blog.id, title=ref.get("title", ""), url=ref.get("url", "")))
    for img in state.get("images", []):
        db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
    db.commit()

    return {"final_post": blog.markdown, "state": state}

@app.post("/edit", response_model=BlogResponse)
def edit_blog(req: EditRequest, db: Session = Depends(get_db)):
    if "topic" not in req.state:
        raise HTTPException(status_code=400, detail="Missing 'topic' in state.")
    
    updated_state = edit_blog_chain(req.state, req.edit_request)

    blog = models.Blog(topic=req.state["topic"], markdown=updated_state.get("final_post", ""))
    db.add(blog)
    db.commit()
    db.refresh(blog)

    for ref in updated_state.get("references", []):
        db.add(models.Reference(blog_id=blog.id, title=ref.get("title", ""), url=ref.get("url", "")))
    for img in updated_state.get("images", []):
        db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
    db.commit()

    return {
        "final_post": updated_state.get("final_post", ""),
        "state": updated_state
    }

# ✅ NEW: Regenerate only images and return new blog post
@app.post("/regenerate-images", response_model=BlogResponse)
def regenerate_images(req: BlogResponse, db: Session = Depends(get_db)):
    updated_state = regenerate_images_only(req.state)

    blog = models.Blog(topic=updated_state.get("topic", ""), markdown=updated_state.get("final_post", ""))
    db.add(blog)
    db.commit()
    db.refresh(blog)

    for img in updated_state.get("images", []):
        db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
    db.commit()

    return {
        "final_post": updated_state.get("final_post", ""),
        "state": updated_state
    }

# Blog listing endpoints
@app.get("/blogs", response_model=List[schemas.BlogOut])
def list_blogs(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return crud.list_blogs(db, skip=skip, limit=limit)

@app.get("/blogs/{blog_id}", response_model=schemas.BlogOut)
def get_blog(blog_id: int, db: Session = Depends(get_db)):
    blog = crud.get_blog(db, blog_id)
    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    return blog


@app.post("/generate", response_model=schemas.BlogResponse)
def generate_blog(req: ExtendedTopicRequest, db: Session = Depends(get_db)):
    # Use appropriate chain based on selected model
    if req.model == "llama":
        state = run_llama_chain(req.topic)
    else:
        state = run_gpt_chain(req.topic)

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


# ✅ Extended EditRequest to include model selection
class ExtendedEditRequest(EditRequest):
    model: Literal["gpt", "llama"] = "gpt"


@app.post("/edit", response_model=BlogResponse)
def edit_blog(req: ExtendedEditRequest, db: Session = Depends(get_db)):
    if "topic" not in req.state:
        raise HTTPException(status_code=400, detail="Missing 'topic' in state.")

    # Use appropriate edit chain based on model
    if req.model == "llama":
        updated_state = edit_llama_chain(req.state, req.edit_request)
    else:
        updated_state = edit_gpt_chain(req.state, req.edit_request)

    blog = models.Blog(topic=req.state["topic"], markdown=updated_state.get("final_post", ""))
    db.add(blog)
    db.commit()
    db.refresh(blog)

    for ref in updated_state.get("citations", []):
        db.add(models.Reference(blog_id=blog.id, title=ref.get("title", ""), url=ref.get("url", "")))

    for img in updated_state.get("images", []):
        db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))

    db.commit()

    return {
        "final_post": updated_state.get("final_post", ""),
        "state": updated_state
    }


# --- CRUD Blog API ---

@app.get("/blogs", response_model=List[schemas.BlogOut])
def list_blogs(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return crud.list_blogs(db, skip=skip, limit=limit)


@app.get("/blogs/{blog_id}", response_model=schemas.BlogOut)
def get_blog(blog_id: int, db: Session = Depends(get_db)):
    blog = crud.get_blog(db, blog_id)
    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    return blog

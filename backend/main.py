"""
Improved Main Application with Enhanced Blog Agents
"""
import sys
import os
from typing import List, Literal, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Load env vars early
load_dotenv()

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import enhanced runners
from backend.langchain_runner import (
    run_blog_chain as run_ollama_chain,
    edit_blog_chain as edit_ollama_chain,
    regenerate_images_only as regen_ollama_images,
    run_blog_with_custom_settings as run_ollama_custom
)
from backend.langchain_runner2 import (
    run_blog_chain as run_groq_chain,
    edit_blog_chain as edit_groq_chain,
    regenerate_images_only as regen_groq_images,
    run_blog_with_custom_settings as run_groq_custom
)

from backend.db import crud, schemas, models
from backend.database import SessionLocal
from backend.db.schemas import EditRequest, BlogResponse, RegenerateImageRequest, ExtendedTopicRequest

app = FastAPI(title="Enhanced Blog Backend", version="2.0.0")

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
    return {
        "message": "Enhanced Blog Backend is running!",
        "version": "2.0.0",
        "features": [
            "Advanced Research Agent",
            "Enhanced Content Generation",
            "Intelligent Search",
            "Quality Analysis",
            "Better Error Handling"
        ]
    }

# -----------------------------
# Enhanced Generate Endpoint
# -----------------------------
class EnhancedTopicRequest(ExtendedTopicRequest):
    target_audience: str = "tech-savvy professionals"
    content_type: str = "informative article"
    word_count_target: int = 1000
    enable_advanced_search: bool = True
    enable_content_analysis: bool = True

@app.post("/generate", response_model=schemas.BlogResponse)
def generate_blog_enhanced(req: EnhancedTopicRequest, db: Session = Depends(get_db)):
    """Enhanced blog generation with advanced features"""
    model = (getattr(req, "model", "ollama") or "ollama").lower()
    if model not in {"ollama", "groq"}:
        raise HTTPException(status_code=400, detail="model must be 'ollama' or 'groq'")

    try:
        if model == "groq":
            state = run_groq_custom(
                topic=req.topic,
                target_audience=req.target_audience,
                content_type=req.content_type,
                word_count_target=req.word_count_target,
                enable_advanced_search=req.enable_advanced_search,
                enable_content_analysis=req.enable_content_analysis
            )
        else:
            state = run_ollama_custom(
                topic=req.topic,
                target_audience=req.target_audience,
                content_type=req.content_type,
                word_count_target=req.word_count_target,
                enable_advanced_search=req.enable_advanced_search,
                enable_content_analysis=req.enable_content_analysis
            )

        # Save to database
        blog = models.Blog(topic=req.topic, markdown=state.get("final_post", ""))
        db.add(blog)
        db.commit()
        db.refresh(blog)

        # Save references and images
        for ref in state.get("citations", []):
            db.add(models.Reference(blog_id=blog.id, title=ref.get("title", ""), url=ref.get("url", "")))
        for img in state.get("images", []):
            db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
        db.commit()

        return {"final_post": blog.markdown, "state": state}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blog generation failed: {str(e)}")

# -----------------------------
# Enhanced Edit Endpoint
# -----------------------------
class EnhancedEditRequest(EditRequest):
    model: Literal["ollama", "groq"] = "ollama"

@app.post("/edit", response_model=BlogResponse)
def edit_blog_enhanced(req: EnhancedEditRequest, db: Session = Depends(get_db)):
    """Enhanced blog editing with better context understanding"""
    if "topic" not in req.state:
        raise HTTPException(status_code=400, detail="Missing 'topic' in state.")

    model = (getattr(req, "model", "ollama") or "ollama").lower()
    if model not in {"ollama", "groq"}:
        raise HTTPException(status_code=400, detail="model must be 'ollama' or 'groq'")

    try:
        if model == "groq":
            updated_state = edit_groq_chain(req.state, req.edit_request)
        else:
            updated_state = edit_ollama_chain(req.state, req.edit_request)

        # Save to database
        blog = models.Blog(topic=req.state["topic"], markdown=updated_state.get("final_post", ""))
        db.add(blog)
        db.commit()
        db.refresh(blog)

        # Save references and images
        for ref in updated_state.get("citations", []):
            db.add(models.Reference(blog_id=blog.id, title=ref.get("title", ""), url=ref.get("url", "")))
        for img in updated_state.get("images", []):
            db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
        db.commit()

        return {"final_post": updated_state.get("final_post", ""), "state": updated_state}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blog editing failed: {str(e)}")

# -----------------------------
# Enhanced Regenerate Images
# -----------------------------
class EnhancedRegenerateImageRequest(RegenerateImageRequest):
    model: Literal["ollama", "groq"] = "ollama"

@app.post("/regenerate-images", response_model=BlogResponse)
def regenerate_images_enhanced(req: EnhancedRegenerateImageRequest, db: Session = Depends(get_db)):
    """Enhanced image regeneration with better relevance scoring"""
    model = (getattr(req, "model", "ollama") or "ollama").lower()
    if model not in {"ollama", "groq"}:
        raise HTTPException(status_code=400, detail="model must be 'ollama' or 'groq'")

    try:
        if model == "groq":
            updated_state = regen_groq_images(req.state)
        else:
            updated_state = regen_ollama_images(req.state)

        # Save to database
        blog = models.Blog(topic=updated_state.get("topic", ""), markdown=updated_state.get("final_post", ""))
        db.add(blog)
        db.commit()
        db.refresh(blog)

        # Save images
        for img in updated_state.get("images", []):
            db.add(models.Image(blog_id=blog.id, url=img.get("url", ""), alt=img.get("alt", ""), license=img.get("license", "")))
        db.commit()

        return {"final_post": updated_state.get("final_post", ""), "state": updated_state}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image regeneration failed: {str(e)}")

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

# -----------------------------
# New Enhanced Endpoints
# -----------------------------
@app.post("/analyze-content")
def analyze_content_quality(content: str, topic: str):
    """Analyze content quality and provide suggestions"""
    try:
        from agents.blog_agents import _analyze_content_quality
        analysis = _analyze_content_quality(content, topic)
        return {
            "quality_score": analysis["quality_score"],
            "readability_score": analysis["readability_score"],
            "structure": analysis["structure"],
            "keywords": analysis["keywords"],
            "suggestions": _generate_quality_suggestions(analysis)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content analysis failed: {str(e)}")

def _generate_quality_suggestions(analysis: Dict[str, Any]) -> List[str]:
    """Generate quality improvement suggestions"""
    suggestions = []
    
    if analysis["quality_score"] < 0.7:
        suggestions.append("Consider improving content structure and readability")
    
    if analysis["readability_score"] < 0.6:
        suggestions.append("Try shortening sentences and using simpler words")
    
    if not analysis["structure"]["has_intro"]:
        suggestions.append("Add a clear introduction section")
    
    if not analysis["structure"]["has_conclusion"]:
        suggestions.append("Add a conclusion section")
    
    if analysis["structure"]["word_count"] < 500:
        suggestions.append("Consider expanding content for better depth")
    
    return suggestions

@app.get("/search-suggestions")
def get_search_suggestions(topic: str):
    """Get search suggestions for a topic"""
    try:
        from agents.blog_agents import search_engine
        return {
            "topic": topic,
            "suggestions": search_engine._generate_related_terms(topic),
            "search_strategies": ["primary", "semantic", "news", "academic"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search suggestions failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


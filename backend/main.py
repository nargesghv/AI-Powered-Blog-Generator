import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from agents.blog_agents import enhanced_blog_chain  # no editor/ctx
from langchain_runner import (
    run_blog_chain,
    run_blog_chain_with_state,
    regenerate_images_only,
)

app = FastAPI(
    title="Enhanced Blog Generator",
    description="AI-powered blog generator with research, writing, images, and citations.",
    version="2.0.0"
)

# CORS (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopicRequest(BaseModel):
    topic: str
    model: str = "ollama"

class StateRequest(BaseModel):
    state: Dict[str, Any]
    model: str = "ollama"

@app.get("/")
async def root():
    return {
        "message": "Enhanced Blog Backend is running!",
        "version": "2.0.0",
        "features": ["Research Agent", "Content Generation", "Image Suggestions", "Citations"]
    }

@app.post("/generate")
async def generate_blog(req: TopicRequest):
    try:
        result = run_blog_chain(req.topic)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/regenerate-images")
async def regenerate_images(req: StateRequest):
    try:
        result = regenerate_images_only(req.state)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



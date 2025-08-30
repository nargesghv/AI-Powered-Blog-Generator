# backend/langchain_runner2.py  (Groq)
from copy import deepcopy
from typing import Dict, Any
from agents.blog_agents2 import blog_chain, BlogState

def run_blog_chain(topic: str) -> BlogState:
    return blog_chain.invoke({"topic": topic})

def edit_blog_chain(state: BlogState, edit_request: str) -> BlogState:
    s = deepcopy(state); s["edit_request"] = edit_request
    return blog_chain.invoke(s)

def regenerate_images_only(state: Dict[str, Any]) -> Dict[str, Any]:
    s = deepcopy(state); s["regenerate_images"] = True
    return blog_chain.invoke(s)
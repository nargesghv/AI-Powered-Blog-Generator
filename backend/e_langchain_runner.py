# backend/langchain_runner.py  (Ollama+MCP)
from copy import deepcopy
from typing import Dict, Any
from agents.blog_agents import blog_chain, BlogState

def run_blog_chain(topic: str) -> BlogState:
    return blog_chain.invoke({"topic": topic})

def run_blog_chain_with_state(initial_state: BlogState) -> BlogState:
    return blog_chain.invoke(deepcopy(initial_state))

def edit_blog_chain(state: BlogState, edit_request: str) -> BlogState:
    if "topic" not in state:
        raise ValueError("Missing 'topic' in state.")
    s = deepcopy(state); s["edit_request"] = edit_request
    return blog_chain.invoke(s)

def regenerate_images_only(state: Dict[str, Any]) -> Dict[str, Any]:
    s = deepcopy(state); s["regenerate_images"] = True
    return blog_chain.invoke(s)

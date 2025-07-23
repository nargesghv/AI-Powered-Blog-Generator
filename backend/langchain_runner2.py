# backend/langchain_runner2.py

from agents.blog_agents2 import blog_chain, BlogState

def run_blog_chain(topic: str) -> BlogState:
    return blog_chain.invoke({"topic": topic})

def edit_blog_chain(state: BlogState, edit_request: str) -> BlogState:
    state["edit_request"] = edit_request
    return blog_chain.invoke(state)

# ✅ Add this one for regenerate images button:
def regenerate_images_only(state: dict) -> dict:
    state["regenerate_images"] = True
    return blog_chain.invoke(state)

# backend/langchain_runner.py

from agents.blog_agents import blog_chain, BlogState

def run_blog_chain(topic: str) -> BlogState:
    initial_state: BlogState = {"topic": topic}
    return blog_chain.invoke(initial_state)

def edit_blog_chain(state: BlogState, edit_request: str) -> BlogState:
    if "topic" not in state:
        raise ValueError("Missing 'topic' in state. Cannot edit blog without it.")
    state["edit_request"] = edit_request
    return blog_chain.invoke(state)

# ✅ Add this one for regenerate images button:
def regenerate_images_only(state: dict) -> dict:
    state["regenerate_images"] = True
    return blog_chain.invoke(state)

"""
Improved LangChain Runner for Ollama + Enhanced Agents
"""
from copy import deepcopy
from typing import Dict, Any
from agents.blog_agents import enhanced_blog_chain, BlogState

def run_blog_chain(topic: str) -> BlogState:
    """Run enhanced blog generation chain"""
    initial_state = {
        "topic": topic,
        "target_audience": "tech-savvy professionals",
        "content_type": "informative article",
        "word_count_target": 1000,
        "enable_advanced_search": True,
        "enable_content_analysis": True
    }
    return enhanced_blog_chain.invoke(initial_state)

def run_blog_chain_with_state(initial_state: BlogState) -> BlogState:
    """Run blog chain with custom initial state"""
    return enhanced_blog_chain.invoke(deepcopy(initial_state))

def edit_blog_chain(state: BlogState, edit_request: str) -> BlogState:
    """Edit existing blog content"""
    if "topic" not in state:
        raise ValueError("Missing 'topic' in state.")
    
    s = deepcopy(state)
    s["edit_request"] = edit_request
    return enhanced_blog_chain.invoke(s)

def regenerate_images_only(state: Dict[str, Any]) -> Dict[str, Any]:
    """Regenerate images for existing blog"""
    s = deepcopy(state)
    s["regenerate_images"] = True
    return enhanced_blog_chain.invoke(s)

def run_blog_with_custom_settings(
    topic: str,
    target_audience: str = "tech-savvy professionals",
    content_type: str = "informative article",
    word_count_target: int = 1000,
    enable_advanced_search: bool = True,
    enable_content_analysis: bool = True
) -> BlogState:
    """Run blog generation with custom settings"""
    initial_state = {
        "topic": topic,
        "target_audience": target_audience,
        "content_type": content_type,
        "word_count_target": word_count_target,
        "enable_advanced_search": enable_advanced_search,
        "enable_content_analysis": enable_content_analysis
    }
    return enhanced_blog_chain.invoke(initial_state)

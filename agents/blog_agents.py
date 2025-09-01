"""
Simplified Multi-Agent Blog System
Faster version (no evaluation, no readability scoring)
"""

import os
import random
import time
from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import requests
from dotenv import load_dotenv, find_dotenv

from langgraph.graph import StateGraph, END
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

# --------------------
# Env setup
# --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(find_dotenv())

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_0")

# --------------------
# LLM setup
# --------------------
_ollama_kwargs: Dict[str, Any] = {
    "model": OLLAMA_MODEL,
    "temperature": 0.2,
    "num_ctx": 2048,
}
if OLLAMA_BASE_URL:
    _ollama_kwargs["base_url"] = OLLAMA_BASE_URL
ollama_llm = ChatOllama(**_ollama_kwargs)

if GROQ_API_KEY:
    groq_summary = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0.0,
        api_key=GROQ_API_KEY,
        max_tokens=2048,
    )
    groq_writer = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0.3,
        api_key=GROQ_API_KEY,
        max_tokens=4096,
    )
    groq_editor = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0.2,
        api_key=GROQ_API_KEY,
        max_tokens=2048,
    )
else:
    # fallback to local Ollama if no GROQ_API_KEY
    groq_summary = ollama_llm
    groq_writer = ollama_llm
    groq_editor = ollama_llm

# --------------------
# Data Types
# --------------------
@dataclass
class Article:
    title: str
    url: str
    snippet: str
    domain: str

@dataclass
class ImageData:
    url: str
    alt: str
    license: str

@dataclass
class Citation:
    title: str
    url: str
    domain: str

class BlogState(TypedDict, total=False):
    topic: str
    target_audience: str
    content_type: str
    word_count_target: int

    research_articles: List[Article]
    research_summary: str
    key_points: List[str]

    markdown_draft: str
    citations: List[Citation]
    images: List[ImageData]
    final_post: str

    edit_request: str
    edit_context: str
    regenerate_images: bool

# --------------------
# Simple Search Agent
# --------------------
def simple_search(topic: str, max_results: int = 5) -> List[Article]:
    if not SERPAPI_KEY:
        return []
    try:
        params = {"engine": "google", "q": topic, "api_key": SERPAPI_KEY, "num": max_results}
        response = requests.get("https://serpapi.com/search", params=params, timeout=20)
        response.raise_for_status()
        organic = response.json().get("organic_results", [])
        articles: List[Article] = []
        for r in organic[:max_results]:
            if r.get("title") and r.get("link"):
                articles.append(Article(
                    title=r["title"],
                    url=r["link"],
                    snippet=r.get("snippet", ""),
                    domain=r["link"].split("/")[2],
                ))
        return articles
    except Exception:
        return []

def research_agent(state: BlogState) -> BlogState:
    topic = state.get("topic", "").strip()
    if not topic:
        return state
    print(f"🔍 Researching: {topic}")
    articles = simple_search(topic)
    state["research_articles"] = [a.__dict__ for a in articles]
    return state

# --------------------
# Summarizer
# --------------------
def summarizer_agent(state: BlogState) -> BlogState:
    articles = state.get("research_articles", [])
    topic = state.get("topic", "")
    if not articles:
        state["research_summary"] = "No research available."
        state["key_points"] = []
        return state

    content = "\n\n".join([f"{a['title']} ({a['domain']}): {a['snippet']}" for a in articles])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize research into a blog outline."),
        ("user", "Topic: {topic}\n\nArticles:\n{content}\n\nPlease summarize and give key points.")
    ])

    chain = prompt | groq_summary | StrOutputParser()
    result = chain.invoke({"topic": topic, "content": content})

    state["research_summary"] = result
    state["key_points"] = [line.strip("-• ") for line in result.split("\n") if line.strip().startswith("-")]
    return state

# --------------------
# Writer
# --------------------
def writer_agent(state: BlogState) -> BlogState:
    topic = state.get("topic", "")
    summary = state.get("research_summary", "")
    key_points = "\n".join(state.get("key_points", []))

    print(f"✍️ Writing blog for: {topic}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional blog writer. Write a clear, engaging blog post in Markdown."),
        ("user", "Topic: {topic}\n\nSummary:\n{summary}\n\nKey Points:\n{key_points}\n\nWrite the blog post.")
    ])

    chain = prompt | groq_writer | StrOutputParser()
    draft = chain.invoke({"topic": topic, "summary": summary, "key_points": key_points})

    state["markdown_draft"] = draft.strip()
    state["final_post"] = draft.strip()
    return state

# --------------------
# Citations
# --------------------
def citation_agent(state: BlogState) -> BlogState:
    articles = state.get("research_articles", [])
    citations = [Citation(title=a["title"], url=a["url"], domain=a["domain"]) for a in articles]
    state["citations"] = [c.__dict__ for c in citations]
    return state

# --------------------
# Images
# --------------------
def image_agent(state: BlogState) -> List[ImageData]:
    if not PEXELS_API_KEY:
        return []
    topic = state.get("topic", "")
    print(f"🖼️ Searching images for: {topic}")
    try:
        res = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": topic, "per_page": 2},
            timeout=20,
        )
        res.raise_for_status()
        photos = res.json().get("photos", [])
        return [ImageData(url=p["src"]["medium"], alt=p.get("alt", topic), license="Pexels") for p in photos]
    except Exception:
        return []

# --------------------
# Merge
# --------------------
def merge_agent(state: BlogState) -> BlogState:
    post = state.get("markdown_draft", "")
    images = state.get("images", [])
    citations = state.get("citations", [])

    if images:
        post += "\n\n## Images\n"
        for i, img in enumerate(images, 1):
            post += f"![{img['alt']}]({img['url']})\n"

    if citations:
        post += "\n\n## References\n"
        for i, c in enumerate(citations, 1):
            post += f"{i}. [{c['title']}]({c['url']})\n"

    post += f"\n\n---\nGenerated {datetime.now().strftime('%Y-%m-%d')}"
    state["final_post"] = post
    return state

# --------------------
# LangGraph Flow
# --------------------
graph = StateGraph(BlogState)
graph.add_node("research", research_agent)
graph.add_node("summarize", summarizer_agent)
graph.add_node("write", writer_agent)
graph.add_node("cite", citation_agent)
graph.add_node("image", lambda s: {**s, "images": [i.__dict__ for i in image_agent(s)]})
graph.add_node("merge", merge_agent)

graph.set_entry_point("research")
graph.add_edge("research", "summarize")
graph.add_edge("summarize", "write")
graph.add_edge("write", "cite")
graph.add_edge("cite", "image")
graph.add_edge("image", "merge")

enhanced_blog_chain = graph.compile()

__all__ = ["enhanced_blog_chain", "BlogState"]

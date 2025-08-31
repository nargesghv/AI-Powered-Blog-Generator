"""
Enhanced Groq-focused Blog Agents with Advanced Search and Content Generation

Key Improvements:
1. Advanced Research Agent with multiple search strategies
2. Intelligent Content Filtering and Ranking
3. Enhanced Writer Agent with better prompts and structure
4. Improved Editor Agent with context-aware editing
5. Better Error Handling and Logging
6. Advanced Search Capabilities
7. Optimized for Groq API usage
"""

import os
import random
import re
import time
from typing import TypedDict, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph, END

# LangChain imports
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_groq import ChatGroq

# Utilities
from urllib.parse import urlparse

# --------------------
# Load environment variables
# --------------------
# Load .env from the nearest location up the tree (more reliable than plain load_dotenv()).
import os

# Always load from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(find_dotenv())
from dotenv import load_dotenv

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_0")


if not GROQ_API_KEY:
    raise RuntimeError("❌ Missing GROQ_API_KEY. Please set it in your .env file.")

# --------------------
# Data Types
# --------------------
@dataclass
class Article:
    title: str
    url: str
    snippet: str
    domain: str
    relevance_score: float
    quality_score: float
    publish_date: Optional[str] = None
    word_count: int = 0
    has_images: bool = False
    language: str = "en"

@dataclass
class ImageData:
    url: str
    alt: str
    license: str
    width: int = 0
    height: int = 0
    relevance_score: float = 0.0

@dataclass
class Citation:
    title: str
    url: str
    domain: str
    citation_type: str = "web"  # web, academic, news, etc.

class BlogState(TypedDict, total=False):
    topic: str
    target_audience: str
    content_type: str
    word_count_target: int
    research_articles: List[Article]
    research_summary: str
    key_points: List[str]
    expert_quotes: List[str]
    markdown_draft: str
    content_structure: Dict[str, Any]
    seo_keywords: List[str]
    citations: List[Citation]
    images: List[ImageData]
    final_post: str
    edit_request: str
    edit_context: str
    quality_score: float
    readability_score: float
    regenerate_images: bool
    enable_advanced_search: bool
    enable_content_analysis: bool

# --------------------
# Groq Models
# --------------------
groq_summary = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.0,
    api_key=GROQ_API_KEY,
    max_tokens=2048
)

groq_writer = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.3,
    api_key=GROQ_API_KEY,
    max_tokens=4096
)

groq_editor = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.2,
    api_key=GROQ_API_KEY,
    max_tokens=2048
)

groq_context = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.0,
    api_key=GROQ_API_KEY,
    max_tokens=1024
)

# --------------------
# Search Engine
# --------------------
class AdvancedSearchEngine:
    def _calculate_relevance_score(self, title: str, snippet: str, topic: str) -> float:
        topic_words = set(topic.lower().split())
        content = f"{title} {snippet}".lower()
        content_words = set(content.split())
        overlap = len(topic_words.intersection(content_words))
        base_score = overlap / len(topic_words) if topic_words else 0
        if topic.lower() in content:
            base_score += 0.3
        return min(1.0, base_score)

    def search_topic(self, topic: str, max_results: int = 10) -> List[Article]:
        if not SERPAPI_KEY:
            print("⚠️ No SERPAPI_KEY provided, skipping search.")
            return []
        try:
            params = {"engine": "google", "q": topic, "api_key": SERPAPI_KEY, "num": max_results}
            res = requests.get("https://serpapi.com/search", params=params, timeout=30)
            res.raise_for_status()
            data = res.json()
            results = []
            for r in data.get("organic_results", []):
                title = r.get("title", "")
                url = r.get("link", "")
                snippet = r.get("snippet", "")
                if not title or not url:
                    continue
                domain = urlparse(url).netloc
                results.append(Article(
                    title=title,
                    url=url,
                    snippet=snippet,
                    domain=domain,
                    relevance_score=self._calculate_relevance_score(title, snippet, topic),
                    quality_score=0.7
                ))
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

search_engine = AdvancedSearchEngine()

# --------------------
# Agents
# --------------------
def enhanced_research_agent(state: BlogState) -> BlogState:
    topic = state.get("topic", "")
    if not topic:
        state["research_articles"] = []
        return state
    print(f"🔍 Researching: {topic}")
    articles = search_engine.search_topic(topic)
    state["research_articles"] = [
        {"title": a.title, "url": a.url, "snippet": a.snippet, "domain": a.domain,
         "relevance_score": a.relevance_score, "quality_score": a.quality_score}
        for a in articles
    ]
    return state

def enhanced_summarizer_agent(state: BlogState) -> BlogState:
    articles = state.get("research_articles", [])
    if not articles:
        state["research_summary"] = "No research available."
        return state
    content = "\n".join([f"{a['title']}: {a['snippet']}" for a in articles])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the research into key insights, points, and quotes."),
        ("user", f"Topic: {state.get('topic')}\n\nArticles:\n{content}")
    ])
    chain = prompt | groq_summary | StrOutputParser()
    result = chain.invoke({})
    state["research_summary"] = result
    return state

def enhanced_writer_agent(state: BlogState) -> BlogState:
    summary = state.get("research_summary", "")
    topic = state.get("topic", "")
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Write a detailed markdown blog post for topic {topic}."),
        ("user", summary)
    ])
    chain = prompt | groq_writer | StrOutputParser()
    draft = chain.invoke({})
    state["markdown_draft"] = draft
    state["final_post"] = draft
    return state

def enhanced_citation_agent(state: BlogState) -> BlogState:
    citations = [Citation(title=a["title"], url=a["url"], domain=a.get("domain", "")) for a in state.get("research_articles", [])]
    state["citations"] = citations
    return state

def enhanced_merge_outputs(state: BlogState) -> BlogState:
    post = state.get("markdown_draft", "")
    citations = state.get("citations", [])
    ref = "\n\n## References\n" + "\n".join([f"- [{c.title}]({c.url})" for c in citations]) if citations else ""
    state["final_post"] = f"{post}{ref}"
    return state

# --------------------
# LangGraph Flow
# --------------------
graph = StateGraph(BlogState)
graph.add_node("research", enhanced_research_agent)
graph.add_node("summarize", enhanced_summarizer_agent)
graph.add_node("write", enhanced_writer_agent)
graph.add_node("cite", enhanced_citation_agent)
graph.add_node("merge", enhanced_merge_outputs)

graph.set_entry_point("research")
graph.add_edge("research", "summarize")
graph.add_edge("summarize", "write")
graph.add_edge("write", "cite")
graph.add_edge("cite", "merge")

enhanced_blog_chain = graph.compile()

# --------------------
# Export
# --------------------
__all__ = ["enhanced_blog_chain", "BlogState", "search_engine"]


"""
Cleaned version (Ollama-focused) of your first script.
- Keeps original 2-file format and flow
- Removes asyncio.run(...) (uses sync runnable chains)
- Fixes typing for citations, safer SerpAPI + Pexels calls
- Uses LangChain pipeline (PromptTemplate | ChatOllama | StrOutputParser) for strings
- Preserves Groq client for writer_agent exactly as in your code
"""
import os
import random
from typing import TypedDict, List

import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

#from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama
from groq import Groq

# --------------------
# Env
# --------------------
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")  # optional

client = Groq(api_key=GROQ_API_KEY)  # uses env if None

# --------------------
# Types
# --------------------
class Article(TypedDict):
    title: str
    url: str
    snippet: str

class ImageData(TypedDict):
    url: str
    alt: str
    license: str

class Citation(TypedDict):
    title: str
    url: str

class BlogState(TypedDict, total=False):
    topic: str
    research_articles: List[Article]
    research_summary: str
    markdown_draft: str
    citations: List[Citation]
    images: List[ImageData]
    final_post: str
    edit_request: str
    edit_context: str
    regenerate_images: bool

# --------------------
# Agents
# --------------------

def research_agent(state: BlogState) -> BlogState:
    topic = state.get("topic", "").strip()
    if not topic or not SERPAPI_KEY:
        state["research_articles"] = []
        return state
    try:
        params = {
            "engine": "google",
            "q": topic,
            "api_key": SERPAPI_KEY,
            "num": 10,
        }
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        organic = response.json().get("organic_results", [])
        articles: List[Article] = []
        for result in organic[:5]:
            title = result.get("title", "")
            url = result.get("link", "")
            snippet = result.get("snippet") or result.get("snippet_highlighted_words") or ""
            if isinstance(snippet, list):
                snippet = " ".join(snippet)
            articles.append({"title": title, "url": url, "snippet": snippet})
        state["research_articles"] = articles
    except Exception:
        state["research_articles"] = []
    return state


# Build a reusable Ollama LLM runnable
_ollama_kwargs = {"model": "deepseek-r1:7b", "temperature": 0.0}
if OLLAMA_BASE_URL:
    _ollama_kwargs["base_url"] = OLLAMA_BASE_URL
ollama_llm = ChatOllama(**_ollama_kwargs)

summ_prompt = PromptTemplate.from_template(
    """
You are a specialist in context understanding and summarizing.
Summarize the following research results into key points and themes for writing a blog on: {topic}

{content}
"""
)
summarizer_chain = summ_prompt | ollama_llm | StrOutputParser()

def summarizer_agent(state: BlogState) -> BlogState:
    articles = state.get("research_articles", [])
    if not articles:
        state["research_summary"] = "No research articles found to summarize."
        return state
    content = "\n\n".join(f"{a['title']}: {a['snippet']}" for a in articles)
    summary = summarizer_chain.invoke({"topic": state.get("topic", ""), "content": content})
    state["research_summary"] = summary or "Summary could not be generated."
    return state


def writer_agent(state: BlogState) -> BlogState:
    summary = state.get("research_summary", "No summary available.")
    prompt = f"""
You are a professional blog writer. Write a high-quality, 1000-word blog post in **Markdown format** **.md** on the topic: "{state.get('topic','')}".
Use the following research summary as your guide:
{summary}
## Guidelines:
- Do **not** include `<think>` or other meta tags in the output.
- Start your response directly with the blog content.
- Use proper Markdown syntax:
  - `# Title`
  - `## Introduction`
  - `## Main Sections`
  - `## Conclusion`
- Include:
  - Bullet points (`- item`)
  - **Bold** key phrases
  - Inline code snippets `code`
  - Hyperlinks `[text](url)`
- Professional, tech-savvy tone.
"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
    )
    state["markdown_draft"] = response.choices[0].message.content.strip()
    return state


def image_agent(state: BlogState) -> BlogState:
    topic = state.get("topic", "")
    if not topic or not PEXELS_API_KEY:
        state["images"] = []
        return state
    page = random.randint(1, 10)
    try:
        res = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": topic, "per_page": 3, "page": page},
            timeout=30,
        )
        res.raise_for_status()
        photos = res.json().get("photos", [])
        state["images"] = [
            {"url": p["src"].get("medium") or p["src"].get("large"), "alt": p.get("alt", topic), "license": "Pexels"}
            for p in photos
        ]
    except Exception:
        state["images"] = []
    return state


def citation_agent(state: BlogState) -> BlogState:
    state["citations"] = [{"title": a["title"], "url": a["url"]} for a in state.get("research_articles", [])]
    return state


def merge_outputs(state: BlogState) -> BlogState:
    post = state.get("markdown_draft", "")
    images_md = "\n".join(f"![{img['alt']}]({img['url']})" for img in state.get("images", []) if img.get("url"))
    citations_md = "\n".join(f"- [{c['title']}]({c['url']})" for c in state.get("citations", []) if c.get("url"))
    state["final_post"] = f"{post}\n\n## Images\n{images_md}\n\n## References\n{citations_md}".strip()
    return state

# ---- RAG Context Extractor ----
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_retriever_from_blog_content(markdown_text: str):
    loader = [Document(page_content=markdown_text)]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    doc_chunks = text_splitter.split_documents(loader)
    vectorstore = Chroma.from_documents(doc_chunks, collection_name=f"edit-context-{random.randint(1,1_000_000)}", embedding=embedding_model)
    return vectorstore.as_retriever()

context_prompt = PromptTemplate.from_template(
    """
You are a specialist in understanding and revising blog content.

This is the original blog content (chunked for search):
{chunked_content}

Here is the edit request:
"{edit_request}"

⚠️ Only return the original Markdown that is directly relevant. Do NOT explain. Do NOT expand.
"""
)
context_chain = context_prompt | ollama_llm | StrOutputParser()

def context_extractor_agent(state: BlogState) -> BlogState:
    if not state.get("edit_request") or not state.get("final_post"):
        return state
    retriever = get_retriever_from_blog_content(state["final_post"])
    docs = retriever.get_relevant_documents(state["edit_request"])  # type: ignore
    chunked_text = "\n\n".join(d.page_content for d in docs)
    extracted_context = context_chain.invoke({
        "edit_request": state["edit_request"],
        "chunked_content": chunked_text,
    })
    state["edit_context"] = extracted_context
    return state

editor_prompt = PromptTemplate.from_template(
    """
You are a professional blog editor.
Below is the specific section of the blog that needs revision:
---
{edit_context}
---
Edit it according to this request:
"{edit_request}"
Apply improvements in structure, clarity, formatting (Markdown), and tone.
Preserve tone and approximate word count. Use clean Markdown.
"""
)
editor_chain = editor_prompt | ollama_llm | StrOutputParser()

def editor_agent(state: BlogState) -> BlogState:
    if not state.get("edit_context") or not state.get("edit_request"):
        return state
    revised_section = editor_chain.invoke({
        "edit_context": state["edit_context"],
        "edit_request": state["edit_request"],
    })
    original = state.get("final_post", "")
    target = state.get("edit_context", "")
    if target and target in original:
        new_post = original.replace(target, revised_section, 1)
        state["final_post"] = new_post
    state["edit_request"] = ""
    state.pop("edit_context", None)
    return state

# ---- LangGraph Flow ----
graph = StateGraph(BlogState)

# Add nodes
graph.add_node("research", research_agent)
graph.add_node("summarize", summarizer_agent)
graph.add_node("write", writer_agent)
graph.add_node("cite", citation_agent)
graph.add_node("image", image_agent)
graph.add_node("merge", merge_outputs)
graph.add_node("context_extract", context_extractor_agent)
graph.add_node("edit", editor_agent)

# Define sequence
graph.set_entry_point("research")
graph.add_edge("research", "summarize")
graph.add_edge("summarize", "write")
graph.add_edge("write", "cite")
graph.add_edge("cite", "image")
graph.add_edge("image", "merge")

def route_after_image(state: BlogState) -> str:
    if state.get("regenerate_images"):
        return "image"
    elif state.get("edit_request"):
        return "context_extract"
    else:
        return END

graph.add_conditional_edges(
    "merge",
    route_after_image,
    {
        "image": "image",
        "context_extract": "context_extract",
        END: END,
    },
)

graph.add_edge("context_extract", "edit")

def route_after_edit(state: BlogState) -> str:
    return "context_extract" if state.get("edit_request") else END

graph.add_conditional_edges(
    "edit",
    route_after_edit,
    {
        "context_extract": "context_extract",
        END: END,
    },
)

blog_chain = graph.compile()


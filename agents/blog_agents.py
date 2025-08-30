"""
Ollama-first pipeline with optional MCP research + filesystem save.

- Keeps original 2-file format and LangGraph flow
- No asyncio.run(...) leaks; uses a tiny sync shim internally
- SerpAPI + Pexels remain as fallbacks
- Summarizer: Ollama (small, RAM-friendly by default)
- Writer: Groq client preserved (as in your code)
- MCP:
  - Research via 'linkup' server (if MCP config is present)
  - Save final post via 'filesystem' server when state['save_with_mcp'] = True
"""

import os
import random
from typing import TypedDict, List, Optional, Dict, Any

import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# --- Modern LangChain community imports ---
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_ollama import ChatOllama
from groq import Groq
# MCP helper (your package)
try:
    from mcp_use import MCPAgent, MCPClient  # ensure it's in your PYTHONPATH
    HAS_MCP = True
except Exception:
    MCPAgent = MCPClient = None
    HAS_MCP = False

# --------------------
# Env
# --------------------
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")  # optional
MCP_CONFIG_FILE = os.getenv("MCP_CONFIG_FILE", "multiserver_setup_config.json")

# Use small, quantized model to avoid RAM errors (fits ~3 GB)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_0")

# Groq client (preserved) - only initialize if API key is available
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

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
    # research
    research_articles: List[Article]
    research_summary: str
    # write
    markdown_draft: str
    # assets
    citations: List[Citation]
    images: List[ImageData]
    final_post: str
    # edits
    edit_request: str
    edit_context: str
    # routing flags
    regenerate_images: bool
    # MCP flags/results
    use_mcp: bool                 # if True, use MCP linkup for research (fallback to SerpAPI)
    save_with_mcp: bool           # if True, save final markdown via MCP filesystem server
    saved_filename: str
    save_result: str

# ------------------------------------------------
# Small async shim so we can keep sync LangGraph
# ------------------------------------------------
def _run_coro_sync(coro):
    """
    Run an async coroutine from sync code without leaking event loops.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Run in a new loop inside a thread
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(lambda: asyncio.run(coro))
            return fut.result()
    else:
        return asyncio.run(coro)

# --------------------
# MCP utilities
# --------------------
def _get_mcp_agent() -> Optional[MCPAgent]:
    if not HAS_MCP:
        return None
    if not os.path.exists(MCP_CONFIG_FILE):
        return None

    # Use a lightweight LLM to drive tool-use prompts
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
        num_ctx=1024,
        **({"base_url": OLLAMA_BASE_URL} if OLLAMA_BASE_URL else {}),
    )
    client_obj = MCPClient.from_config_file(MCP_CONFIG_FILE)
    return MCPAgent(
        llm=llm,
        client=client_obj,
        use_server_manager=False,
        max_steps=30
    )

def mcp_search(topic: str) -> Optional[str]:
    agent = _get_mcp_agent()
    if not agent:
        return None
    prompt = (
        f"Use the 'linkup' server to search the web for: {topic}. "
        "Return concise, useful bullet points with titles and links."
    )
    return _run_coro_sync(agent.run(prompt))

def mcp_save_markdown(filename: str, content: str) -> Optional[str]:
    agent = _get_mcp_agent()
    if not agent:
        return None
    cmd = (
        "Use the tool `write_file` from the `filesystem` server and write "
        f"filename: '{filename}' at filestore directory and save content: {content}"
    )
    return _run_coro_sync(agent.run(cmd))

# --------------------
# LLMs
# --------------------
# Ollama LLM (summarizer / editor / context)
_ollama_kwargs: Dict[str, Any] = {
    "model": OLLAMA_MODEL,
    "temperature": 0.0,
    "num_ctx": 1024,
}
if OLLAMA_BASE_URL:
    _ollama_kwargs["base_url"] = OLLAMA_BASE_URL

ollama_llm = ChatOllama(**_ollama_kwargs)

# Summarizer chain
summ_prompt = PromptTemplate.from_template(
    "You are a specialist in context understanding and summarizing.\n"
    "Summarize the following research results into key points and themes for writing a blog on: {topic}\n\n{content}\n"
)
summarizer_chain = summ_prompt | ollama_llm | StrOutputParser()

# Context extractor chain (for edits)
context_prompt = PromptTemplate.from_template(
    "You are a specialist in understanding and revising blog content.\n\n"
    "This is the original blog content (chunked for search):\n{chunked_content}\n\n"
    "Here is the edit request:\n\"{edit_request}\"\n\n"
    "From the blog, identify the exact portion(s) that need editing.\n"
    "Only return the original Markdown that is directly relevant. Do NOT explain. Do NOT expand.\n\nOutput:\n"
)
context_chain = context_prompt | ollama_llm | StrOutputParser()

# --------------------
# Agents
# --------------------
def research_agent(state: BlogState) -> BlogState:
    topic = state.get("topic", "").strip()
    use_mcp = bool(state.get("use_mcp"))

    if not topic:
        state["research_articles"] = []
        return state

    # Try MCP linkup if requested and available
    if use_mcp and HAS_MCP and os.path.exists(MCP_CONFIG_FILE):
        try:
            res = mcp_search(topic)
            if res:
                # Convert simple MCP text output into pseudo articles for the summarizer
                lines = [ln.strip() for ln in str(res).splitlines() if ln.strip()]
                articles: List[Article] = []
                for ln in lines[:8]:
                    # A naive parse: split on " - " or first space
                    title = ln
                    url = ""
                    if "http" in ln:
                        parts = ln.split("http", 1)
                        title = parts[0].strip(" -:•")
                        url = "http" + parts[1].strip()
                    articles.append({"title": title or "Result", "url": url, "snippet": ln})
                state["research_articles"] = articles
                return state
        except Exception:
            # Fall through to SerpAPI
            pass

    # Fallback: SerpAPI
    if not SERPAPI_KEY:
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

def summarizer_agent(state: BlogState) -> BlogState:
    articles = state.get("research_articles", [])
    if not articles:
        state["research_summary"] = "No research articles found to summarize."
        return state
    content = "\n\n".join(f"{a['title']}: {a['snippet']}" for a in articles)
    try:
        summary = summarizer_chain.invoke({"topic": state.get("topic", ""), "content": content})
    except Exception:
        summary = "Summarizer failed (Ollama). Try a smaller model or switch to Groq writer."
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
    if client:
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
            )
            state["markdown_draft"] = response.choices[0].message.content.strip()
        except Exception as e:
            state["markdown_draft"] = f"Error generating blog with Groq: {str(e)}"
    else:
        state["markdown_draft"] = "Groq client not available. Please set GROQ_API_KEY environment variable."
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
            {"url": p["src"]["medium"], "alt": p.get("alt", topic), "license": "Pexels"}
            for p in photos
        ][:3]
    except Exception:
        state["images"] = []
    return state

def citation_agent(state: BlogState) -> BlogState:
    state["citations"] = [
        {"title": a.get("title", ""), "url": a.get("url", "")}
        for a in state.get("research_articles", [])
    ]
    return state

def merge_outputs(state: BlogState) -> BlogState:
    post = state.get("markdown_draft", "") or ""
    images_md = "\n".join(
        f"![{img['alt']}]({img['url']})" for img in state.get("images", [])
    )
    citations_md = "\n".join(
        f"- [{c['title']}]({c['url']})" if c.get("url") else f"- {c['title']}"
        for c in state.get("citations", [])
    )
    state["final_post"] = f"{post}\n\n## Images\n{images_md}\n\n## References\n{citations_md}".strip()
    return state

# ---- RAG Context Extractor ----
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_retriever_from_blog_content(markdown_text: str):
    loader = [Document(page_content=markdown_text)]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_chunks = text_splitter.split_documents(loader)
    vectorstore = Chroma.from_documents(
        doc_chunks, collection_name="edit-context", embedding=embedding_model
    )
    return vectorstore.as_retriever()

def context_extractor_agent(state: BlogState) -> BlogState:
    if not state.get("edit_request") or not state.get("final_post"):
        return state
    retriever = get_retriever_from_blog_content(state["final_post"])
    docs = retriever.get_relevant_documents(state["edit_request"])
    chunked_text = "\n\n".join(d.page_content for d in docs)

    try:
        extracted = context_chain.invoke({
            "edit_request": state["edit_request"],
            "chunked_content": chunked_text
        })
    except Exception:
        extracted = ""
    state["edit_context"] = extracted
    return state

def editor_agent(state: BlogState) -> BlogState:
    if not state.get("edit_context") or not state.get("edit_request"):
        return state

    prompt = f"""
You are a professional blog editor.
Below is the specific section of the blog that needs revision:
---
{state['edit_context']}
---

Edit it according to this request:
"{state['edit_request']}"

Apply improvements in structure, clarity, formatting (Markdown), and tone.
Preserve tone/voice and approximate word count.
Output only the revised Markdown section.
"""
    try:
        revised = (PromptTemplate.from_template("{p}").format(p=prompt)
                   | ollama_llm | StrOutputParser()).invoke({})
        original = state["final_post"]
        state["final_post"] = original.replace(state["edit_context"], revised or state["edit_context"])
    except Exception:
        pass

    # clear edit request and context
    state["edit_request"] = ""
    state.pop("edit_context", None)
    return state

# --------------------
# MCP save (optional)
# --------------------
def mcp_save_agent(state: BlogState) -> BlogState:
    if not state.get("save_with_mcp"):
        return state
    if not state.get("final_post"):
        return state
    if not (HAS_MCP and os.path.exists(MCP_CONFIG_FILE)):
        state["save_result"] = "MCP not available or config file missing."
        return state

    topic = (state.get("topic") or "blog").strip() or "blog"
    ts = os.path.basename(os.getenv("MCP_SAVE_TS", "")) or ""
    from datetime import datetime as _dt
    stamp = _dt.now().strftime("%Y%m%d_%H%M%S") if not ts else ts
    filename = f"blog_{topic.replace(' ', '_').lower()}_{stamp}.md"

    try:
        res = mcp_save_markdown(filename, state["final_post"])
        state["saved_filename"] = filename
        state["save_result"] = str(res)
    except Exception as e:
        state["save_result"] = f"Save failed: {e}"
    return state

# --------------------
# LangGraph Flow
# --------------------
graph = StateGraph(BlogState)

graph.add_node("research", research_agent)
graph.add_node("summarize", summarizer_agent)
graph.add_node("write", writer_agent)
graph.add_node("cite", citation_agent)
graph.add_node("image", image_agent)
graph.add_node("merge", merge_outputs)
graph.add_node("context_extract", context_extractor_agent)
graph.add_node("edit", editor_agent)
graph.add_node("mcp_save", mcp_save_agent)  # optional last step

graph.set_entry_point("research")
graph.add_edge("research", "summarize")
graph.add_edge("summarize", "write")
graph.add_edge("write", "cite")
graph.add_edge("cite", "image")
graph.add_edge("image", "merge")

def route_after_merge(state: BlogState) -> str:
    if state.get("regenerate_images"):
        return "image"
    if state.get("edit_request"):
        return "context_extract"
    if state.get("save_with_mcp"):
        return "mcp_save"
    return END

graph.add_conditional_edges(
    "merge",
    route_after_merge,
    {
        "image": "image",
        "context_extract": "context_extract",
        "mcp_save": "mcp_save",
        END: END,
    }
)

graph.add_edge("context_extract", "edit")

def route_after_edit(state: BlogState) -> str:
    # If user stacks edits, loop; otherwise end or go to save if requested
    if state.get("edit_request"):
        return "context_extract"
    if state.get("save_with_mcp"):
        return "mcp_save"
    return END

graph.add_conditional_edges(
    "edit",
    route_after_edit,
    {
        "context_extract": "context_extract",
        "mcp_save": "mcp_save",
        END: END,
    }
)

blog_chain = graph.compile()
# (Optional) make exports explicit
__all__ = ["blog_chain", "BlogState"]

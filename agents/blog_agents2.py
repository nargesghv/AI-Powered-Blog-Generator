import os
import requests
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import openai
import asyncio
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain_groq import ChatGroq
from groq import Groq
import random

# Load environment variables
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY
client = Groq()
#client = openai.OpenAI()
# ---- Data Models ----
class Article(TypedDict):
    title: str
    url: str
    snippet: str

class ImageData(TypedDict):
    url: str
    alt: str
    license: str

class BlogState(TypedDict, total=False):
    topic: str
    research_articles: List[Article]
    research_summary: str
    markdown_draft: str
    citations: List[str]
    images: List[ImageData]
    final_post: str
    edit_request: str
    edit_context: str

# ---- Agents ----
def research_agent(state: BlogState) -> BlogState:
    topic = state.get("topic", "")
    response = requests.get(
        "https://serpapi.com/search",
        params={"q": topic, "api_key": SERPAPI_KEY}
    )
    articles = [
        {
            "title": result.get("title", ""),
            "url": result.get("link", ""),
            "snippet": result.get("snippet", "")
        }
        for result in response.json().get("organic_results", [])[:5]
    ]
    state["research_articles"] = articles
    return state

def summarizer_agent(state: BlogState) -> BlogState:
    articles = state.get("research_articles", [])
    if not articles:
        state["research_summary"] = "No research articles found to summarize."
        return state

    content = "\n\n".join(f"{a['title']}: {a['snippet']}" for a in articles)
    prompt = f"""you are a specialist in context understanding and summarizing. Summarize the following research results into key points and themes for writting a blog on: {state['topic']}\n\n{content}"""
    
    llm = ChatGroq(
     model="deepseek-r1-distill-llama-70b",
     temperature=0,
     api_key=GROQ_API_KEY
     )
    summary = asyncio.run(llm.ainvoke(prompt))
    state["research_summary"] = summary or "Summary could not be generated."
    return state

def writer_agent(state: BlogState) -> BlogState:
    summary = state.get("research_summary", "No summary available.")
    prompt = f"""
    You are a professional blog writer. Write a high-quality, 1000-word blog post in **Markdown format** **.md**on the topic: "{state['topic']}".
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
   - **Bullet points** where applicable (`- item`)
   - **Bold** key phrases using `**bold**`
   - Inline code snippets using backticks `` `code` ``
   - Hyperlinks using `[text](url)`
   - Write in a **professional, tech-savvy** tone.
   - Keep the style **clear, structured, and well-formatted** for a Markdown file.
   ---

   **Example Markdown Structure**:

   ```markdown
   # The Rise of AI in Daily Life

   ## Introduction
   AI is rapidly shaping our daily experiences...

   ## How AI Impacts Daily Tasks
   - Voice assistants
   - Smart recommendations
   - Automation at home and work

   ## Conclusion
   The future of AI is deeply intertwined with how we live and work.
   """

    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    state["markdown_draft"] = response.choices[0].message.content.strip()
    return state
    

def image_agent(state: BlogState) -> BlogState:
    topic = state["topic"]
    page = random.randint(1, 10)  # Try random pages
    res = requests.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization": PEXELS_API_KEY},
        params={"query": topic, "per_page": 3, "page": page}
    )

    photos = res.json().get("photos", [])
    if not photos:
        state["images"] = []
        return state

    state["images"] = [
        {
            "url": photo["src"]["medium"],
            "alt": photo["alt"],
            "license": "Pexels"
        }
        for photo in photos
    ]
    return state

def citation_agent(state: BlogState) -> BlogState:
    state["citations"] = [{"title": a["title"], "url": a["url"]} for a in state.get("research_articles", [])]
    return state

def merge_outputs(state: BlogState) -> BlogState:
    post = state.get("markdown_draft", "")
    images_md = "\n".join(
        f"![{img['alt']}]({img['url']})" for img in state.get("images", [])
    )
    citations_md = "\n".join(f"- [{c['title']}]({c['url']})" for c in state.get("citations", []))

    state["final_post"] = f"{post}\n\n## Images\n{images_md}\n\n## References\n{citations_md}"
    return state

# ---- RAG Context Extractor ----

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_retriever_from_blog_content(markdown_text: str):
    loader = [Document(page_content=markdown_text)]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    doc_chunks = text_splitter.split_documents(loader)
    vectorstore = Chroma.from_documents(doc_chunks, collection_name="edit-context", embedding=embedding_model)
    return vectorstore.as_retriever()

llm = ChatGroq(
     model="deepseek-r1-distill-llama-70b",
     temperature=0,
     api_key=GROQ_API_KEY
     )
context_prompt = PromptTemplate.from_template("""
You are a specialist in understanding and revising blog content.

This is the original blog content (chunked for search):
{chunked_content}

Here is the edit request:
"{edit_request}"                                          
From the blog, identify the exact portion(s) that need editing based on edit request and similarity in meaning or maybe position.

⚠️ Only return the original Markdown that is directly relevant. Do NOT explain. Do NOT expand.

Output:
""")

context_chain = context_prompt | llm | StrOutputParser()

def context_extractor_agent(state: BlogState) -> BlogState:
    if not state.get("edit_request") or not state.get("final_post"):
        return state

    retriever = get_retriever_from_blog_content(state["final_post"])
    docs = retriever.get_relevant_documents(state["edit_request"])
    chunked_text = "\n\n".join(d.page_content for d in docs)

    extracted_context = context_chain.invoke({
        "edit_request": state["edit_request"],
        "chunked_content": chunked_text
    })

    state["edit_context"] = extracted_context
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

   ## Follow these guidelines:
   - Fully understand the **intent** of the edit request before starting.
   - Regenerate a new version of the provided section based on the edit request and surrounding blog context.
   - **Preserve** the same tone, voice, and approximate word count.
   - Use clean, professional **Markdown formatting** with headings, bullet points, and clear structure.
   ---
   ### One-shot example:
   **Original Section:**
   > Artificial Intelligence is making changes in the world. This includes art, where people use computers to make designs. AI tools are popular now. Some artists are not happy.
   **Edit Request:**
   > Make it more engaging and insightful; clarify how AI impacts creativity.
   **Revised Section:**
   > ### The Creative Power of Artificial Intelligence
   > 
   > Artificial Intelligence (AI) is transforming the way we create and perceive art. By leveraging advanced algorithms, artists can now produce unique, compelling designs that were once unimaginable. From generative art to deep learning-driven compositions, AI is not just a tool—it's becoming a creative partner. While some traditionalists voice concerns, many embrace the technology as a new medium of expression.
   ---
   Now, based on that structure and clarity, regenerate the revised Markdown section:
   """


    llm = ChatGroq(
     model="llama3-8b-8192",
     temperature=0.7,
     api_key=GROQ_API_KEY
     )
    revised_section = asyncio.run(llm.ainvoke(prompt)).content
    original = state["final_post"]
    new_post = original.replace(state["edit_context"], revised_section)

    state["final_post"] = new_post
    state["edit_request"] = ""
    state.pop("edit_context", None)
    return state

# ---- LangGraph Flow ----
graph = StateGraph(BlogState)

graph.add_node("research", research_agent)
graph.add_node("summarize", summarizer_agent)
graph.add_node("write", writer_agent)
graph.add_node("cite", citation_agent)
graph.add_node("image", image_agent)
graph.add_node("merge", merge_outputs)
graph.add_node("context_extract", context_extractor_agent)
graph.add_node("edit", editor_agent)

graph.set_entry_point("research")
graph.add_edge("research", "summarize")
graph.add_edge("summarize", "write")
graph.add_edge("write", "cite")
graph.add_edge("cite", "image")
# ⬇️ Conditional edge for image regeneration
def route_after_image(state: BlogState) -> str:
    return "image" if state.get("regenerate_images") else "merge"

graph.add_conditional_edges("image", route_after_image, {
    "image": "image",
    "merge": "merge"
})

def route_after_merge(state: BlogState) -> str:
    return "context_extract" if state.get("edit_request") else END
def route_after_edit(state: BlogState) -> str:
    return "context_extract" if state.get("edit_request") else END

graph.add_conditional_edges("merge", route_after_merge, {
    "context_extract": "context_extract",
    END: END
})

graph.add_edge("context_extract", "edit")
graph.add_conditional_edges("edit", route_after_edit, {
    "context_extract": "context_extract",
    END: END
})

blog_chain = graph.compile()

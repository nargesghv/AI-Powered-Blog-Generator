"""
Enhanced Multi-Agent Blog System with Advanced Search and Content Generation

Key Improvements:
1. Advanced Research Agent with multiple search strategies
2. Intelligent Content Filtering and Ranking
3. Enhanced Writer Agent with better prompts and structure
4. Improved Editor Agent with context-aware editing
5. Better Error Handling and Logging
6. Advanced Search Capabilities
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

# Enhanced LangChain imports
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

# Optional imports that might not be available
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# --------------------
# Environment / Config
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

# --------------------
# LLM Configuration
# --------------------
_ollama_kwargs: Dict[str, Any] = {
    "model": OLLAMA_MODEL,
    "temperature": 0.1,
    "num_ctx": 2048,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}
if OLLAMA_BASE_URL:
    _ollama_kwargs["base_url"] = OLLAMA_BASE_URL
ollama_llm = ChatOllama(**_ollama_kwargs)

# Prefer Groq if key is present; otherwise fall back to Ollama seamlessly
if GROQ_API_KEY:
    groq_summary = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.0, api_key=GROQ_API_KEY)
    groq_writer  = ChatGroq(model="llama3-8b-8192",              temperature=0.3, api_key=GROQ_API_KEY)
    groq_editor  = ChatGroq(model="llama3-8b-8192",              temperature=0.2, api_key=GROQ_API_KEY)
else:
    groq_summary = ollama_llm
    groq_writer  = ollama_llm
    groq_editor  = ollama_llm

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
    # Core content
    topic: str
    target_audience: str
    content_type: str
    word_count_target: int

    # Research
    research_articles: List[Article]
    research_summary: str
    key_points: List[str]
    expert_quotes: List[str]

    # Content generation
    markdown_draft: str
    content_structure: Dict[str, Any]
    seo_keywords: List[str]

    # Assets
    citations: List[Citation]
    images: List[ImageData]
    final_post: str

    # Editing and refinement
    edit_request: str
    edit_context: str
    quality_score: float
    readability_score: float

    # Control flags
    regenerate_images: bool
    enable_advanced_search: bool
    enable_content_analysis: bool

# --------------------
# Advanced Search
# --------------------
from urllib.parse import urlparse

class AdvancedSearchEngine:
    def __init__(self):
        self.search_strategies = {
            'primary': self._primary_search,
            'semantic': self._semantic_search,
            'news': self._news_search,
            'academic': self._academic_search
        }
        self.domain_authority_scores = self._load_domain_authority()

    def _load_domain_authority(self) -> Dict[str, int]:
        return {
            'wikipedia.org': 100,
            'github.com': 95,
            'stackoverflow.com': 90,
            'medium.com': 80,
            'dev.to': 75,
            'hackernews.com': 85,
            'techcrunch.com': 90,
            'arstechnica.com': 85,
            'wired.com': 80,
            'theverge.com': 75,
            'reuters.com': 95,
            'bbc.com': 95,
            'cnn.com': 90,
            'nytimes.com': 95,
            'guardian.com': 90,
        }

    def search_topic(self, topic: str, max_results: int = 15) -> List[Article]:
        all_articles: List[Article] = []
        primary_results = self._primary_search(topic, max_results // 2)
        all_articles.extend(primary_results)
        semantic_results = self._semantic_search(topic, max_results // 4)
        all_articles.extend(semantic_results)
        news_results = self._news_search(topic, max_results // 4)
        all_articles.extend(news_results)
        unique_articles = self._deduplicate_articles(all_articles)
        ranked_articles = self._rank_articles(unique_articles, topic)
        return ranked_articles[:max_results]

    def _primary_search(self, topic: str, max_results: int) -> List[Article]:
        if not SERPAPI_KEY:
            return []
        try:
            params = {
                "engine": "google",
                "q": topic,
                "api_key": SERPAPI_KEY,
                "num": max_results,
                "gl": "us",
                "hl": "en",
                "safe": "active",
            }
            response = requests.get("https://serpapi.com/search", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            articles: List[Article] = []
            for result in data.get("organic_results", []):
                article = self._parse_serpapi_result(result, topic)
                if article:
                    articles.append(article)
            return articles
        except Exception as e:
            print(f"Primary search error: {e}")
            return []

    def _semantic_search(self, topic: str, max_results: int) -> List[Article]:
        if not SERPAPI_KEY:
            return []
        related_terms = self._generate_related_terms(topic)
        articles: List[Article] = []
        for term in related_terms[:3]:
            try:
                params = {
                    "engine": "google",
                    "q": f"{topic} {term}",
                    "api_key": SERPAPI_KEY,
                    "num": max_results // 3 or 1,
                    "gl": "us",
                    "hl": "en",
                }
                response = requests.get("https://serpapi.com/search", params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                for result in data.get("organic_results", []):
                    article = self._parse_serpapi_result(result, topic)
                    if article:
                        articles.append(article)
                time.sleep(0.5)
            except Exception as e:
                print(f"Semantic search error for term '{term}': {e}")
                continue
        return articles

    def _news_search(self, topic: str, max_results: int) -> List[Article]:
        if not SERPAPI_KEY:
            return []
        try:
            params = {
                "engine": "google",
                "q": topic,
                "api_key": SERPAPI_KEY,
                "num": max_results,
                "tbm": "nws",
                "tbs": "qdr:m",
            }
            response = requests.get("https://serpapi.com/search", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            articles: List[Article] = []
            for result in data.get("news_results", []):
                article = self._parse_news_result(result, topic)
                if article:
                    articles.append(article)
            return articles
        except Exception as e:
            print(f"News search error: {e}")
            return []

    def _academic_search(self, topic: str, max_results: int) -> List[Article]:
        return []

    def _generate_related_terms(self, topic: str) -> List[str]:
        expansions = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning'],
            'programming': ['coding', 'development', 'software engineering'],
            'web': ['website', 'web development', 'frontend', 'backend'],
            'data': ['analytics', 'big data', 'data science', 'statistics'],
            'security': ['cybersecurity', 'privacy', 'encryption', 'protection'],
            'cloud': ['aws', 'azure', 'google cloud', 'serverless'],
            'mobile': ['ios', 'android', 'app development', 'react native']
        }
        related: List[str] = []
        topic_lower = topic.lower()
        for key, terms in expansions.items():
            if key in topic_lower:
                related.extend(terms)
        return related[:5]

    def _parse_serpapi_result(self, result: Dict, topic: str) -> Optional[Article]:
        try:
            title = result.get("title", "")
            url = result.get("link", "")
            snippet = result.get("snippet", "") or result.get("snippet_highlighted_words", "")
            if isinstance(snippet, list):
                snippet = " ".join(snippet)
            if not title or not url:
                return None
            domain = urlparse(url).netloc
            relevance_score = self._calculate_relevance_score(title, snippet, topic)
            quality_score = self._calculate_quality_score(title, snippet, domain)
            return Article(
                title=title,
                url=url,
                snippet=snippet,
                domain=domain,
                relevance_score=relevance_score,
                quality_score=quality_score,
                word_count=len(snippet.split()),
                has_images=bool(result.get("thumbnail")),
            )
        except Exception as e:
            print(f"Error parsing SerpAPI result: {e}")
            return None

    def _parse_news_result(self, result: Dict, topic: str) -> Optional[Article]:
        try:
            title = result.get("title", "")
            url = result.get("link", "")
            snippet = result.get("snippet", "")
            date = result.get("date", "")
            if not title or not url:
                return None
            domain = urlparse(url).netloc
            relevance_score = self._calculate_relevance_score(title, snippet, topic)
            quality_score = self._calculate_quality_score(title, snippet, domain)
            return Article(
                title=title,
                url=url,
                snippet=snippet,
                domain=domain,
                relevance_score=relevance_score,
                quality_score=quality_score,
                publish_date=date,
                word_count=len(snippet.split()),
                has_images=bool(result.get("thumbnail")),
            )
        except Exception as e:
            print(f"Error parsing news result: {e}")
            return None

    def _calculate_relevance_score(self, title: str, snippet: str, topic: str) -> float:
        topic_words = set(topic.lower().split())
        content = f"{title} {snippet}".lower()
        content_words = set(content.split())
        overlap = len(topic_words.intersection(content_words))
        base_score = overlap / len(topic_words) if topic_words else 0.0
        if topic.lower() in content:
            base_score += 0.3
        quality_indicators = ['study', 'research', 'analysis', 'report', 'data', 'statistics', 'survey']
        quality_boost = sum(0.1 for indicator in quality_indicators if indicator in content)
        return min(1.0, base_score + quality_boost)

    def _calculate_quality_score(self, title: str, snippet: str, domain: str) -> float:
        domain_score = self.domain_authority_scores.get(domain, 50) / 100
        content = f"{title} {snippet}".lower()
        quality_indicators = ['study', 'research', 'analysis', 'report', 'data', 'statistics']
        content_quality = sum(0.1 for indicator in quality_indicators if indicator in content)
        length_score = min(1.0, len(snippet.split()) / 50)
        return (domain_score * 0.5 + content_quality * 0.3 + length_score * 0.2)

    def _deduplicate_articles(self, articles: List[Article]) -> List[Article]:
        seen_urls: set = set()
        seen_titles: List[str] = []
        unique_articles: List[Article] = []
        for article in articles:
            if article.url in seen_urls:
                continue
            title_lower = article.title.lower()
            if any(self._title_similarity(title_lower, t) > 0.8 for t in seen_titles):
                continue
            seen_urls.add(article.url)
            seen_titles.append(title_lower)
            unique_articles.append(article)
        return unique_articles

    def _title_similarity(self, title1: str, title2: str) -> float:
        words1 = set(title1.split())
        words2 = set(title2.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

    def _rank_articles(self, articles: List[Article], topic: str) -> List[Article]:
        def combined_score(a: Article) -> float:
            return a.relevance_score * 0.6 + a.quality_score * 0.4
        return sorted(articles, key=combined_score, reverse=True)

search_engine = AdvancedSearchEngine()

# --------------------
# Agents
# --------------------

def enhanced_research_agent(state: BlogState) -> BlogState:
    topic = state.get("topic", "").strip()
    enable_advanced_search = state.get("enable_advanced_search", True)
    if not topic:
        state["research_articles"] = []
        return state
    print(f"🔍 Researching topic: {topic}")
    try:
        if enable_advanced_search:
            articles = search_engine.search_topic(topic, max_results=12)
        else:
            articles = _simple_search(topic)
        research_articles = []
        for article in articles:
            research_articles.append({
                "title": article.title,
                "url": article.url,
                "snippet": article.snippet,
                "domain": article.domain,
                "relevance_score": article.relevance_score,
                "quality_score": article.quality_score,
            })
        state["research_articles"] = research_articles
        print(f"✅ Found {len(research_articles)} research articles")
    except Exception as e:
        print(f"❌ Research error: {e}")
        state["research_articles"] = []
    return state


def _simple_search(topic: str) -> List[Article]:
    if not SERPAPI_KEY:
        return []
    try:
        params = {"engine": "google", "q": topic, "api_key": SERPAPI_KEY, "num": 8}
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        organic = response.json().get("organic_results", [])
        articles: List[Article] = []
        for result in organic[:5]:
            title = result.get("title", "")
            url = result.get("link", "")
            snippet = result.get("snippet", "") or result.get("snippet_highlighted_words", "")
            if isinstance(snippet, list):
                snippet = " ".join(snippet)
            if title and url:
                domain = urlparse(url).netloc
                articles.append(Article(
                    title=title,
                    url=url,
                    snippet=snippet,
                    domain=domain,
                    relevance_score=0.8,
                    quality_score=0.7,
                ))
        return articles
    except Exception:
        return []


def enhanced_summarizer_agent(state: BlogState) -> BlogState:
    articles = state.get("research_articles", [])
    topic = state.get("topic", "")
    if not articles:
        state["research_summary"] = "No research articles found to summarize."
        state["key_points"] = []
        state["expert_quotes"] = []
        return state
    print(f"📝 Summarizing {len(articles)} articles")
    try:
        content = "\n\n".join([
            f"**{a['title']}** (Score: {a.get('relevance_score', 0):.2f})\n"
            f"Domain: {a.get('domain', 'Unknown')}\n"
            f"Content: {a['snippet']}"
            for a in articles
        ])
        summarizer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research analyst. Your task is to:
1. Extract key insights and themes from research articles
2. Identify the most important points for a blog post
3. Note any expert quotes or statistics
4. Organize information logically

Focus on:
- Main themes and trends
- Supporting evidence and data
- Expert opinions and quotes
- Recent developments or news
- Practical applications or implications"""),
            ("user", """Topic: {topic}

Research Articles:
{content}

Please provide:
1. A comprehensive summary of key findings
2. A list of 5-7 key points for the blog
3. Any notable quotes or statistics (if available)

Format your response as:
SUMMARY:
[Your comprehensive summary here]

KEY POINTS:
- [Key point 1]
- [Key point 2]
- [etc.]

QUOTES/STATISTICS:
- [Quote or statistic 1]
- [Quote or statistic 2]
- [etc.]""")
        ])
        summarizer_chain = summarizer_prompt | groq_summary | StrOutputParser()
        result = summarizer_chain.invoke({"topic": topic, "content": content})
        summary, key_points, quotes = _parse_summarizer_response(result)
        state["research_summary"] = summary
        state["key_points"] = key_points
        state["expert_quotes"] = quotes
        print(f"✅ Generated summary with {len(key_points)} key points")
    except Exception as e:
        print(f"❌ Summarization error: {e}")
        state["research_summary"] = "Summarization failed. Using basic summary."
        state["key_points"] = []
        state["expert_quotes"] = []
    return state


def _parse_summarizer_response(response: str) -> Tuple[str, List[str], List[str]]:
    try:
        sections = response.split('\n\n')
        summary = ""
        key_points: List[str] = []
        quotes: List[str] = []
        current_section = None
        for section in sections:
            if section.startswith("SUMMARY:"):
                summary = section.replace("SUMMARY:", "").strip()
                current_section = "summary"
            elif section.startswith("KEY POINTS:"):
                current_section = "key_points"
                lines = section.replace("KEY POINTS:", "").strip().split('\n')
                key_points = [line.strip('- ').strip() for line in lines if line.strip()]
            elif section.startswith("QUOTES/STATISTICS:"):
                current_section = "quotes"
                lines = section.replace("QUOTES/STATISTICS:", "").strip().split('\n')
                quotes = [line.strip('- ').strip() for line in lines if line.strip()]
            elif current_section == "summary" and section.strip():
                summary += " " + section.strip()
        return summary, key_points, quotes
    except Exception:
        return response, [], []


def enhanced_writer_agent(state: BlogState) -> BlogState:
    topic = state.get("topic", "")
    summary = state.get("research_summary", "No summary available.")
    key_points = state.get("key_points", [])
    quotes = state.get("expert_quotes", [])
    target_audience = state.get("target_audience", "tech-savvy professionals")
    content_type = state.get("content_type", "informative article")
    word_count_target = state.get("word_count_target", 1000)
    print(f"✍️ Writing blog post: {topic}")
    try:
        writer_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a professional blog writer specializing in {content_type}. 
Your audience is {target_audience}.

Writing Guidelines:
1. Create engaging, well-structured content
2. Use clear headings and subheadings
3. Include practical examples and insights
4. Maintain a professional yet accessible tone
5. Target approximately {word_count_target} words
6. Use proper Markdown formatting
7. Include relevant statistics and quotes when available
8. End with actionable takeaways or conclusions

Structure your post with:
- Compelling introduction
- Well-organized main sections
- Supporting evidence and examples
- Clear conclusions
- Call-to-action if appropriate"""),
            ("user", """Topic: {topic}

Research Summary:
{summary}

Key Points to Cover:
{key_points}

Expert Quotes/Statistics:
{quotes}

Please write a comprehensive blog post that incorporates this research while maintaining originality and engaging writing style.""")
        ])
        key_points_text = "\n".join([f"- {point}" for point in key_points])
        quotes_text = "\n".join([f"- {quote}" for quote in quotes]) if quotes else "None available"
        writer_chain = writer_prompt | groq_writer | StrOutputParser()
        markdown_draft = writer_chain.invoke({
            "topic": topic,
            "summary": summary,
            "key_points": key_points_text,
            "quotes": quotes_text,
        })
        content_analysis = _analyze_content_quality(markdown_draft, topic)
        state["markdown_draft"] = markdown_draft.strip()
        state["content_structure"] = content_analysis["structure"]
        state["seo_keywords"] = content_analysis["keywords"]
        state["quality_score"] = content_analysis["quality_score"]
        state["readability_score"] = content_analysis["readability_score"]
        print(f"✅ Generated {len(markdown_draft.split())} words with quality score: {content_analysis['quality_score']:.2f}")
    except Exception as e:
        print(f"❌ Writing error: {e}")
        state["markdown_draft"] = f"# {topic}\n\nError generating content: {str(e)}"
        state["quality_score"] = 0.0
        state["readability_score"] = 0.0
    return state


def _analyze_content_quality(content: str, topic: str) -> Dict[str, Any]:
    try:
        headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        word_count = len(content.split())
        topic_words = set(topic.lower().split())
        content_words = content.lower().split()
        keyword_density = sum(1 for w in content_words if w in topic_words) / max(1, len(content_words))
        has_intro = any('introduction' in h.lower() or h.lower().startswith(topic.lower()[:10]) for h in headings)
        has_conclusion = any('conclusion' in h.lower() or 'summary' in h.lower() for h in headings)
        has_subheadings = len([h for h in headings if h.startswith('##')]) >= 2
        structure_score = sum([has_intro, has_conclusion, has_subheadings]) / 3
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = word_count / max(1, len(sentences))
        readability_score = max(0.0, 1 - (avg_sentence_length - 15) / 20)
        quality_score = (structure_score * 0.4 + readability_score * 0.3 + min(1.0, keyword_density * 10) * 0.3)
        return {
            "structure": {
                "headings": headings,
                "word_count": word_count,
                "has_intro": has_intro,
                "has_conclusion": has_conclusion,
                "has_subheadings": has_subheadings,
            },
            "keywords": list(topic_words),
            "quality_score": quality_score,
            "readability_score": readability_score,
        }
    except Exception as e:
        print(f"Content analysis error: {e}")
        return {
            "structure": {"headings": [], "word_count": 0},
            "keywords": [],
            "quality_score": 0.0,
            "readability_score": 0.0,
        }


def enhanced_image_agent(state: BlogState) -> List[ImageData]:
    topic = state.get("topic", "")
    content = state.get("markdown_draft", "")
    if not topic or not PEXELS_API_KEY:
        return []
    print(f"🖼️ Finding images for: {topic}")
    try:
        content_keywords = _extract_image_keywords(content, topic)
        images: List[ImageData] = []
        for keyword in content_keywords[:3]:
            try:
                res = requests.get(
                    "https://api.pexels.com/v1/search",
                    headers={"Authorization": PEXELS_API_KEY},
                    params={
                        "query": keyword,
                        "per_page": 2,
                        "page": random.randint(1, 5),
                        "orientation": "landscape",
                    },
                    timeout=30,
                )
                res.raise_for_status()
                photos = res.json().get("photos", [])
                for photo in photos:
                    relevance_score = _calculate_image_relevance(photo, keyword, topic)
                    if relevance_score > 0.3:
                        images.append(ImageData(
                            url=photo["src"]["medium"],
                            alt=photo.get("alt", f"{keyword} related image"),
                            license="Pexels",
                            width=photo.get("width", 0),
                            height=photo.get("height", 0),
                            relevance_score=relevance_score,
                        ))
                time.sleep(0.5)
            except Exception as e:
                print(f"Image search error for '{keyword}': {e}")
                continue
        images.sort(key=lambda x: x.relevance_score, reverse=True)
        return images[:3]
    except Exception as e:
        print(f"❌ Image search error: {e}")
        return []


def _extract_image_keywords(content: str, topic: str) -> List[str]:
    topic_words = topic.lower().split()
    content_words = content.lower().split()
    stopish = {
        'this','that','with','from','they','have','been','were','said','each','which','their',
        'time','will','about','there','when','your','can','she','use','many','some','very',
        'come','here','just','like','long','make','over','such','take','than','them','well'
    }
    important_words: List[str] = []
    for word in content_words:
        if len(word) > 4 and word not in stopish:
            important_words.append(word)
    keywords = topic_words + important_words[:5]
    # Remove duplicates while preserving order
    seen = set()
    uniq: List[str] = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq[:5]


def _calculate_image_relevance(photo: Dict, keyword: str, topic: str) -> float:
    alt_text = (photo.get("alt") or "").lower()
    keyword_score = 1.0 if keyword.lower() in alt_text else 0.5
    topic_words = topic.lower().split()
    topic_score = sum(0.2 for w in topic_words if w in alt_text)
    width = photo.get("width", 0)
    height = photo.get("height", 0)
    quality_score = min(1.0, (width * height) / float(1920 * 1080) if width and height else 0.0)
    return (keyword_score * 0.4 + topic_score * 0.4 + quality_score * 0.2)


def enhanced_citation_agent(state: BlogState) -> BlogState:
    articles = state.get("research_articles", [])
    citations: List[Citation] = []
    for a in articles:
        url = a.get("url")
        if url:
            citations.append(Citation(
                title=a.get("title", ""),
                url=url,
                domain=a.get("domain", ""),
                citation_type="web",
            ))
    if articles and "relevance_score" in articles[0]:
        def score_of(x: Citation) -> float:
            for a in articles:
                if a["url"] == x.url:
                    return a.get("relevance_score", 0.0)
            return 0.0
        citations.sort(key=score_of, reverse=True)
    state["citations"] = citations
    return state


def enhanced_merge_outputs(state: BlogState) -> BlogState:
    post = state.get("markdown_draft", "") or ""
    images = state.get("images", [])
    citations = state.get("citations", [])
    images_section = ""
    if images:
        images_section = "\n\n## 📸 Visual Elements\n\n"
        for i, img in enumerate(images, 1):
            images_section += f"![{img.alt}]({img.url})\n*Figure {i}: {img.alt}*\n\n"
    citations_section = ""
    if citations:
        citations_section = "\n\n## 📚 References\n\n"
        for i, c in enumerate(citations, 1):
            domain_display = f" ({c.domain})" if c.domain else ""
            citations_section += f"{i}. [{c.title}]({c.url}){domain_display}\n"
    metadata_section = f"\n\n---\n\n*This article was generated using AI research and writing tools. Last updated: {datetime.now().strftime('%Y-%m-%d')}*"
    state["final_post"] = f"{post}{images_section}{citations_section}{metadata_section}".strip()
    return state

# --------------------
# RAG / Editing
# --------------------
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def get_enhanced_retriever(markdown_text: str):
    loader = [Document(page_content=markdown_text)]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    doc_chunks = text_splitter.split_documents(loader)
    for i, chunk in enumerate(doc_chunks):
        chunk.metadata = {
            "chunk_id": i,
            "word_count": len(chunk.page_content.split()),
            "has_heading": bool(re.search(r'^#{1,6}\s+', chunk.page_content, re.MULTILINE)),
        }
    vectorstore = Chroma.from_documents(
        doc_chunks,
        embedding=embedding_model,
        collection_name=f"edit-context-{int(time.time())}",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def enhanced_context_extractor_agent(state: BlogState) -> BlogState:
    if not state.get("edit_request") or not state.get("final_post"):
        return state
    print(f"🔍 Extracting context for edit: {state['edit_request'][:50]}...")
    try:
        retriever = get_enhanced_retriever(state["final_post"])
        docs = retriever.get_relevant_documents(state["edit_request"])
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at understanding blog content and edit requests.
Your task is to identify the exact portions of the blog that need to be modified based on the edit request.

Guidelines:
1. Only return the original Markdown content that is directly relevant
2. Include enough context to understand the section
3. Do NOT explain or expand on the content
4. Do NOT add new information
5. Preserve the original formatting exactly"""),
            ("user", """Edit Request: "{edit_request}"

Blog Content (relevant sections):
{chunked_content}

Return only the original Markdown that needs to be edited:""")
        ])
        chunked_text = "\n\n---\n\n".join(d.page_content for d in docs)
        context_chain = context_prompt | groq_editor | StrOutputParser()
        extracted = context_chain.invoke({
            "edit_request": state["edit_request"],
            "chunked_content": chunked_text,
        })
        state["edit_context"] = extracted.strip()
        print(f"✅ Extracted context: {len(extracted)} characters")
    except Exception as e:
        print(f"❌ Context extraction error: {e}")
        state["edit_context"] = ""
    return state


def enhanced_editor_agent(state: BlogState) -> BlogState:
    if not state.get("edit_context") or not state.get("edit_request"):
        return state
    print("✏️ Editing content based on request")
    try:
        editor_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional blog editor with expertise in content improvement.

Your task is to edit the provided content section according to the edit request while:
1. Maintaining the original tone and style
2. Preserving the approximate word count
3. Keeping the Markdown formatting intact
4. Ensuring the edit flows naturally with the rest of the content
5. Making improvements in clarity, structure, and engagement

Guidelines:
- Apply the requested changes precisely
- Improve readability and flow
- Maintain professional quality
- Use proper Markdown formatting
- Keep the content engaging and informative"""),
            ("user", """Edit Request: "{edit_request}"

Content Section to Edit:
---
{edit_context}
---

Please provide the revised version of this section:""")
        ])
        editor_chain = editor_prompt | groq_editor | StrOutputParser()
        revised_section = editor_chain.invoke({
            "edit_request": state["edit_request"],
            "edit_context": state["edit_context"],
        })
        original = state.get("final_post", "")
        target = state.get("edit_context", "")
        if target and target in original:
            state["final_post"] = original.replace(target, revised_section.strip(), 1)
            print("✅ Successfully edited content")
        else:
            print("⚠️ Could not find exact match for replacement")
        # Clear edit request/context after applying
        state["edit_request"] = ""
        state.pop("edit_context", None)
    except Exception as e:
        print(f"❌ Editing error: {e}")
    return state

# --------------------
# LangGraph Flow
# --------------------

graph = StateGraph(BlogState)
graph.add_node("research", enhanced_research_agent)
graph.add_node("summarize", enhanced_summarizer_agent)
graph.add_node("write", enhanced_writer_agent)
graph.add_node("cite", enhanced_citation_agent)
graph.add_node("image", lambda state: {**state, "images": enhanced_image_agent(state)})
graph.add_node("merge", enhanced_merge_outputs)
graph.add_node("context_extract", enhanced_context_extractor_agent)
graph.add_node("edit", enhanced_editor_agent)

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
    return END

graph.add_conditional_edges(
    "merge",
    route_after_merge,
    {"image": "image", "context_extract": "context_extract", END: END},
)

graph.add_edge("context_extract", "edit")

def route_after_edit(state: BlogState) -> str:
    if state.get("edit_request"):
        return "context_extract"
    return END

graph.add_conditional_edges(
    "edit",
    route_after_edit,
    {"context_extract": "context_extract", END: END},
)

enhanced_blog_chain = graph.compile()

__all__ = ["enhanced_blog_chain", "BlogState", "search_engine"]

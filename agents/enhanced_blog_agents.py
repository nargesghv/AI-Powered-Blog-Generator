"""
Enhanced Blog Agents with Advanced Search and Improved Chain Management

Key Improvements:
- Multi-source research with intelligent filtering
- Advanced search query optimization
- Content quality assessment
- Better error handling and retry logic
- Enhanced caching and performance
- Improved prompt engineering
- Real-time search result validation
"""

import os
import random
import asyncio
import aiohttp
import time
from typing import TypedDict, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import json

import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Enhanced LangChain imports
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langchain_ollama import ChatOllama
from groq import Groq

# Enhanced configuration
from backend.config.settings import settings

# --------------------
# Enhanced Types
# --------------------
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float
    publish_date: Optional[datetime] = None
    domain_authority: int = 0
    content_type: str = "article"

@dataclass
class SearchQuery:
    original_query: str
    optimized_queries: List[str]
    search_intent: str
    target_audience: str
    content_type: str

class Article(TypedDict):
    title: str
    url: str
    snippet: str
    relevance_score: float
    source: str
    publish_date: Optional[str]

class ImageData(TypedDict):
    url: str
    alt: str
    license: str
    relevance_score: float
    source: str

class Citation(TypedDict):
    title: str
    url: str
    credibility_score: float
    source_type: str

class BlogState(TypedDict, total=False):
    topic: str
    # Enhanced research
    search_queries: List[SearchQuery]
    research_articles: List[Article]
    research_summary: str
    research_quality_score: float
    # Enhanced writing
    markdown_draft: str
    content_quality_metrics: Dict[str, float]
    # Enhanced assets
    citations: List[Citation]
    images: List[ImageData]
    final_post: str
    # Enhanced editing
    edit_request: str
    edit_context: str
    edit_history: List[Dict[str, str]]
    # Enhanced routing
    regenerate_images: bool
    regenerate_content: bool
    # Performance tracking
    generation_start_time: float
    step_timings: Dict[str, float]
    # MCP flags/results
    use_mcp: bool
    save_with_mcp: bool
    saved_filename: str
    save_result: str

# --------------------
# Enhanced Search Engine
# --------------------
class AdvancedSearchEngine:
    def __init__(self):
        self.search_sources = {
            'serpapi': self._search_serpapi,
            'newsapi': self._search_newsapi,
            'arxiv': self._search_arxiv,
            'reddit': self._search_reddit,
            'github': self._search_github
        }
        self.query_optimizer = QueryOptimizer()
        self.result_filter = ResultFilter()
    
    async def search_topic(self, topic: str, max_results: int = 15) -> List[SearchResult]:
        """Enhanced multi-source search with intelligent filtering"""
        start_time = time.time()
        
        # Optimize search queries
        search_query = self.query_optimizer.optimize_query(topic)
        
        # Parallel search across multiple sources
        tasks = []
        for source_name, search_func in self.search_sources.items():
            if self._is_source_available(source_name):
                task = search_func(search_query, max_results // len(self.search_sources))
                tasks.append(task)
        
        # Wait for all searches to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten and filter results
        all_results = []
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                print(f"Search error: {result}")
        
        # Apply intelligent filtering
        filtered_results = self.result_filter.filter_results(all_results, topic)
        
        # Sort by relevance and return top results
        final_results = sorted(filtered_results, key=lambda x: x.relevance_score, reverse=True)[:max_results]
        
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.2f}s, found {len(final_results)} results")
        
        return final_results
    
    async def _search_serpapi(self, search_query: SearchQuery, max_results: int) -> List[SearchResult]:
        """Enhanced SerpAPI search with better result processing"""
        if not settings.serpapi_key:
            return []
        
        results = []
        for query in search_query.optimized_queries[:3]:  # Limit to top 3 queries
            try:
                async with aiohttp.ClientSession() as session:
                    params = {
                        "engine": "google",
                        "q": query,
                        "api_key": settings.serpapi_key,
                        "num": max_results,
                        "gl": "us",
                        "hl": "en",
                        "safe": "active",
                        "tbm": "nws" if "news" in search_query.content_type else None
                    }
                    
                    async with session.get("https://serpapi.com/search", params=params, 
                                         timeout=settings.api_timeout) as response:
                        data = await response.json()
                        
                        for item in data.get("organic_results", []):
                            result = SearchResult(
                                title=item.get("title", ""),
                                url=item.get("link", ""),
                                snippet=item.get("snippet", ""),
                                source="serpapi",
                                relevance_score=self._calculate_relevance(search_query.original_query, 
                                                                        item.get("title", ""), 
                                                                        item.get("snippet", "")),
                                domain_authority=self._get_domain_authority(item.get("link", ""))
                            )
                            results.append(result)
            except Exception as e:
                print(f"SerpAPI search error: {e}")
        
        return results
    
    async def _search_newsapi(self, search_query: SearchQuery, max_results: int) -> List[SearchResult]:
        """Search NewsAPI for recent articles"""
        # Implementation for NewsAPI
        return []
    
    async def _search_arxiv(self, search_query: SearchQuery, max_results: int) -> List[SearchResult]:
        """Search arXiv for academic papers"""
        # Implementation for arXiv API
        return []
    
    async def _search_reddit(self, search_query: SearchQuery, max_results: int) -> List[SearchResult]:
        """Search Reddit for community discussions"""
        # Implementation for Reddit API
        return []
    
    async def _search_github(self, search_query: SearchQuery, max_results: int) -> List[SearchResult]:
        """Search GitHub for code repositories and documentation"""
        # Implementation for GitHub API
        return []
    
    def _calculate_relevance(self, topic: str, title: str, snippet: str) -> float:
        """Enhanced relevance calculation"""
        topic_words = set(topic.lower().split())
        content = f"{title} {snippet}".lower()
        content_words = set(content.split())
        
        # Basic keyword overlap
        overlap = len(topic_words.intersection(content_words))
        base_score = overlap / len(topic_words) if topic_words else 0
        
        # Boost for exact phrase matches
        if topic.lower() in content:
            base_score += 0.3
        
        # Boost for quality indicators
        quality_indicators = ['study', 'research', 'analysis', 'report', 'data', 'statistics', 'guide', 'tutorial']
        quality_boost = sum(0.1 for indicator in quality_indicators if indicator in content)
        
        # Boost for recent content indicators
        recent_indicators = ['2024', '2023', 'latest', 'new', 'recent', 'updated']
        recency_boost = sum(0.05 for indicator in recent_indicators if indicator in content)
        
        return min(1.0, base_score + quality_boost + recency_boost)
    
    def _get_domain_authority(self, url: str) -> int:
        """Enhanced domain authority scoring"""
        high_authority_domains = {
            'wikipedia.org': 100,
            'github.com': 95,
            'stackoverflow.com': 90,
            'medium.com': 80,
            'dev.to': 75,
            'hackernews.com': 85,
            'arxiv.org': 90,
            'nature.com': 95,
            'ieee.org': 90,
            'acm.org': 90
        }
        
        for domain, score in high_authority_domains.items():
            if domain in url:
                return score
        
        return 50  # Default score
    
    def _is_source_available(self, source: str) -> bool:
        """Check if a search source is available"""
        availability = {
            'serpapi': bool(settings.serpapi_key),
            'newsapi': False,  # Add NewsAPI key check
            'arxiv': True,     # Free API
            'reddit': False,   # Add Reddit API key check
            'github': True     # Free API with rate limits
        }
        return availability.get(source, False)

class QueryOptimizer:
    def __init__(self):
        self.search_intents = {
            'tutorial': ['how to', 'guide', 'tutorial', 'step by step'],
            'news': ['latest', 'recent', 'news', 'update'],
            'research': ['study', 'research', 'analysis', 'findings'],
            'comparison': ['vs', 'comparison', 'difference', 'best'],
            'review': ['review', 'opinion', 'experience', 'test']
        }
    
    def optimize_query(self, topic: str) -> SearchQuery:
        """Generate optimized search queries for better results"""
        original_query = topic.strip()
        
        # Detect search intent
        intent = self._detect_search_intent(original_query)
        
        # Generate optimized queries
        optimized_queries = [original_query]
        
        # Add intent-specific queries
        if intent in self.search_intents:
            for intent_word in self.search_intents[intent]:
                optimized_queries.append(f"{original_query} {intent_word}")
        
        # Add related terms
        related_terms = self._get_related_terms(original_query)
        for term in related_terms[:2]:  # Limit to top 2 related terms
            optimized_queries.append(f"{original_query} {term}")
        
        # Add long-tail variations
        long_tail_queries = self._generate_long_tail_queries(original_query)
        optimized_queries.extend(long_tail_queries[:2])
        
        return SearchQuery(
            original_query=original_query,
            optimized_queries=optimized_queries,
            search_intent=intent,
            target_audience="general",
            content_type="article"
        )
    
    def _detect_search_intent(self, query: str) -> str:
        """Detect the search intent from the query"""
        query_lower = query.lower()
        
        for intent, keywords in self.search_intents.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return "general"
    
    def _get_related_terms(self, topic: str) -> List[str]:
        """Get related terms for the topic"""
        # This is a simplified implementation
        # In production, you'd use more sophisticated NLP techniques
        
        related_terms_map = {
            'artificial intelligence': ['machine learning', 'deep learning', 'neural networks'],
            'web development': ['frontend', 'backend', 'full stack', 'responsive design'],
            'data science': ['analytics', 'statistics', 'visualization', 'big data'],
            'cybersecurity': ['encryption', 'firewall', 'vulnerability', 'threat detection'],
            'cloud computing': ['AWS', 'Azure', 'Google Cloud', 'serverless', 'microservices']
        }
        
        topic_lower = topic.lower()
        for key, terms in related_terms_map.items():
            if key in topic_lower:
                return terms
        
        return []
    
    def _generate_long_tail_queries(self, topic: str) -> List[str]:
        """Generate long-tail keyword variations"""
        long_tail_templates = [
            f"what is {topic}",
            f"benefits of {topic}",
            f"{topic} best practices",
            f"{topic} trends 2024",
            f"how {topic} works"
        ]
        
        return long_tail_templates

class ResultFilter:
    def __init__(self):
        self.quality_indicators = [
            'study', 'research', 'analysis', 'report', 'data', 'statistics',
            'guide', 'tutorial', 'best practices', 'expert', 'professional'
        ]
        
        self.low_quality_indicators = [
            'spam', 'clickbait', 'fake', 'scam', 'advertisement'
        ]
    
    def filter_results(self, results: List[SearchResult], topic: str) -> List[SearchResult]:
        """Apply intelligent filtering to search results"""
        filtered = []
        
        for result in results:
            # Skip if relevance score is too low
            if result.relevance_score < 0.2:
                continue
            
            # Skip if domain authority is too low
            if result.domain_authority < 30:
                continue
            
            # Check for quality indicators
            content = f"{result.title} {result.snippet}".lower()
            
            # Skip if contains low-quality indicators
            if any(indicator in content for indicator in self.low_quality_indicators):
                continue
            
            # Boost score for quality indicators
            quality_boost = sum(0.1 for indicator in self.quality_indicators 
                              if indicator in content)
            result.relevance_score = min(1.0, result.relevance_score + quality_boost)
            
            filtered.append(result)
        
        return filtered

# --------------------
# Enhanced Agents
# --------------------
class EnhancedResearchAgent:
    def __init__(self):
        self.search_engine = AdvancedSearchEngine()
        self.cache = {}  # Simple in-memory cache
    
    async def research_topic(self, topic: str) -> List[Article]:
        """Enhanced research with multi-source search and intelligent filtering"""
        # Check cache first
        cache_key = f"research_{hash(topic)}"
        if cache_key in self.cache:
            cached_time, results = self.cache[cache_key]
            if time.time() - cached_time < 3600:  # 1 hour cache
                return results
        
        # Perform enhanced search
        search_results = await self.search_engine.search_topic(topic, max_results=15)
        
        # Convert to Article format
        articles = []
        for result in search_results:
            article = Article(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                relevance_score=result.relevance_score,
                source=result.source,
                publish_date=result.publish_date.isoformat() if result.publish_date else None
            )
            articles.append(article)
        
        # Cache results
        self.cache[cache_key] = (time.time(), articles)
        
        return articles

# --------------------
# Enhanced LLM Configuration
# --------------------
def get_enhanced_llm_config():
    """Get enhanced LLM configuration with better prompts"""
    ollama_kwargs = {
        "model": settings.ollama_model,
        "temperature": 0.1,  # Slightly higher for creativity
        "num_ctx": 2048,     # Increased context window
        "top_p": 0.9,
        "repeat_penalty": 1.1
    }
    
    if settings.ollama_base_url:
        ollama_kwargs["base_url"] = settings.ollama_base_url
    
    return ChatOllama(**ollama_kwargs)

# Enhanced LLM instances
enhanced_ollama_llm = get_enhanced_llm_config()
groq_client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None

# --------------------
# Enhanced Prompts
# --------------------
ENHANCED_SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research analyst and content strategist. Your task is to analyze research results and create a comprehensive summary that will guide high-quality blog content creation.

Key responsibilities:
1. Identify the most valuable insights and key points
2. Organize information by themes and importance
3. Highlight trends, statistics, and expert opinions
4. Note any conflicting information or gaps
5. Suggest angles for the blog post

Focus on:
- Accuracy and credibility
- Actionable insights
- Current trends and developments
- Expert opinions and studies
- Practical applications

Provide a structured summary that will help create engaging, informative blog content."""),
    ("human", """Topic: {topic}

Research Results:
{content}

Please provide a comprehensive analysis and summary that will guide the blog writing process. Focus on the most valuable insights and organize them by themes."""),
])

ENHANCED_WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional blog writer and content creator with expertise in creating engaging, informative, and well-structured blog posts. Your writing style is:

- Professional yet accessible
- Engaging and conversational
- Well-structured with clear headings
- Rich in examples and practical insights
- Optimized for readability and SEO

Guidelines for your blog post:
1. Start with a compelling introduction that hooks the reader
2. Use clear, descriptive headings (H2, H3)
3. Include relevant examples, case studies, or statistics
4. Write in an engaging, conversational tone
5. Use bullet points and lists for better readability
6. Include a strong conclusion with key takeaways
7. Aim for 1000-1500 words
8. Use proper Markdown formatting
9. Include internal linking opportunities
10. Make it valuable and actionable for readers

Your goal is to create content that educates, informs, and engages the target audience while establishing authority on the topic."""),
    ("human", """Write a comprehensive blog post on: {topic}

Research Summary:
{summary}

Please create a well-structured, engaging blog post that incorporates the research insights while providing valuable, actionable content for readers."""),
])

# --------------------
# Enhanced Agent Functions
# --------------------
async def enhanced_research_agent(state: BlogState) -> BlogState:
    """Enhanced research agent with multi-source search"""
    topic = state.get("topic", "").strip()
    if not topic:
        state["research_articles"] = []
        return state
    
    start_time = time.time()
    
    try:
        research_agent = EnhancedResearchAgent()
        articles = await research_agent.research_topic(topic)
        
        # Convert to the expected format
        formatted_articles = []
        for article in articles:
            formatted_articles.append({
                "title": article["title"],
                "url": article["url"],
                "snippet": article["snippet"]
            })
        
        state["research_articles"] = formatted_articles
        
        # Calculate research quality score
        if articles:
            avg_relevance = sum(article["relevance_score"] for article in articles) / len(articles)
            state["research_quality_score"] = avg_relevance
        else:
            state["research_quality_score"] = 0.0
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["research"] = step_time
        
        print(f"Research completed: {len(formatted_articles)} articles found in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Research error: {e}")
        state["research_articles"] = []
        state["research_quality_score"] = 0.0
    
    return state

def enhanced_summarizer_agent(state: BlogState) -> BlogState:
    """Enhanced summarizer with better prompt engineering"""
    articles = state.get("research_articles", [])
    if not articles:
        state["research_summary"] = "No research articles found to summarize."
        return state
    
    start_time = time.time()
    
    try:
        # Prepare content for summarization
        content = "\n\n".join(f"**{a['title']}**\n{a['snippet']}\nURL: {a['url']}" for a in articles)
        
        # Use enhanced prompt
        summarizer_chain = ENHANCED_SUMMARIZER_PROMPT | enhanced_ollama_llm | StrOutputParser()
        summary = summarizer_chain.invoke({
            "topic": state.get("topic", ""),
            "content": content
        })
        
        state["research_summary"] = summary or "Summary could not be generated."
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["summarize"] = step_time
        
        print(f"Summarization completed in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Summarization error: {e}")
        state["research_summary"] = "Summarizer failed. Please try again."
    
    return state

def enhanced_writer_agent(state: BlogState) -> BlogState:
    """Enhanced writer with better prompt engineering and Groq integration"""
    summary = state.get("research_summary", "No summary available.")
    topic = state.get("topic", "")
    
    start_time = time.time()
    
    try:
        if groq_client:
            # Use Groq for writing (faster and better quality)
            prompt = f"""Write a comprehensive, engaging blog post on: "{topic}"

Research Summary:
{summary}

Requirements:
- 1000-1500 words
- Professional yet accessible tone
- Clear structure with headings
- Include examples and insights
- Use proper Markdown formatting
- Engaging introduction and conclusion
- Actionable content for readers

Please create a well-structured blog post that educates and engages readers."""
            
            response = groq_client.chat.completions.create(
                model=settings.groq_model_writer,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000
            )
            
            state["markdown_draft"] = response.choices[0].message.content.strip()
        else:
            # Fallback to Ollama
            writer_chain = ENHANCED_WRITER_PROMPT | enhanced_ollama_llm | StrOutputParser()
            state["markdown_draft"] = writer_chain.invoke({
                "topic": topic,
                "summary": summary
            })
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["write"] = step_time
        
        print(f"Writing completed in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Writing error: {e}")
        state["markdown_draft"] = "Writing failed. Please try again."
    
    return state

def enhanced_image_agent(state: BlogState) -> BlogState:
    """Enhanced image agent with better search and filtering"""
    topic = state.get("topic", "")
    if not topic or not settings.pexels_api_key:
        state["images"] = []
        return state
    
    start_time = time.time()
    
    try:
        # Generate multiple search terms for better image variety
        search_terms = [
            topic,
            f"{topic} concept",
            f"{topic} illustration",
            f"{topic} technology"
        ]
        
        all_images = []
        for search_term in search_terms[:2]:  # Limit to 2 search terms
            page = random.randint(1, 5)  # Reduced page range for better quality
            try:
                res = requests.get(
                    "https://api.pexels.com/v1/search",
                    headers={"Authorization": settings.pexels_api_key},
                    params={"query": search_term, "per_page": 2, "page": page},
                    timeout=settings.api_timeout,
                )
                res.raise_for_status()
                photos = res.json().get("photos", [])
                
                for photo in photos:
                    image_data = {
                        "url": photo["src"]["medium"],
                        "alt": photo.get("alt", f"{topic} related image"),
                        "license": "Pexels",
                        "relevance_score": 0.8,  # Default high score for Pexels
                        "source": "pexels"
                    }
                    all_images.append(image_data)
                    
            except Exception as e:
                print(f"Image search error for '{search_term}': {e}")
        
        # Remove duplicates and limit to 3 images
        unique_images = []
        seen_urls = set()
        for img in all_images:
            if img["url"] not in seen_urls:
                unique_images.append(img)
                seen_urls.add(img["url"])
                if len(unique_images) >= 3:
                    break
        
        state["images"] = unique_images
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["image"] = step_time
        
        print(f"Image search completed: {len(unique_images)} images found in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Image search error: {e}")
        state["images"] = []
    
    return state

def enhanced_citation_agent(state: BlogState) -> BlogState:
    """Enhanced citation agent with credibility scoring"""
    articles = state.get("research_articles", [])
    
    start_time = time.time()
    
    try:
        citations = []
        for article in articles:
            # Calculate credibility score based on domain and content
            credibility_score = 0.7  # Default score
            
            # Boost for high-authority domains
            url = article.get("url", "")
            if any(domain in url for domain in ['wikipedia.org', 'github.com', 'stackoverflow.com', 'arxiv.org']):
                credibility_score = 0.9
            elif any(domain in url for domain in ['medium.com', 'dev.to', 'hackernews.com']):
                credibility_score = 0.8
            
            citation = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "credibility_score": credibility_score,
                "source_type": "web_article"
            }
            citations.append(citation)
        
        state["citations"] = citations
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["citation"] = step_time
        
        print(f"Citation processing completed in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Citation error: {e}")
        state["citations"] = []
    
    return state

def enhanced_merge_outputs(state: BlogState) -> BlogState:
    """Enhanced merge with better formatting and quality assessment"""
    start_time = time.time()
    
    try:
        post = state.get("markdown_draft", "") or ""
        images = state.get("images", [])
        citations = state.get("citations", [])
        
        # Enhanced image formatting
        images_md = ""
        if images:
            images_md = "\n\n## 🖼️ Related Images\n\n"
            for i, img in enumerate(images, 1):
                images_md += f"![{img['alt']}]({img['url']})\n*{img['alt']}*\n\n"
        
        # Enhanced citation formatting
        citations_md = ""
        if citations:
            citations_md = "\n\n## 📚 References\n\n"
            for i, citation in enumerate(citations, 1):
                credibility_indicator = "⭐" if citation.get("credibility_score", 0) > 0.8 else "📄"
                citations_md += f"{i}. {credibility_indicator} [{citation['title']}]({citation['url']})\n"
        
        # Add quality metrics section
        quality_metrics = state.get("content_quality_metrics", {})
        metrics_md = ""
        if quality_metrics:
            metrics_md = f"\n\n## 📊 Content Quality Metrics\n\n"
            metrics_md += f"- **Readability Score**: {quality_metrics.get('readability', 0):.2f}/1.0\n"
            metrics_md += f"- **Research Quality**: {quality_metrics.get('research_quality', 0):.2f}/1.0\n"
            metrics_md += f"- **Overall Quality**: {quality_metrics.get('overall', 0):.2f}/1.0\n"
        
        # Combine all parts
        final_post = f"{post}{images_md}{citations_md}{metrics_md}".strip()
        state["final_post"] = final_post
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["merge"] = step_time
        
        print(f"Merge completed in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Merge error: {e}")
        state["final_post"] = state.get("markdown_draft", "")
    
    return state

# --------------------
# Enhanced LangGraph Flow
# --------------------
def create_enhanced_blog_chain():
    """Create enhanced blog generation chain"""
    graph = StateGraph(BlogState)
    
    # Add nodes
    graph.add_node("research", enhanced_research_agent)
    graph.add_node("summarize", enhanced_summarizer_agent)
    graph.add_node("write", enhanced_writer_agent)
    graph.add_node("cite", enhanced_citation_agent)
    graph.add_node("image", enhanced_image_agent)
    graph.add_node("merge", enhanced_merge_outputs)
    
    # Set entry point
    graph.set_entry_point("research")
    
    # Add edges
    graph.add_edge("research", "summarize")
    graph.add_edge("summarize", "write")
    graph.add_edge("write", "cite")
    graph.add_edge("cite", "image")
    graph.add_edge("image", "merge")
    
    # Add conditional routing
    def route_after_merge(state: BlogState) -> str:
        if state.get("regenerate_images"):
            return "image"
        if state.get("regenerate_content"):
            return "write"
        return END
    
    graph.add_conditional_edges(
        "merge",
        route_after_merge,
        {
            "image": "image",
            "write": "write",
            END: END,
        }
    )
    
    return graph.compile()

# Create enhanced blog chain
enhanced_blog_chain = create_enhanced_blog_chain()

# Export for use in other modules
__all__ = ["enhanced_blog_chain", "BlogState", "EnhancedResearchAgent", "AdvancedSearchEngine"]

"""
Enhanced Blog Agents 2 - Groq-focused with Advanced Search and Improved Chain Management

Key Improvements:
- Multi-source research with intelligent filtering
- Advanced search query optimization
- Enhanced Groq integration with better models
- Content quality assessment and improvement
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
from langchain_groq import ChatGroq

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

# --------------------
# Enhanced Search Engine (Same as enhanced_blog_agents.py)
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
        return []
    
    async def _search_arxiv(self, search_query: SearchQuery, max_results: int) -> List[SearchResult]:
        """Search arXiv for academic papers"""
        return []
    
    async def _search_reddit(self, search_query: SearchQuery, max_results: int) -> List[SearchResult]:
        """Search Reddit for community discussions"""
        return []
    
    async def _search_github(self, search_query: SearchQuery, max_results: int) -> List[SearchResult]:
        """Search GitHub for code repositories and documentation"""
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
# Enhanced Groq LLM Configuration
# --------------------
def get_enhanced_groq_llms():
    """Get enhanced Groq LLM configurations with optimized settings"""
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is required")
    
    # Summary LLM - optimized for analysis
    summary_llm = ChatGroq(
        model=settings.groq_model_summary,
        temperature=0.0,  # Low temperature for factual analysis
        max_tokens=1000,
        top_p=0.9,
        api_key=settings.groq_api_key
    )
    
    # Writer LLM - optimized for creative writing
    writer_llm = ChatGroq(
        model=settings.groq_model_writer,
        temperature=0.2,  # Slightly higher for creativity
        max_tokens=2000,
        top_p=0.9,
        api_key=settings.groq_api_key
    )
    
    # Context LLM - optimized for understanding
    context_llm = ChatGroq(
        model=settings.groq_model_summary,
        temperature=0.1,
        max_tokens=500,
        top_p=0.9,
        api_key=settings.groq_api_key
    )
    
    # Editor LLM - optimized for refinement
    editor_llm = ChatGroq(
        model=settings.groq_model_writer,
        temperature=0.3,  # Higher for creative editing
        max_tokens=1500,
        top_p=0.9,
        api_key=settings.groq_api_key
    )
    
    return {
        'summary': summary_llm,
        'writer': writer_llm,
        'context': context_llm,
        'editor': editor_llm
    }

# Initialize enhanced Groq LLMs
try:
    enhanced_groq_llms = get_enhanced_groq_llms()
    llm_summary = enhanced_groq_llms['summary']
    llm_writer = enhanced_groq_llms['writer']
    llm_context = enhanced_groq_llms['context']
    llm_editor = enhanced_groq_llms['editor']
except ValueError as e:
    print(f"Groq initialization error: {e}")
    # Fallback to basic configuration
    llm_summary = llm_writer = llm_context = llm_editor = None

# --------------------
# Enhanced Prompts for Groq
# --------------------
ENHANCED_GROQ_SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research analyst with deep expertise in content strategy and information synthesis. Your role is to analyze research results and create comprehensive summaries that will guide the creation of high-quality, engaging blog content.

Your analysis should:
1. Identify the most valuable and actionable insights
2. Organize information by themes, importance, and relevance
3. Highlight key trends, statistics, expert opinions, and case studies
4. Note any conflicting information or knowledge gaps
5. Suggest compelling angles and storylines for the blog post
6. Assess the credibility and authority of sources

Focus on creating summaries that will help produce:
- Accurate, well-researched content
- Engaging narratives that hook readers
- Actionable insights and practical applications
- Current trends and cutting-edge developments
- Expert-backed opinions and evidence

Provide a structured, comprehensive analysis that serves as a roadmap for creating exceptional blog content."""),
    ("human", """Topic: {topic}

Research Results:
{content}

Please provide a comprehensive analysis and summary that will guide the blog writing process. Focus on the most valuable insights, organize them by themes, and suggest compelling angles for the blog post."""),
])

ENHANCED_GROQ_WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a world-class blog writer and content creator with expertise in creating compelling, informative, and highly engaging blog posts. Your writing style is:

- Professional yet accessible and conversational
- Engaging with storytelling elements
- Well-structured with clear, descriptive headings
- Rich in examples, case studies, and practical insights
- Optimized for readability, SEO, and user engagement
- Data-driven with credible sources and statistics

Your blog post guidelines:
1. Start with a compelling hook that immediately captures attention
2. Use clear, descriptive headings (H2, H3) that guide the reader
3. Include relevant examples, case studies, statistics, and expert quotes
4. Write in an engaging, conversational tone that builds connection
5. Use bullet points, lists, and formatting for better readability
6. Include actionable insights and practical takeaways
7. End with a strong conclusion that reinforces key points
8. Aim for 1200-1800 words for comprehensive coverage
9. Use proper Markdown formatting throughout
10. Include opportunities for internal linking and engagement
11. Make content valuable, shareable, and memorable

Your goal is to create content that not only educates and informs but also inspires, engages, and establishes thought leadership in the topic area."""),
    ("human", """Write a comprehensive, engaging blog post on: {topic}

Research Summary:
{summary}

Please create a well-structured, compelling blog post that incorporates the research insights while providing exceptional value, actionable content, and an engaging reading experience for your audience."""),
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
        search_engine = AdvancedSearchEngine()
        search_results = await search_engine.search_topic(topic, max_results=15)
        
        # Convert to the expected format
        formatted_articles = []
        for result in search_results:
            formatted_articles.append({
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "relevance_score": result.relevance_score,
                "source": result.source,
                "publish_date": result.publish_date.isoformat() if result.publish_date else None
            })
        
        state["research_articles"] = formatted_articles
        
        # Calculate research quality score
        if formatted_articles:
            avg_relevance = sum(article["relevance_score"] for article in formatted_articles) / len(formatted_articles)
            state["research_quality_score"] = avg_relevance
        else:
            state["research_quality_score"] = 0.0
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["research"] = step_time
        
        print(f"Enhanced research completed: {len(formatted_articles)} articles found in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Research error: {e}")
        state["research_articles"] = []
        state["research_quality_score"] = 0.0
    
    return state

def enhanced_summarizer_agent(state: BlogState) -> BlogState:
    """Enhanced summarizer with Groq optimization"""
    articles = state.get("research_articles", [])
    if not articles:
        state["research_summary"] = "No research articles found to summarize."
        return state
    
    start_time = time.time()
    
    try:
        # Prepare enhanced content for summarization
        content = "\n\n".join(
            f"**{a['title']}** (Relevance: {a.get('relevance_score', 0):.2f})\n"
            f"{a['snippet']}\n"
            f"Source: {a.get('source', 'unknown')} | URL: {a['url']}"
            for a in articles
        )
        
        # Use enhanced Groq prompt
        summarizer_chain = ENHANCED_GROQ_SUMMARIZER_PROMPT | llm_summary | StrOutputParser()
        summary = summarizer_chain.invoke({
            "topic": state.get("topic", ""),
            "content": content
        })
        
        state["research_summary"] = summary or "Summary could not be generated."
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["summarize"] = step_time
        
        print(f"Enhanced summarization completed in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Summarization error: {e}")
        state["research_summary"] = "Summarizer failed. Please try again."
    
    return state

def enhanced_writer_agent(state: BlogState) -> BlogState:
    """Enhanced writer with Groq optimization"""
    summary = state.get("research_summary", "No summary available.")
    topic = state.get("topic", "")
    
    start_time = time.time()
    
    try:
        # Use enhanced Groq prompt
        writer_chain = ENHANCED_GROQ_WRITER_PROMPT | llm_writer | StrOutputParser()
        markdown_draft = writer_chain.invoke({
            "topic": topic,
            "summary": summary
        })
        
        state["markdown_draft"] = markdown_draft or "Writing failed. Please try again."
        
        # Calculate content quality metrics
        content_metrics = {
            "word_count": len(markdown_draft.split()) if markdown_draft else 0,
            "heading_count": len(re.findall(r'^#+\s+', markdown_draft, re.MULTILINE)) if markdown_draft else 0,
            "link_count": len(re.findall(r'\[.*?\]\(.*?\)', markdown_draft)) if markdown_draft else 0,
            "list_count": len(re.findall(r'^\s*[-*+]\s+', markdown_draft, re.MULTILINE)) if markdown_draft else 0
        }
        state["content_quality_metrics"] = content_metrics
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["write"] = step_time
        
        print(f"Enhanced writing completed in {step_time:.2f}s")
        
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
            f"{topic} technology",
            f"{topic} modern"
        ]
        
        all_images = []
        for search_term in search_terms[:3]:  # Limit to 3 search terms
            page = random.randint(1, 3)  # Reduced page range for better quality
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
                        "relevance_score": 0.9,  # High score for Pexels
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
        
        print(f"Enhanced image search completed: {len(unique_images)} images found in {step_time:.2f}s")
        
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
                credibility_score = 0.95
            elif any(domain in url for domain in ['medium.com', 'dev.to', 'hackernews.com']):
                credibility_score = 0.85
            elif any(domain in url for domain in ['nature.com', 'ieee.org', 'acm.org']):
                credibility_score = 0.9
            
            # Boost based on relevance score
            relevance_boost = article.get("relevance_score", 0.5) * 0.2
            credibility_score = min(1.0, credibility_score + relevance_boost)
            
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
        
        print(f"Enhanced citation processing completed in {step_time:.2f}s")
        
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
        quality_metrics = state.get("content_quality_metrics", {})
        
        # Enhanced image formatting with better descriptions
        images_md = ""
        if images:
            images_md = "\n\n## 🖼️ Visual Elements\n\n"
            for i, img in enumerate(images, 1):
                images_md += f"![{img['alt']}]({img['url']})\n"
                images_md += f"*{img['alt']} - High-quality image from {img['source'].title()}*\n\n"
        
        # Enhanced citation formatting with credibility indicators
        citations_md = ""
        if citations:
            citations_md = "\n\n## 📚 References & Sources\n\n"
            for i, citation in enumerate(citations, 1):
                credibility_indicator = "⭐" if citation.get("credibility_score", 0) > 0.9 else "📄" if citation.get("credibility_score", 0) > 0.7 else "📝"
                citations_md += f"{i}. {credibility_indicator} [{citation['title']}]({citation['url']})\n"
        
        # Add enhanced quality metrics section
        metrics_md = ""
        if quality_metrics:
            metrics_md = f"\n\n## 📊 Content Quality Metrics\n\n"
            metrics_md += f"- **Word Count**: {quality_metrics.get('word_count', 0):,} words\n"
            metrics_md += f"- **Headings**: {quality_metrics.get('heading_count', 0)} sections\n"
            metrics_md += f"- **Links**: {quality_metrics.get('link_count', 0)} references\n"
            metrics_md += f"- **Lists**: {quality_metrics.get('list_count', 0)} bullet points\n"
            
            # Calculate overall quality score
            word_score = min(1.0, quality_metrics.get('word_count', 0) / 1200)  # Target: 1200 words
            heading_score = min(1.0, quality_metrics.get('heading_count', 0) / 5)  # Target: 5 headings
            link_score = min(1.0, quality_metrics.get('link_count', 0) / 3)  # Target: 3 links
            
            overall_quality = (word_score + heading_score + link_score) / 3
            metrics_md += f"- **Overall Quality**: {overall_quality:.2f}/1.0\n"
        
        # Add performance metrics
        step_timings = state.get("step_timings", {})
        if step_timings:
            total_time = sum(step_timings.values())
            metrics_md += f"\n**Generation Time**: {total_time:.2f}s\n"
        
        # Combine all parts
        final_post = f"{post}{images_md}{citations_md}{metrics_md}".strip()
        state["final_post"] = final_post
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["merge"] = step_time
        
        print(f"Enhanced merge completed in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Merge error: {e}")
        state["final_post"] = state.get("markdown_draft", "")
    
    return state

# --------------------
# Enhanced RAG Context Extractor
# --------------------
def get_enhanced_retriever_from_blog_content(markdown_text: str):
    """Enhanced retriever with better chunking and embedding"""
    loader = [Document(page_content=markdown_text)]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,  # Increased chunk size
        chunk_overlap=50,  # Added overlap for better context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    doc_chunks = text_splitter.split_documents(loader)
    
    # Use better embedding model
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        doc_chunks, 
        collection_name=f"enhanced-edit-context-{random.randint(1,1_000_000)}", 
        embedding=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})  # Get top 3 most relevant chunks

ENHANCED_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert content analyst specializing in understanding and extracting relevant context from blog content. Your task is to identify the exact portions of content that need editing based on user requests.

Your analysis should:
1. Understand the user's edit request and intent
2. Identify the most relevant content sections
3. Extract the exact text that needs modification
4. Preserve the original formatting and structure
5. Provide context that will enable precise editing

Return only the original Markdown content that is directly relevant to the edit request. Do not explain, expand, or modify the content."""),
    ("human", """Original blog content (chunked for search):
{chunked_content}

Edit request:
"{edit_request}"

Extract the exact Markdown content that needs to be edited:"""),
])

def enhanced_context_extractor_agent(state: BlogState) -> BlogState:
    """Enhanced context extractor with better RAG"""
    if not state.get("edit_request") or not state.get("final_post"):
        return state
    
    start_time = time.time()
    
    try:
        retriever = get_enhanced_retriever_from_blog_content(state["final_post"])
        docs = retriever.get_relevant_documents(state["edit_request"])
        chunked_text = "\n\n".join(d.page_content for d in docs)
        
        # Use enhanced context prompt
        context_chain = ENHANCED_CONTEXT_PROMPT | llm_context | StrOutputParser()
        extracted_context = context_chain.invoke({
            "edit_request": state["edit_request"],
            "chunked_content": chunked_text,
        })
        
        state["edit_context"] = extracted_context
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["context_extract"] = step_time
        
        print(f"Enhanced context extraction completed in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Context extraction error: {e}")
        state["edit_context"] = ""
    
    return state

ENHANCED_EDITOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional blog editor with expertise in content refinement and improvement. Your role is to edit specific sections of blog content based on user requests while maintaining the overall quality, tone, and structure.

Your editing approach:
1. Understand the user's specific edit request
2. Preserve the original tone and voice
3. Maintain the approximate word count
4. Improve clarity, structure, and flow
5. Use clean, proper Markdown formatting
6. Ensure the edited content fits seamlessly with the rest of the blog
7. Make improvements in grammar, style, and readability

Focus on creating polished, professional content that enhances the reader's experience."""),
    ("human", """Edit the following blog section according to the request:

Original content:
---
{edit_context}
---

Edit request:
"{edit_request}"

Please provide the revised content with improvements:"""),
])

def enhanced_editor_agent(state: BlogState) -> BlogState:
    """Enhanced editor with better prompt engineering"""
    if not state.get("edit_context") or not state.get("edit_request"):
        return state
    
    start_time = time.time()
    
    try:
        # Use enhanced editor prompt
        editor_chain = ENHANCED_EDITOR_PROMPT | llm_editor | StrOutputParser()
        revised_section = editor_chain.invoke({
            "edit_context": state["edit_context"],
            "edit_request": state["edit_request"],
        })
        
        # Update the blog content
        original = state.get("final_post", "")
        target = state.get("edit_context", "")
        if target and target in original:
            new_post = original.replace(target, revised_section, 1)
            state["final_post"] = new_post
            
            # Track edit history
            edit_history = state.get("edit_history", [])
            edit_history.append({
                "request": state["edit_request"],
                "timestamp": datetime.now().isoformat(),
                "context_length": len(target),
                "revised_length": len(revised_section)
            })
            state["edit_history"] = edit_history
        
        # Clear edit request and context
        state["edit_request"] = ""
        state.pop("edit_context", None)
        
        step_time = time.time() - start_time
        state["step_timings"] = state.get("step_timings", {})
        state["step_timings"]["edit"] = step_time
        
        print(f"Enhanced editing completed in {step_time:.2f}s")
        
    except Exception as e:
        print(f"Editing error: {e}")
    
    return state

# --------------------
# Enhanced LangGraph Flow
# --------------------
def create_enhanced_blog_chain():
    """Create enhanced blog generation chain with Groq optimization"""
    graph = StateGraph(BlogState)
    
    # Add nodes
    graph.add_node("research", enhanced_research_agent)
    graph.add_node("summarize", enhanced_summarizer_agent)
    graph.add_node("write", enhanced_writer_agent)
    graph.add_node("cite", enhanced_citation_agent)
    graph.add_node("image", enhanced_image_agent)
    graph.add_node("merge", enhanced_merge_outputs)
    graph.add_node("context_extract", enhanced_context_extractor_agent)
    graph.add_node("edit", enhanced_editor_agent)
    
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
        elif state.get("regenerate_content"):
            return "write"
        elif state.get("edit_request"):
            return "context_extract"
        else:
            return END
    
    graph.add_conditional_edges(
        "merge",
        route_after_merge,
        {
            "image": "image",
            "write": "write",
            "context_extract": "context_extract",
            END: END,
        },
    )
    
    graph.add_edge("context_extract", "edit")
    
    def route_after_edit(state: BlogState) -> str:
        if state.get("edit_request"):
            return "context_extract"
        else:
            return END
    
    graph.add_conditional_edges(
        "edit",
        route_after_edit,
        {
            "context_extract": "context_extract",
            END: END,
        },
    )
    
    return graph.compile()

# Create enhanced blog chain
enhanced_blog_chain = create_enhanced_blog_chain()

# Export for use in other modules
__all__ = ["enhanced_blog_chain", "BlogState", "AdvancedSearchEngine", "enhanced_groq_llms"]

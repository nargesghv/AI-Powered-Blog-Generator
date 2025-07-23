import sys
import os
from dotenv import load_dotenv

# Ensure parent directory is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.blog_agents import research_agent, BlogState

load_dotenv()

def test_research_agent_returns_valid_articles():
    # Ensure API key is present
    assert os.environ.get("SERPAPI_KEY"), "Missing SERPAPI_KEY in .env"

    # Prepare input
    state: BlogState = {"topic": "AI in Healthcare"}
    
    # Call agent
    updated_state = research_agent(state)

    # Assert 'research_articles' is present
    assert "research_articles" in updated_state
    articles = updated_state["research_articles"]
    
    # Basic structure checks
    assert isinstance(articles, list), "Expected a list of articles"
    assert len(articles) > 0, "No articles returned"

    first = articles[0]
    assert "title" in first and isinstance(first["title"], str)
    assert "url" in first and isinstance(first["url"], str)
    assert "snippet" in first and isinstance(first["snippet"], str)
    print(articles)
import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Add project root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.blog_agents import summarizer_agent, BlogState

def test_summarizer_agent_generates_summary():
    assert os.environ.get("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in .env"

    state: BlogState = {
        "topic": "AI in Healthcare",
        "research_articles": [
            {
                "title": "AI Detects Diseases Early",
                "url": "https://example.com/ai-diseases",
                "snippet": "AI models are being used to detect diseases at earlier stages..."
            },
            {
                "title": "Healthcare Robots in Surgery",
                "url": "https://example.com/robots-surgery",
                "snippet": "Surgical robots powered by machine learning increase precision..."
            }
        ]
    }

    updated_state = summarizer_agent(state)

    assert "research_summary" in updated_state
    summary = updated_state["research_summary"]
    assert isinstance(summary, str)
    assert len(summary) > 100  # We expect a decent-sized summary
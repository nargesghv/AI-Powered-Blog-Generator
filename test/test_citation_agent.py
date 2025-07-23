import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.blog_agents import citation_agent, BlogState

def test_citation_agent_formats_references():
    state: BlogState = {
        "research_articles": [
            {
                "title": "AI Diagnoses Faster",
                "url": "https://example.com/ai-diagnosis",
                "snippet": "AI helps detect diseases early."
            },
            {
                "title": "AI Surgery Robots",
                "url": "https://example.com/robot-surgeons",
                "snippet": "ML improves precision in surgical robots."
            }
        ]
    }

    updated_state = citation_agent(state)

    assert "citations" in updated_state
    citations = updated_state["citations"]

    assert isinstance(citations, list)
    assert len(citations) == 2
    assert citations[0].startswith("- [AI Diagnoses Faster](")
    assert citations[1].endswith(")")
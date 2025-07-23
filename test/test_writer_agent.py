import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.blog_agents import writer_agent, BlogState

def test_writer_agent_generates_markdown():
    assert os.environ.get("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in .env"

    state: BlogState = {
        "topic": "AI in Healthcare",
        "research_summary": """
        - AI is improving diagnostic speed and accuracy.
        - Healthcare robots are assisting in surgery.
        - AI chatbots are being used for patient communication and mental health triage.
        """
    }

    updated_state = writer_agent(state)

    assert "markdown_draft" in updated_state
    markdown = updated_state["markdown_draft"]

    assert isinstance(markdown, str)
    assert "# " in markdown or "## " in markdown, "Markdown headings expected"
    assert len(markdown) > 500, "Markdown is too short"
    print(markdown)

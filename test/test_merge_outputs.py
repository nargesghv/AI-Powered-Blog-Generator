import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.blog_agents import merge_outputs, BlogState

def test_merge_outputs_creates_final_post():
    state: BlogState = {
        "markdown_draft": "# AI and Medicine\nAI is helping healthcare in many ways...",
        "images": [
            {"url": "https://example.com/image1.jpg", "alt": "Doctor with AI", "license": "Pexels"}
        ],
        "citations": [
            "- [AI Research](https://example.com/ai-research)"
        ]
    }

    updated_state = merge_outputs(state)

    assert "final_post" in updated_state
    final = updated_state["final_post"]

    assert "# AI and Medicine" in final
    assert "![Doctor with AI](https://example.com/image1.jpg)" in final
    assert "## References" in final
    assert "- [AI Research](https://example.com/ai-research)" in final
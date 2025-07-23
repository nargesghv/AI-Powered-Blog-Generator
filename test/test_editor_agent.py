import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.blog_agents import editor_agent, BlogState

def test_editor_agent_applies_edit():
    state: BlogState = {
        "final_post": "# AI in Healthcare\nAI is useful.",
        "edit_request": "Change the title to 'How AI is Transforming Healthcare'"
    }

    updated_state = editor_agent(state)

    assert "final_post" in updated_state
    final = updated_state["final_post"]

    assert "How AI is Transforming Healthcare" in final
    assert "# AI in Healthcare" not in final
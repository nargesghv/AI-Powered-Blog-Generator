import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.blog_agents import image_agent, BlogState

def test_image_agent_returns_images():
    assert os.environ.get("PEXELS_API_KEY"), "Missing PEXELS_API_KEY in .env"

    state: BlogState = {
        "topic": "AI in Healthcare"
    }

    updated_state = image_agent(state)

    assert "images" in updated_state
    images = updated_state["images"]

    assert isinstance(images, list)
    assert len(images) > 0, "No images returned"

    first = images[0]
    assert "url" in first and isinstance(first["url"], str)
    assert "alt" in first and isinstance(first["alt"], str)
    assert "license" in first and first["license"] == "Pexels"
# test/test_main.py

from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import app


client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Backend is running!"}

def test_generate_blog():
    response = client.post("/generate", json={"topic": "How AI is transforming renewable energy"})
    assert response.status_code == 200
    data = response.json()
    assert "final_post" in data
    assert len(data["final_post"]) > 100
    assert "state" in data
    assert "research_summary" in data["state"]

def test_edit_blog():
    # Fake initial state for test
    initial_state = {
        "topic": "How AI is transforming renewable energy",
        "final_post": "Original post here...",
    }
    response = client.post("/edit", json={
        "state": initial_state,
        "edit_request": "Make it more concise and add a conclusion."
    })
    assert response.status_code == 200
    result = response.json()
    assert "final_post" in result
    assert "more concise" not in result["final_post"].lower()  # Post should be revised, not repeat instruction
def test_list_and_fetch_blogs():
    # First ensure a blog exists
    client.post("/generate", json={"topic": "Testing blog storage"})

    # Then fetch list
    list_resp = client.get("/blogs")
    assert list_resp.status_code == 200
    assert isinstance(list_resp.json(), list)
    assert len(list_resp.json()) > 0

    first_blog_id = list_resp.json()[0]["id"]
    get_resp = client.get(f"/blogs/{first_blog_id}")
    assert get_resp.status_code == 200
    assert "markdown" in get_resp.json()

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";

function App() {
  const [topic, setTopic] = useState("");
  const [model, setModel] = useState("gpt");
  const [response, setResponse] = useState(null);
  const [editRequest, setEditRequest] = useState("");
  const [loading, setLoading] = useState(false);
  const [wasEdited, setWasEdited] = useState(false);

  const generateBlog = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, model }),
      });
      const data = await res.json();
      setResponse(data);
      setWasEdited(false);
    } catch (err) {
      console.error("Error generating blog:", err);
    } finally {
      setLoading(false);
    }
  };

  const sendEdit = async () => {
    if (!editRequest.trim()) return;

    const safeState = {
      ...response.state,
      topic: response.state?.topic || topic,
    };

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/edit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          state: safeState,
          edit_request: editRequest,
          model,
        }),
      });
      const data = await res.json();
      setResponse(data);
      setEditRequest("");
      setWasEdited(true);
    } catch (err) {
      console.error("Error sending edit:", err);
    } finally {
      setLoading(false);
    }
  };

  const regenerateImages = async () => {
    if (!response?.state) return;
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/regenerate-images", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          state: response.state,
          final_post: response.final_post,
        }),
      });
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      console.error("Error regenerating images:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-2xl mx-auto bg-white p-6 rounded shadow">
        <h1 className="text-2xl font-bold mb-4">🧠 AI Blog Generator</h1>

        {/* Topic input */}
        <input
          className="w-full p-2 border rounded mb-4"
          placeholder="Enter a topic for the blog..."
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        />

        {/* Model Selector */}
        <div className="mb-4">
          <label className="font-medium mr-2">Model:</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="p-2 border rounded"
          >
            <option value="gpt">OpenAI GPT</option>
            <option value="llama">LLaMA (Groq)</option>
          </select>
        </div>

        <button
          onClick={generateBlog}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          Generate
        </button>

        {loading && (
          <p className="mt-4 text-sm text-gray-500">⏳ Processing request...</p>
        )}

        {/* Markdown preview */}
        {response?.final_post && (
          <div className="mt-6">
            <h2 className="text-lg font-semibold mb-2">
              {wasEdited ? "✍️ Edited Blog" : "📝 Generated Blog"}
            </h2>
            <div className="prose prose-lg max-w-none bg-gray-50 p-4 rounded">
              <ReactMarkdown>{response.final_post}</ReactMarkdown>
            </div>
          </div>
        )}

        {/* Edit section */}
        {response?.state && (
          <div className="mt-6">
            <textarea
              className="w-full p-2 border rounded h-20"
              placeholder="Request edits (e.g., make it funnier, shorten the intro)"
              value={editRequest}
              onChange={(e) => setEditRequest(e.target.value)}
            />
            <button
              onClick={sendEdit}
              className="bg-green-600 text-white px-4 py-2 rounded mt-2 hover:bg-green-700"
            >
              Send Edit Request
            </button>
          </div>
        )}

        {/* Image previews and regenerate button */}
        {response?.state?.images?.length > 0 && (
          <div className="mt-8">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold">🖼 Suggested Images</h3>
              <button
                onClick={regenerateImages}
                className="text-sm text-blue-600 hover:underline"
              >
                🔁 Regenerate Images
              </button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {response.state.images.map((img, idx) => (
                <div key={idx} className="border rounded overflow-hidden">
                  <img src={img.url} alt={img.alt} className="w-full h-auto" />
                  <div className="p-2 text-sm text-gray-600">
                    <p>
                      <strong>Alt:</strong> {img.alt}
                    </p>
                    <p>
                      <strong>License:</strong> {img.license}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

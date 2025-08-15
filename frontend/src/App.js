import React, { useState } from "react";
import ReactMarkdown from "react-markdown";

// Use env var or fallback to localhost
const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

function App() {
  const [topic, setTopic] = useState("");
  // ✅ Only two options: ollama or croq
  const [model, setModel] = useState("ollama");
  const [response, setResponse] = useState(null);
  const [editRequest, setEditRequest] = useState("");
  const [loading, setLoading] = useState(false);
  const [wasEdited, setWasEdited] = useState(false);
  const [error, setError] = useState("");

  const handleFetch = async (url, body) => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail || "Request failed");
      }

      setResponse(data);
      return data;
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong");
      return null;
    } finally {
      setLoading(false);
    }
  };

  const generateBlog = async () => {
    if (!topic.trim()) {
      setError("Please enter a topic.");
      return;
    }

    const payload = { topic, model }; // model: "ollama" | "croq"
    const data = await handleFetch(`${API_BASE}/generate`, payload);
    if (data) setWasEdited(false);
  };

  const sendEdit = async () => {
    if (!editRequest.trim() || !response?.state) return;

    const safeState = {
      ...response.state,
      topic: response.state?.topic || topic, // ensure topic exists
    };

    const payload = {
      state: safeState,
      edit_request: editRequest,
      model, // "ollama" | "croq"
    };

    const data = await handleFetch(`${API_BASE}/edit`, payload);
    if (data) {
      setEditRequest("");
      setWasEdited(true);
    }
  };

  const regenerateImages = async () => {
    if (!response?.state) return;

    const payload = {
      state: response.state,
      model, // ✅ include model so backend picks correct runner
    };

    await handleFetch(`${API_BASE}/regenerate-images`, payload);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-2xl mx-auto bg-white p-6 rounded shadow">
        <h1 className="text-2xl font-bold mb-4">🧠 AI Blog Generator</h1>

        <input
          className="w-full p-2 border rounded mb-4"
          placeholder="Enter a topic for the blog..."
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          disabled={loading}
        />

        <div className="mb-4">
          <label className="font-medium mr-2">Model:</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="p-2 border rounded"
            disabled={loading}
          >
            <option value="ollama">Ollama (local)</option>
            <option value="croq">Groq (LLaMA)</option>
          </select>
        </div>

        <button
          onClick={generateBlog}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-60"
          disabled={loading}
        >
          Generate
        </button>

        {loading && (
          <p className="mt-4 text-sm text-gray-500">⏳ Processing request...</p>
        )}

        {error && (
          <p className="mt-4 text-sm text-red-600">⚠️ {error}</p>
        )}

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

        {response?.state && (
          <div className="mt-6">
            <textarea
              className="w-full p-2 border rounded h-20"
              placeholder="Request edits (e.g., make it funnier, shorten the intro)"
              value={editRequest}
              onChange={(e) => setEditRequest(e.target.value)}
              disabled={loading}
            />
            <button
              onClick={sendEdit}
              className="bg-green-600 text-white px-4 py-2 rounded mt-2 hover:bg-green-700 disabled:opacity-60"
              disabled={loading || !editRequest.trim()}
            >
              Send Edit Request
            </button>
          </div>
        )}

        {response?.state?.images?.length > 0 && (
          <div className="mt-8">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold">🖼 Suggested Images</h3>
              <button
                onClick={regenerateImages}
                className="text-sm text-blue-600 hover:underline disabled:opacity-60"
                disabled={loading}
              >
                🔁 Regenerate Images
              </button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {response.state.images.map((img, idx) => (
                <div key={idx} className="border rounded overflow-hidden">
                  <img src={img.url} alt={img.alt || `Image ${idx + 1}`} className="w-full h-auto" />
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

        {response?.state?.citations?.length > 0 && (
          <div className="mt-8">
            <h3 className="font-semibold mb-2">🔗 References</h3>
            <ul className="list-disc list-inside text-sm text-gray-700">
              {response.state.citations.map((c, idx) => (
                <li key={idx}>
                  <a href={c.url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                    {c.title || c.url}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { motion } from "framer-motion";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8001";

// Stages shown in progress bar
const STAGES = ["Research", "Summarize", "Write", "Images", "Final"];

function App() {
  const [topic, setTopic] = useState("");
  const [model, setModel] = useState("ollama");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [currentStage, setCurrentStage] = useState("");

  const handleFetch = async (url, body) => {
    setLoading(true);
    setError("");
    setCurrentStage("Starting...");
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Request failed");
      setResponse(data);
      setCurrentStage("Final");
      return data;
    } catch (err) {
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
    setCurrentStage("Research");
    const payload = { topic, model };
    await handleFetch(`${API_BASE}/generate`, payload);
  };

  const regenerateImages = async () => {
    if (!response?.state) return;
    setCurrentStage("Images");
    const payload = { state: response.state, model };
    await handleFetch(`${API_BASE}/regenerate-images`, payload);
  };

  const stageIndex = STAGES.indexOf(currentStage);

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-100 to-purple-200 p-6">
      <div className="max-w-3xl mx-auto bg-white p-6 rounded-2xl shadow-xl">
        <h1 className="text-3xl font-bold mb-6 text-center">
          🧠 AI Blog Generator
        </h1>

        {/* Input Section */}
        <input
          className="w-full p-3 border rounded-lg mb-4"
          placeholder="Enter a topic..."
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
            <option value="groq">Groq (cloud)</option>
          </select>
        </div>

        <button
          onClick={generateBlog}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-60"
          disabled={loading}
        >
          Generate
        </button>

        {/* Progress Bar */}
        {loading && (
          <div className="mt-6">
            <div className="flex justify-between mb-2 text-sm font-medium">
              {STAGES.map((s, idx) => (
                <span
                  key={s}
                  className={idx <= stageIndex ? "text-blue-600" : "text-gray-400"}
                >
                  {s}
                </span>
              ))}
            </div>
            <div className="w-full bg-gray-200 h-2 rounded-full">
              <motion.div
                className="bg-blue-500 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${((stageIndex + 1) / STAGES.length) * 100}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
            <p className="mt-2 text-sm text-gray-500">
              ⏳ {currentStage}...
            </p>
          </div>
        )}

        {error && (
          <p className="mt-4 text-sm text-red-600">⚠️ {error}</p>
        )}

        {/* Blog Output */}
        {response?.final_post && (
          <div className="mt-6">
            <h2 className="text-lg font-semibold mb-2">📝 Generated Blog</h2>
            <div className="prose prose-lg max-w-none bg-gray-50 p-4 rounded-lg">
              <ReactMarkdown>{response.final_post}</ReactMarkdown>
            </div>
          </div>
        )}

        {/* Images */}
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
                <div key={idx} className="border rounded-lg overflow-hidden">
                  <img
                    src={img.url}
                    alt={img.alt || `Image ${idx + 1}`}
                    className="w-full h-auto"
                  />
                  <div className="p-2 text-sm text-gray-600">
                    <p><strong>Alt:</strong> {img.alt}</p>
                    <p><strong>License:</strong> {img.license}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Citations */}
        {response?.state?.citations?.length > 0 && (
          <div className="mt-8">
            <h3 className="font-semibold mb-2">🔗 References</h3>
            <ul className="list-disc list-inside text-sm text-gray-700">
              {response.state.citations.map((c, idx) => (
                <li key={idx}>
                  <a
                    href={c.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-blue-600 hover:underline"
                  >
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


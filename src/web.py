"""Web interface for Vault using FastAPI."""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import tempfile
import shutil

from .ingestion import DocumentLoader
from .vectorstore import VectorStore
from .rag import RAGEngine

app = FastAPI(title="Vault", description="Personal Knowledge Base powered by Parallax")

# Initialize components
vectorstore = VectorStore()
rag_engine = RAGEngine(vectorstore)


class QueryRequest(BaseModel):
    question: str
    stream: bool = True


class IngestRequest(BaseModel):
    path: str


# HTML Template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vault - Personal Knowledge Base</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e4e4e7;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem 0;
        }

        h1 {
            font-size: 3rem;
            background: linear-gradient(90deg, #06b6d4, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: #94a3b8;
            font-size: 1.1rem;
        }

        .badges {
            margin-top: 1rem;
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            flex-wrap: wrap;
        }

        .badge {
            display: inline-block;
            background: rgba(6, 182, 212, 0.2);
            border: 1px solid #06b6d4;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            color: #06b6d4;
        }

        .badge.privacy {
            background: rgba(34, 197, 94, 0.2);
            border-color: #22c55e;
            color: #22c55e;
        }

        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 1.5rem 0;
            padding: 1rem;
            background: rgba(255,255,255,0.05);
            border-radius: 0.75rem;
        }

        .stat { text-align: center; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #06b6d4; }
        .stat-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }

        .chat-container {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 1rem;
            overflow: hidden;
        }

        .chat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: rgba(0,0,0,0.2);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .chat-header .turn-info {
            font-size: 0.8rem;
            color: #94a3b8;
        }

        .chat-header button {
            padding: 0.4rem 0.8rem;
            font-size: 0.8rem;
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #ef4444;
            color: #ef4444;
        }

        .chat-header button:hover {
            background: rgba(239, 68, 68, 0.3);
        }

        .messages {
            height: 400px;
            overflow-y: auto;
            padding: 1.5rem;
        }

        .message {
            margin-bottom: 1rem;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-user {
            text-align: right;
        }

        .message-user .bubble {
            background: linear-gradient(135deg, #06b6d4, #8b5cf6);
            display: inline-block;
            padding: 0.75rem 1rem;
            border-radius: 1rem 1rem 0 1rem;
            max-width: 80%;
            text-align: left;
        }

        .message-assistant .bubble {
            background: rgba(255,255,255,0.08);
            display: inline-block;
            padding: 0.75rem 1rem;
            border-radius: 1rem 1rem 1rem 0;
            max-width: 80%;
            white-space: pre-wrap;
        }

        .input-area {
            display: flex;
            gap: 0.5rem;
            padding: 1rem;
            background: rgba(0,0,0,0.2);
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        input[type="text"] {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 0.5rem;
            background: rgba(255,255,255,0.05);
            color: #e4e4e7;
            font-size: 1rem;
        }

        input[type="text"]:focus {
            outline: none;
            border-color: #06b6d4;
        }

        button {
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #06b6d4, #8b5cf6);
            border: none;
            border-radius: 0.5rem;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }

        button:hover { transform: translateY(-1px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .upload-section {
            margin-top: 2rem;
            padding: 2rem;
            background: rgba(255,255,255,0.03);
            border: 2px dashed rgba(255,255,255,0.2);
            border-radius: 1rem;
            text-align: center;
        }

        .upload-section:hover {
            border-color: #06b6d4;
        }

        .sources {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: rgba(0,0,0,0.2);
            border-radius: 0.5rem;
            font-size: 0.75rem;
            color: #94a3b8;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #06b6d4;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 0.5rem;
        }

        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #06b6d4;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }

        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .footer {
            text-align: center;
            margin-top: 2rem;
            color: #64748b;
            font-size: 0.85rem;
        }

        .footer a {
            color: #06b6d4;
            text-decoration: none;
        }

        .memory-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: #22c55e;
        }

        .memory-indicator::before {
            content: '';
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Vault</h1>
            <p class="subtitle">Personal Knowledge Base powered by Parallax</p>
            <div class="badges">
                <span class="badge privacy">100% Local</span>
                <span class="badge privacy">No Cloud</span>
                <span class="badge">Multi-turn Chat</span>
            </div>
        </header>

        <div class="stats-bar">
            <div class="stat">
                <div class="stat-value" id="chunk-count">-</div>
                <div class="stat-label">Chunks</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="doc-count">-</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="parallax-status">-</div>
                <div class="stat-label">Parallax</div>
            </div>
        </div>

        <div class="chat-container">
            <div class="chat-header">
                <div class="memory-indicator">
                    <span>Conversation Memory Active</span>
                </div>
                <button onclick="clearConversation()">New Chat</button>
            </div>
            <div class="messages" id="messages">
                <div class="message message-assistant">
                    <div class="bubble">
                        Hello! I'm Vault, your personal knowledge base assistant powered by Parallax.

I can answer questions about your indexed documents and remember our conversation for follow-up questions.

Upload some documents below, then ask me anything!
                    </div>
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="question" placeholder="Ask a question about your documents..." />
                <button id="send-btn" onclick="sendQuestion()">Send</button>
            </div>
        </div>

        <div class="upload-section">
            <h3>Add Documents</h3>
            <p style="margin: 1rem 0; color: #94a3b8;">Supports PDF, DOCX, Markdown, Text, Email (.eml), CSV, and JSON</p>
            <input type="file" id="file-input" multiple accept=".pdf,.md,.txt,.docx,.eml,.csv,.json" style="display:none" />
            <button onclick="document.getElementById('file-input').click()">Choose Files</button>
            <div id="upload-status" style="margin-top: 1rem;"></div>
        </div>

        <div class="footer">
            <p>Built with <a href="https://github.com/GradientHQ/parallax" target="_blank">Parallax</a> for the <a href="https://gradient.network/campaign/" target="_blank">Gradient Network Campaign</a></p>
            <p style="margin-top: 0.5rem; font-size: 0.75rem;">Your data never leaves your machine</p>
        </div>
    </div>

    <script>
        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('chunk-count').textContent = data.chunks;
                document.getElementById('doc-count').textContent = data.documents;
                document.getElementById('parallax-status').textContent = data.parallax_available ? 'Online' : 'Offline';
                document.getElementById('parallax-status').style.color = data.parallax_available ? '#22c55e' : '#ef4444';
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }

        async function clearConversation() {
            try {
                await fetch('/api/clear-history', { method: 'POST' });
                document.getElementById('messages').innerHTML = `
                    <div class="message message-assistant">
                        <div class="bubble">
                            Conversation cleared! I'm ready for a fresh start. What would you like to know about your documents?
                        </div>
                    </div>
                `;
            } catch (e) {
                console.error('Failed to clear history:', e);
            }
        }

        async function sendQuestion() {
            const input = document.getElementById('question');
            const question = input.value.trim();
            if (!question) return;

            input.value = '';
            const messages = document.getElementById('messages');

            // Add user message
            messages.innerHTML += `
                <div class="message message-user">
                    <div class="bubble">${escapeHtml(question)}</div>
                </div>
            `;

            // Add typing indicator
            const loadingId = 'loading-' + Date.now();
            messages.innerHTML += `
                <div class="message message-assistant" id="${loadingId}">
                    <div class="bubble">
                        <div class="typing-indicator">
                            <span></span><span></span><span></span>
                        </div>
                    </div>
                </div>
            `;
            messages.scrollTop = messages.scrollHeight;

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, stream: false })
                });

                const data = await response.json();
                document.getElementById(loadingId).remove();

                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    const sourceNames = [...new Set(data.sources.map(s => s.source.split('/').pop()))].slice(0, 3);
                    sourcesHtml = `<div class="sources">Sources: ${escapeHtml(sourceNames.join(', '))}</div>`;
                }

                // Format the answer (preserve newlines)
                const formattedAnswer = escapeHtml(data.answer);

                messages.innerHTML += `
                    <div class="message message-assistant">
                        <div class="bubble">${formattedAnswer}</div>
                        ${sourcesHtml}
                    </div>
                `;

            } catch (e) {
                document.getElementById(loadingId).remove();
                messages.innerHTML += `
                    <div class="message message-assistant">
                        <div class="bubble" style="border: 1px solid #ef4444;">Error: ${escapeHtml(e.message)}</div>
                    </div>
                `;
            }

            messages.scrollTop = messages.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        document.getElementById('question').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendQuestion();
        });

        document.getElementById('file-input').addEventListener('change', async (e) => {
            const files = e.target.files;
            if (!files.length) return;

            const status = document.getElementById('upload-status');
            status.innerHTML = '<div class="loading"></div> Uploading and indexing...';

            let successCount = 0;
            for (const file of files) {
                const formData = new FormData();
                formData.append('file', file);

                try {
                    const res = await fetch('/api/upload', { method: 'POST', body: formData });
                    if (res.ok) successCount++;
                } catch (err) {
                    console.error('Upload failed:', err);
                }
            }

            status.innerHTML = `<span style="color: #22c55e;">Uploaded ${successCount} file(s) successfully!</span>`;
            loadStats();

            // Add message about new documents
            if (successCount > 0) {
                const messages = document.getElementById('messages');
                messages.innerHTML += `
                    <div class="message message-assistant">
                        <div class="bubble" style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3);">
                            ${successCount} new document(s) indexed! You can now ask questions about them.
                        </div>
                    </div>
                `;
                messages.scrollTop = messages.scrollHeight;
            }

            setTimeout(() => { status.textContent = ''; }, 5000);
        });

        loadStats();
        setInterval(loadStats, 30000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface."""
    return HTML_TEMPLATE


@app.get("/api/stats")
async def get_stats():
    """Get knowledge base statistics."""
    stats = vectorstore.get_stats()
    sources = vectorstore.list_sources()
    return {
        "chunks": stats["total_chunks"],
        "documents": len(sources),
        "parallax_available": rag_engine.parallax.is_available()
    }


@app.post("/api/query")
async def query(request: QueryRequest):
    """Query the knowledge base with conversation memory."""
    sources = rag_engine.get_sources_for_query(request.question)

    if not rag_engine.parallax.is_available():
        # Return just the sources if Parallax is not available
        context = "\n\n".join([s["content"] for s in sources[:3]])
        return {
            "answer": f"Parallax is not available. Here's the relevant context:\n\n{context}",
            "sources": sources
        }

    try:
        answer = rag_engine.query(request.question, stream=False, use_history=True)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear-history")
async def clear_history():
    """Clear conversation history."""
    rag_engine.clear_history()
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and ingest a document."""
    # Check file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in DocumentLoader.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {DocumentLoader.SUPPORTED_EXTENSIONS}"
        )

    # Save to temp file and process
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        loader = DocumentLoader()
        chunks = list(loader.process_file(tmp_path))

        # Update source path to use original filename
        for chunk in chunks:
            chunk.source = file.filename

        count = vectorstore.add_chunks(chunks)
        return {"filename": file.filename, "chunks_added": count}
    finally:
        os.unlink(tmp_path)


@app.get("/api/sources")
async def list_sources():
    """List all indexed sources."""
    return {"sources": vectorstore.list_sources()}


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the web server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

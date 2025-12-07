"""RAG engine using Parallax for local inference."""

import httpx
from typing import Optional, Generator
from dataclasses import dataclass, field
import json

from .vectorstore import VectorStore


class ParallaxClient:
    """Client for Parallax inference API."""

    def __init__(self, base_url: str = "http://localhost:3001"):
        self.base_url = base_url
        self.api_url = f"{base_url}/v1/chat/completions"

    def is_available(self) -> bool:
        """Check if Parallax server is running."""
        try:
            with httpx.Client(timeout=30.0) as client:
                # Try a minimal chat request - this works reliably
                response = client.post(
                    self.api_url,
                    json={
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "stream": False
                    }
                )
                return response.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Generator[str, None, None] | str:
        """Send a chat completion request to Parallax."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }

        if stream:
            return self._stream_chat(payload)
        else:
            return self._sync_chat(payload)

    def _sync_chat(self, payload: dict) -> str:
        """Non-streaming chat completion."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']

    def _stream_chat(self, payload: dict) -> Generator[str, None, None]:
        """Streaming chat completion."""
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", self.api_url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


@dataclass
class ChatMessage:
    """A message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    sources: list[dict] = field(default_factory=list)


@dataclass
class ChatSession:
    """Maintains conversation history for multi-turn chat."""
    messages: list[ChatMessage] = field(default_factory=list)
    max_history: int = 10  # Keep last N exchanges

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(ChatMessage(role="user", content=content))
        self._trim_history()

    def add_assistant_message(self, content: str, sources: list[dict] = None):
        """Add an assistant response to history."""
        self.messages.append(ChatMessage(
            role="assistant",
            content=content,
            sources=sources or []
        ))
        self._trim_history()

    def _trim_history(self):
        """Keep only the last max_history messages."""
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]

    def get_history_summary(self) -> str:
        """Get a summary of recent conversation for context."""
        if not self.messages:
            return ""

        summary_parts = []
        for msg in self.messages[-6:]:  # Last 3 exchanges
            prefix = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            summary_parts.append(f"{prefix}: {content}")

        return "\n".join(summary_parts)

    def clear(self):
        """Clear conversation history."""
        self.messages = []


class RAGEngine:
    """Retrieval-Augmented Generation engine with conversation memory."""

    SYSTEM_PROMPT = """You are Vault, a helpful AI assistant with access to a personal knowledge base.
Your role is to answer questions based on the provided context from the user's documents.

Guidelines:
- Answer based primarily on the provided context from the knowledge base
- If the context doesn't contain enough information, say so clearly
- Cite the source documents when relevant (e.g., "According to [document name]...")
- Be concise but thorough
- You have access to conversation history - use it to understand follow-up questions
- If asked about something not in the context, you can provide general knowledge but clarify it's not from the knowledge base
- When referencing previous answers, be consistent with what you said before"""

    def __init__(
        self,
        vectorstore: VectorStore,
        parallax_url: str = "http://localhost:3001",
        n_results: int = 5
    ):
        self.vectorstore = vectorstore
        self.parallax = ParallaxClient(parallax_url)
        self.n_results = n_results
        self.session = ChatSession()

    def build_context(self, query: str) -> tuple[str, list[dict]]:
        """Retrieve relevant context for a query."""
        results = self.vectorstore.search(query, n_results=self.n_results)

        if not results:
            return "", []

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['source'].split('/')[-1]  # Just filename
            context_parts.append(
                f"[{i}] Source: {source}\n{result['content']}"
            )

        context = "\n\n---\n\n".join(context_parts)
        return context, results

    def _build_prompt(self, question: str, use_history: bool) -> tuple[str | None, list[dict]]:
        """Build the prompt with context and history."""
        context, sources = self.build_context(question)

        if not context:
            return None, sources

        history_summary = self.session.get_history_summary() if use_history else ""

        if history_summary and len(self.session.messages) > 1:
            user_message = f"""Previous conversation:
{history_summary}

---

Context from knowledge base:

{context}

---

Current question: {question}

Please answer based on the context above. If this is a follow-up question, use the conversation history to understand the context."""
        else:
            user_message = f"""Context from knowledge base:

{context}

---

Question: {question}

Please answer based on the context above."""

        return user_message, sources

    def query(
        self,
        question: str,
        stream: bool = True,
        max_tokens: int = 1024,
        use_history: bool = True
    ) -> Generator[str, None, None] | str:
        """Query the knowledge base with conversation memory."""
        if stream:
            return self._query_stream(question, max_tokens, use_history)
        else:
            return self._query_sync(question, max_tokens, use_history)

    def _query_sync(self, question: str, max_tokens: int, use_history: bool) -> str:
        """Non-streaming query."""
        if use_history:
            self.session.add_user_message(question)

        user_message, sources = self._build_prompt(question, use_history)

        if user_message is None:
            no_context_msg = "I couldn't find relevant documents in your knowledge base for this query. Try rephrasing or make sure you've indexed relevant documents."
            if use_history:
                self.session.add_assistant_message(no_context_msg, [])
            return no_context_msg

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        response = self.parallax.chat(messages, max_tokens=max_tokens, stream=False)
        if use_history:
            self.session.add_assistant_message(response, sources)
        return response

    def _query_stream(self, question: str, max_tokens: int, use_history: bool) -> Generator[str, None, None]:
        """Streaming query."""
        if use_history:
            self.session.add_user_message(question)

        user_message, sources = self._build_prompt(question, use_history)

        if user_message is None:
            no_context_msg = "I couldn't find relevant documents in your knowledge base for this query. Try rephrasing or make sure you've indexed relevant documents."
            if use_history:
                self.session.add_assistant_message(no_context_msg, [])
            yield no_context_msg
            return

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        response_chunks = []
        for chunk in self.parallax.chat(messages, max_tokens=max_tokens, stream=True):
            response_chunks.append(chunk)
            yield chunk

        if use_history:
            self.session.add_assistant_message("".join(response_chunks), sources)

    def get_sources_for_query(self, question: str) -> list[dict]:
        """Get source documents for a query without generating a response."""
        _, sources = self.build_context(question)
        return sources

    def clear_history(self):
        """Clear conversation history."""
        self.session.clear()

    def get_history(self) -> list[ChatMessage]:
        """Get conversation history."""
        return self.session.messages

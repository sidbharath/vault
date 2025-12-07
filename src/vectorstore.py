"""Vector storage using ChromaDB."""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from .ingestion import Chunk


class VectorStore:
    """ChromaDB-based vector store for document chunks."""

    def __init__(self, persist_dir: str = "./data/chromadb"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection with default embedding function
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add document chunks to the vector store."""
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {"source": chunk.source, **chunk.metadata}
            for chunk in chunks
        ]

        # Upsert to handle duplicates
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Search for similar chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        matches = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                matches.append({
                    "content": doc,
                    "source": results['metadatas'][0][i].get('source', 'Unknown'),
                    "distance": results['distances'][0][i] if results['distances'] else None,
                    "metadata": results['metadatas'][0][i]
                })

        return matches

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "total_chunks": self.collection.count(),
            "persist_dir": str(self.persist_dir)
        }

    def list_sources(self) -> list[str]:
        """List all unique sources in the collection."""
        # Get all metadatas
        result = self.collection.get(include=["metadatas"])
        sources = set()
        if result and result['metadatas']:
            for meta in result['metadatas']:
                if meta and 'source' in meta:
                    sources.add(meta['source'])
        return sorted(sources)

    def delete_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        # Find chunks with this source
        result = self.collection.get(
            where={"source": source},
            include=["metadatas"]
        )

        if result and result['ids']:
            self.collection.delete(ids=result['ids'])
            return len(result['ids'])
        return 0

    def clear(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection("knowledge_base")
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

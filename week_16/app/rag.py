import json
import os
import re
from typing import Any


class SimpleRAG:
    def __init__(self, documents: list[dict[str, Any]] | None = None, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = documents or []
        self.persist_dir = os.path.join(os.getcwd(), "data", "vector_store")
        os.makedirs(self.persist_dir, exist_ok=True)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        for document in documents:
            text = str(document.get("text", "")).strip()
            metadata = document.get("metadata", {})
            if not text:
                continue
            for index, chunk in enumerate(self._chunk_text(text)):
                self.documents.append({
                    "id": document.get("id", f"doc-{len(self.documents)}"),
                    "text": chunk,
                    "metadata": {**metadata, "chunk_index": index},
                })

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + 500)
            chunk = cleaned[start:end].strip()
            if len(chunk) < 80 and start != 0:
                chunks[-1] = f"{chunks[-1]} {chunk}".strip()
                break
            chunks.append(chunk)
            if end >= len(cleaned):
                break
            start = max(0, end - 80)
        return chunks

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.documents or not str(query).strip():
            return []
        query_tokens = self._tokens(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for document in self.documents:
            score = self._similarity(query_tokens, document["text"])
            scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{
            "id": document.get("id", "unknown"),
            "text": document["text"],
            "metadata": document.get("metadata", {}),
            "score": round(score, 4),
        } for score, document in scored[:max(1, top_k)] if score > 0]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
            "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
            "to", "was", "were", "what", "when", "where", "which", "who", "with", "why",
            "how", "can", "could", "should", "would", "do", "does", "did", "about",
            "you", "your", "we", "our", "they", "them", "there", "then"
        }
        return {
            token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text)
            if token.lower() not in stopwords and len(token) > 2
        }

    @staticmethod
    def _similarity(query_tokens: set[str], text: str) -> float:
        text_tokens = SimpleRAG._tokens(text)
        overlap = len(query_tokens & text_tokens)
        if not overlap:
            return 0.0
        return overlap / max(len(query_tokens), 1)

    def save(self) -> None:
        path = os.path.join(self.persist_dir, "rag_store.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.documents, handle)

    def load(self) -> None:
        path = os.path.join(self.persist_dir, "rag_store.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            self.documents = json.load(handle)

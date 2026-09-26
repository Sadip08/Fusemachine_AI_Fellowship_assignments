from __future__ import annotations

import re
from typing import Any


class ToolValidationError(ValueError):
    """Raised when a tool input is invalid."""


def validate_query(query: str) -> str:
    query = (query or "").strip()
    if not query:
        raise ToolValidationError("query cannot be empty")
    if len(query) > 1000:
        raise ToolValidationError("query is too long")
    return query


def search_documents(rag: Any, query: str, top_k: int = 3) -> dict[str, Any]:
    cleaned_query = validate_query(query)
    if top_k <= 0:
        raise ToolValidationError("top_k must be greater than zero")
    results = rag.search(cleaned_query, top_k=top_k)
    return {
        "status": "ok",
        "query": cleaned_query,
        "hits": results,
        "total_hits": len(results),
    }


def evaluate_evidence(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    cleaned_query = validate_query(query)
    lowered = cleaned_query.lower()
    if not results:
        return {
            "sufficient": False,
            "coverage": 0.0,
            "reason": "No relevant evidence found.",
            "missing_information": ["Document retrieval did not return evidence for this question."],
            "supporting_evidence": [],
        }

    text = " ".join(str(item.get("text", "")) for item in results)
    query_tokens = {
        token.lower() for token in re.findall(r"[a-zA-Z0-9]+", cleaned_query)
        if token.lower() not in {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "what", "when", "where", "which", "who", "with", "why", "how", "can", "could", "should", "would", "do", "does", "did", "about", "you", "your", "we", "our", "they", "them", "there", "then"} and len(token) > 2
    }
    evidence_tokens = {
        token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text)
        if token.lower() not in {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "what", "when", "where", "which", "who", "with", "why", "how", "can", "could", "should", "would", "do", "does", "did", "about", "you", "your", "we", "our", "they", "them", "there", "then"} and len(token) > 2
    }
    overlap = len(query_tokens & evidence_tokens)
    coverage = overlap / max(len(query_tokens), 1)
    missing = []

    generic_markers = ("what is the", "what are", "what is", "support setup", "general information")
    if any(marker in lowered for marker in generic_markers) and len(query_tokens) <= 4:
        missing.append("The question is too generic to answer confidently without more specific details.")

    if coverage < 0.2:
        missing.append("The retrieved evidence only weakly matches the query terms.")
    if len(results) < 2:
        missing.append("Need at least two supporting snippets before answering confidently.")
    evidence = [
        {
            "id": item.get("id", "unknown"),
            "text": item.get("text", ""),
            "score": item.get("score", 0.0),
        }
        for item in results[:3]
    ]
    return {
        "sufficient": coverage >= 0.2 and len(results) >= 2 and len(missing) == 0,
        "coverage": round(coverage, 4),
        "reason": "Evidence matches the request sufficiently." if len(missing) == 0 else "More evidence is required.",
        "missing_information": missing,
        "supporting_evidence": evidence,
    }


def summarize_evidence(results: list[dict[str, Any]]) -> list[str]:
    if not results:
        return []
    top_items: list[str] = []
    for item in results[:3]:
        text = str(item.get("text", "")).strip()
        if text:
            top_items.append(text[:220])
    return top_items

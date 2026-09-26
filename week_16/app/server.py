import asyncio
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.agent import AgentRuntime
from app.cache import ResponseCache
from app.config import settings
from app.rag import SimpleRAG

app = FastAPI(title="Agentified W16 Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SimpleRAG()
response_cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
agent_runtime = AgentRuntime(rag, max_iterations=settings.max_iterations)
REQUEST_Timestamps: dict[str, list[float]] = {}


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def check(self, key: str) -> bool:
        now = time.time()
        bucket = REQUEST_Timestamps.setdefault(key, [])
        bucket[:] = [stamp for stamp in bucket if now - stamp < self.window_seconds]
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True


rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_per_minute,
    window_seconds=60,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "max_iterations": settings.max_iterations,
        "provider": "agentic-rag",
    }


@app.post("/api/ingest")
def ingest_documents(payload: dict[str, Any]) -> dict[str, Any]:
    documents = payload.get("documents", [])
    if not documents:
        raise HTTPException(status_code=400, detail="documents list cannot be empty")
    rag.add_documents(documents)
    rag.save()
    return {"status": "ok", "documents_ingested": len(documents), "total_chunks": len(rag.documents)}


@app.post("/api/search")
def search_documents(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    results = rag.search(query, top_k=int(payload.get("top_k", 3)))
    return {"results": results}


@app.post("/api/chat")
async def chat(request: Request) -> dict[str, Any]:
    body = await request.json()
    question = body.get("message", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="message is required")

    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down your requests.")

    cache_key = f"chat:{question.lower()}"
    cached = response_cache.get(cache_key)
    if cached:
        return {"answer": cached["answer"], "status": cached["status"], "cached": True, "tool_history": cached.get("tool_history", [])}

    state = agent_runtime.run(question)
    payload = {
        "answer": state.final_answer,
        "status": state.status,
        "cached": False,
        "tool_history": state.tool_history,
        "retrieval_results": state.retrieval_results,
        "evidence_sufficiency": state.evidence_sufficiency,
        "failure_state": state.failure_state,
    }
    response_cache.set(cache_key, payload)
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host=settings.api_host, port=settings.api_port, reload=False)

import time
from typing import Any


class ResponseCache:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, dict[str, Any]]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        item = self._store.get(key)
        if item is None:
            return None
        timestamp, payload = item
        if time.time() - timestamp > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        return payload

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._store[key] = (time.time(), value)

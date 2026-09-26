import json
from typing import Any

import requests
from openai import OpenAI

from app.config import settings


class LLMClient:
    def __init__(self) -> None:
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.local_base_url = settings.local_llm_base_url
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def generate(self, prompt: str, system_prompt: str = "You are a helpful production assistant.") -> str:
        for attempt in range(3):
            try:
                if self.client is not None:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                        top_p=0.9,
                    )
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        self.last_usage = {
                            "input_tokens": getattr(usage, "prompt_tokens", 0),
                            "output_tokens": getattr(usage, "completion_tokens", 0),
                            "total_tokens": getattr(usage, "total_tokens", 0),
                        }
                    return response.choices[0].message.content or "I could not generate a response."

                if self.local_base_url:
                    payload = {
                        "model": settings.local_llm_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "top_p": 0.9,
                    }
                    response = requests.post(
                        f"{self.local_base_url.rstrip('/')}/v1/chat/completions",
                        json=payload,
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage")
                    if usage is not None:
                        self.last_usage = {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                    return data["choices"][0]["message"]["content"] or "I could not generate a response."

                return self._fallback_response(prompt)
            except Exception as exc:  # pragma: no cover - runtime fallback path
                if attempt == 2:
                    return self._fallback_response(prompt, error=str(exc))
        return self._fallback_response(prompt)

    @staticmethod
    def _fallback_response(prompt: str, error: str | None = None) -> str:
        base = (
            "I am running in degraded mode because no primary model provider is available. "
            "The assistant can still answer from retrieved context and operational guidance. "
        )
        if error:
            base += f"Last error: {error}."
        return base + f"User request: {prompt[:200]}"

    @property
    def usage(self) -> dict[str, int]:
        return getattr(self, "last_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

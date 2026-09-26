import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    backend_url: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    local_llm_base_url: str | None = os.getenv("LOCAL_LLM_BASE_URL")
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "5"))


settings = Settings()

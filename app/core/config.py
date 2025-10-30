import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def getenv_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def getenv_int(key: str, default: int) -> int:
    v = os.getenv(key)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


@dataclass(frozen=True)
class Config:
    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # Index paths
    FAISS_DIR: str = os.getenv("FAISS_DIR", "./data/indexes/faiss")
    BM25_DIR: str = os.getenv("BM25_DIR", "./data/indexes/bm25")

    # Hybrid flags
    ENABLE_BM25: bool = getenv_bool("ENABLE_BM25", True)
    ENABLE_RRF: bool = getenv_bool("ENABLE_RRF", True)
    ENABLE_RERANKER: bool = getenv_bool("ENABLE_RERANKER", False)  # 기본 OFF (라이트)

    # Retrieval knobs
    TOP_K: int = getenv_int("TOP_K", 4)              # 최종 컨텍스트 개수(3~5 권장)
    CANDIDATE_K: int = getenv_int("CANDIDATE_K", 6)  # 각 리트리버 후보(5~8 권장)
    RRF_K: int = getenv_int("RRF_K", 60)             # 50~100

    # Server
    APP_ENV: str = os.getenv("APP_ENV", "dev")
    APP_PORT: int = getenv_int("APP_PORT", 8000)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")


cfg = Config()
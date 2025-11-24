# app/rag/retrievers/hybrid_rrf_redis.py
"""
Redis 캐싱을 사용하는 하이브리드 검색
"""
import os
import hashlib
from typing import List
from langchain_core.documents import Document
from app.rag.retrievers.hybrid_rrf import HybridRRF
from app.core.cache_utils_redis import get_cache, set_cache

# 검색 캐시 설정
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "3600"))  # 1시간


def _cache_key(query: str, top_k: int) -> str:
    """캐시 키 생성"""
    return hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()


class RedisHybridRRF(HybridRRF):
    """Redis 캐싱이 추가된 하이브리드 검색"""

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Redis 캐싱된 검색"""
        key = _cache_key(query, self.top_k)

        # Redis 캐시 확인
        cached_docs = get_cache(key, prefix="search")
        if cached_docs is not None:
            return cached_docs

        # 캐시 미스 - 실제 검색
        docs = super()._get_relevant_documents(query)

        # Redis에 저장
        set_cache(key, docs, ttl=SEARCH_CACHE_TTL, prefix="search")

        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """비동기 버전도 캐싱"""
        return self._get_relevant_documents(query)


def load_redis_hybrid_from_env(**kwargs) -> RedisHybridRRF:
    """Redis 캐싱 버전 하이브리드 로더"""
    from app.rag.retrievers.hybrid_rrf import load_hybrid_from_env

    # 기본 하이브리드 로드
    base = load_hybrid_from_env(**kwargs)

    # Redis 캐싱 버전으로 변환
    return RedisHybridRRF(
        faiss=base.faiss,
        bm25_stores=base.bm25_stores,
        top_k=base.top_k,
        candidate_k=base.candidate_k,
        rrf_k=base.rrf_k,
        w_faiss=base.w_faiss,
        w_bm25=base.w_bm25,
    )
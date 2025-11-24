# app/api/routes_cache.py
"""
캐시 관리 및 모니터링 API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

try:
    from app.core.cache_utils import get_cache_stats, clear_cache, get_cache, set_cache

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()


class CacheTestRequest(BaseModel):
    key: str
    value: str
    ttl: Optional[int] = 300
    prefix: Optional[str] = "test"


@router.get("/stats")
def get_stats():
    """
    캐시 통계 조회

    Returns:
        - redis_available: Redis 사용 가능 여부
        - redis_keys: 저장된 키 개수
        - redis_hits: 캐시 히트 수
        - redis_misses: 캐시 미스 수
        - hit_rate: 히트율
        - redis_memory: 사용 중인 메모리
    """
    if not CACHE_AVAILABLE:
        raise HTTPException(status_code=503, detail="캐시 시스템 사용 불가")

    try:
        stats = get_cache_stats()
        return {
            "status": "ok",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"캐시 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear/{prefix}")
def clear_cache_by_prefix(prefix: str):
    """
    특정 프리픽스의 캐시 삭제

    Args:
        prefix: 캐시 프리픽스 (search, moderation, quiz 등)

    Examples:
        POST /api/cache/clear/search     # 검색 캐시 삭제
        POST /api/cache/clear/moderation # 모더레이션 캐시 삭제
    """
    if not CACHE_AVAILABLE:
        raise HTTPException(status_code=503, detail="캐시 시스템 사용 불가")

    try:
        clear_cache(prefix)
        logger.info(f"캐시 삭제됨: {prefix}")
        return {
            "status": "ok",
            "message": f"{prefix} 캐시가 삭제되었습니다"
        }
    except Exception as e:
        logger.error(f"캐시 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
def test_cache(request: CacheTestRequest):
    """
    캐시 동작 테스트

    키-값 쌍을 저장하고 다시 읽어서 확인
    """
    if not CACHE_AVAILABLE:
        raise HTTPException(status_code=503, detail="캐시 시스템 사용 불가")

    try:
        # 캐시에 저장
        set_cache(request.key, request.value, ttl=request.ttl, prefix=request.prefix)

        # 캐시에서 읽기
        cached_value = get_cache(request.key, prefix=request.prefix)

        if cached_value == request.value:
            return {
                "status": "ok",
                "message": "캐시 동작 정상",
                "key": request.key,
                "value": cached_value,
                "ttl": request.ttl
            }
        else:
            return {
                "status": "error",
                "message": "캐시 값 불일치",
                "expected": request.value,
                "actual": cached_value
            }

    except Exception as e:
        logger.error(f"캐시 테스트 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def cache_health():
    """
    캐시 시스템 헬스체크
    """
    if not CACHE_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "캐시 시스템 미설치"
        }

    try:
        stats = get_cache_stats()
        redis_ok = stats.get("redis_available", False)

        return {
            "status": "healthy" if redis_ok else "degraded",
            "redis": "connected" if redis_ok else "disconnected",
            "fallback": "memory_cache"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/keys/{prefix}")
def list_cache_keys(prefix: str, limit: int = 100):
    """
    캐시 키 목록 조회 (디버깅용)

    Args:
        prefix: 프리픽스
        limit: 최대 개수
    """
    if not CACHE_AVAILABLE:
        raise HTTPException(status_code=503, detail="캐시 시스템 사용 불가")

    try:
        from app.core.cache_utils_redis import get_redis_client
        client = get_redis_client()

        if not client:
            return {"keys": [], "message": "Redis 사용 불가 (메모리 캐시 모드)"}

        pattern = f"{prefix}:*"
        keys = [k.decode() for k in client.keys(pattern)[:limit]]

        return {
            "prefix": prefix,
            "count": len(keys),
            "keys": keys[:limit]
        }

    except Exception as e:
        logger.error(f"키 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
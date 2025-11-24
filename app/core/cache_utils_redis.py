# app/core/cache_utils_redis.py
"""
Redis 기반 캐싱 시스템 (프로덕션용)
- 분산 환경 지원
- 영구 저장 (서버 재시작 후에도 유지)
- TTL 자동 관리
"""
import os
import json
import hashlib
from typing import Any, Optional
from functools import wraps
import pickle

try:
    import redis
    from redis.connection import ConnectionPool

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  Redis 미설치. 메모리 캐시 사용 중. 설치: pip install redis")

# Redis 설정
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
DEFAULT_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1시간

# Connection Pool (성능 향상)
_pool = None
_client = None


def get_redis_client():
    """Redis 클라이언트 싱글톤"""
    global _pool, _client

    if not REDIS_AVAILABLE:
        return None

    if _client is None:
        try:
            _pool = ConnectionPool(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=False,  # bytes로 저장 (pickle 사용)
                max_connections=20,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            _client = redis.Redis(connection_pool=_pool)

            # 연결 테스트
            _client.ping()
            print(f"✅ Redis 연결 성공: {REDIS_HOST}:{REDIS_PORT}")

        except redis.ConnectionError as e:
            print(f"❌ Redis 연결 실패: {e}")
            print("   메모리 캐시로 폴백합니다.")
            _client = None

    return _client


# 메모리 캐시 폴백 (Redis 실패 시)
_MEMORY_CACHE = {}


def cache_key(*args, **kwargs) -> str:
    """캐시 키 생성"""
    key_str = f"{args}{sorted(kwargs.items())}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache(key: str, prefix: str = "cache") -> Optional[Any]:
    """
    캐시에서 값 가져오기

    Args:
        key: 캐시 키
        prefix: 키 프리픽스 (네임스페이스)
    """
    full_key = f"{prefix}:{key}"
    client = get_redis_client()

    if client:
        try:
            value = client.get(full_key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            print(f"⚠️  Redis get 에러: {e}")

    # Redis 실패 시 메모리 캐시 사용
    return _MEMORY_CACHE.get(full_key)


def set_cache(key: str, value: Any, ttl: int = DEFAULT_TTL, prefix: str = "cache"):
    """
    캐시에 값 저장

    Args:
        key: 캐시 키
        value: 저장할 값
        ttl: 유효 시간 (초)
        prefix: 키 프리픽스
    """
    full_key = f"{prefix}:{key}"
    client = get_redis_client()

    if client:
        try:
            serialized = pickle.dumps(value)
            client.setex(full_key, ttl, serialized)
            return True
        except Exception as e:
            print(f"⚠️  Redis set 에러: {e}")

    # Redis 실패 시 메모리 캐시에 저장
    _MEMORY_CACHE[full_key] = value
    return False


def delete_cache(key: str, prefix: str = "cache"):
    """캐시 삭제"""
    full_key = f"{prefix}:{key}"
    client = get_redis_client()

    if client:
        try:
            client.delete(full_key)
        except Exception:
            pass

    _MEMORY_CACHE.pop(full_key, None)


def clear_cache(prefix: str = "cache"):
    """특정 프리픽스의 모든 캐시 삭제"""
    client = get_redis_client()

    if client:
        try:
            pattern = f"{prefix}:*"
            keys = client.keys(pattern)
            if keys:
                client.delete(*keys)
                print(f"🗑️  {len(keys)}개 캐시 삭제됨 ({prefix})")
        except Exception as e:
            print(f"⚠️  캐시 삭제 실패: {e}")

    # 메모리 캐시도 정리
    to_delete = [k for k in _MEMORY_CACHE.keys() if k.startswith(f"{prefix}:")]
    for k in to_delete:
        del _MEMORY_CACHE[k]


def cached(ttl: int = DEFAULT_TTL, prefix: str = "cache"):
    """
    동기 함수용 캐싱 데코레이터

    Usage:
        @cached(ttl=3600, prefix="search")
        def expensive_function(query):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"

            # 캐시 확인
            cached_value = get_cache(key, prefix)
            if cached_value is not None:
                return cached_value

            # 캐시 미스 - 함수 실행
            result = func(*args, **kwargs)

            # 캐시 저장
            set_cache(key, result, ttl, prefix)

            return result

        return wrapper

    return decorator


def async_cached(ttl: int = DEFAULT_TTL, prefix: str = "cache"):
    """
    비동기 함수용 캐싱 데코레이터

    Usage:
        @async_cached(ttl=1800, prefix="moderation")
        async def moderate_text(text):
            ...
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"

            # 캐시 확인
            cached_value = get_cache(key, prefix)
            if cached_value is not None:
                return cached_value

            # 캐시 미스 - 함수 실행
            result = await func(*args, **kwargs)

            # 캐시 저장
            set_cache(key, result, ttl, prefix)

            return result

        return wrapper

    return decorator


def get_cache_stats() -> dict:
    """캐시 통계 (모니터링용)"""
    client = get_redis_client()

    stats = {
        "redis_available": client is not None,
        "memory_cache_size": len(_MEMORY_CACHE),
    }

    if client:
        try:
            info = client.info("stats")
            stats.update({
                "redis_keys": client.dbsize(),
                "redis_hits": info.get("keyspace_hits", 0),
                "redis_misses": info.get("keyspace_misses", 0),
                "redis_memory": client.info("memory").get("used_memory_human", "N/A"),
            })

            # 히트율 계산
            hits = stats["redis_hits"]
            misses = stats["redis_misses"]
            if hits + misses > 0:
                stats["hit_rate"] = f"{hits / (hits + misses) * 100:.1f}%"
        except Exception:
            pass

    return stats


# 초기화 시 Redis 연결 테스트
if REDIS_AVAILABLE:
    get_redis_client()
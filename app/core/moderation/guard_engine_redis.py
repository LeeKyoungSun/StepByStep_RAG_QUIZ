# app/core/moderation/guard_engine_redis.py
"""
Redis 캐싱을 사용하는 최적화된 가드 엔진
"""
from __future__ import annotations
import os, hashlib
from typing import Dict, Any, List, Tuple, Optional
from openai import AsyncOpenAI
from app.core.moderation.moderation import Decision
from app.core.moderation.guard_engine import (
    BLOCK_PATTERNS, VICTIM_WHITELIST, EDU_WHITELIST,
    _normalize_full, _match_any, _POLICY,
    SAFE_FALLBACK, BLOCK_REPLY, _is_minor_sexual_request
)

# Redis 캐시 사용
from app.core.cache_utils_redis import get_cache, set_cache

# 비동기 OpenAI 클라이언트
_ASYNC_OAI = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("MODERATION_MODEL", "omni-moderation-latest")

# 캐시 설정
MOD_CACHE_TTL = int(os.getenv("MOD_CACHE_TTL", "1800"))  # 30분


def _hash_text(text: str) -> str:
    """텍스트 해시 (캐시 키용)"""
    return hashlib.md5(text.encode()).hexdigest()


def _get_cached_decision(text: str, channel: str) -> Optional[Decision]:
    """Redis에서 캐시된 결정 가져오기"""
    key = f"{channel}:{_hash_text(text)}"
    cached = get_cache(key, prefix="moderation")
    return cached


def _set_cached_decision(text: str, channel: str, decision: Decision):
    """Redis에 결정 저장"""
    key = f"{channel}:{_hash_text(text)}"
    set_cache(key, decision, ttl=MOD_CACHE_TTL, prefix="moderation")


async def _openai_moderate_async(text: str) -> Tuple[Dict[str, bool], Dict[str, float], bool]:
    """비동기 OpenAI 모더레이션"""
    resp = await _ASYNC_OAI.moderations.create(model=_MODEL, input=text)
    res = resp.results[0]
    cats: Dict[str, bool] = dict(res.categories)
    scs: Dict[str, float] = dict(res.category_scores)
    flagged: bool = bool(res.flagged)
    return cats, scs, flagged


def _pick_action(categories: Dict[str, bool], scores: Dict[str, float], *, channel: str) -> Tuple[str, List[str]]:
    """점수 기반 액션 결정"""
    reasons: List[str] = []

    def on(cat):
        return bool(categories.get(cat, False))

    def sc(cat):
        return float(scores.get(cat, 0.0))

    # 강력 차단
    if on("sexual/minors"):
        reasons.append("sexual_minors")
        return "block", reasons

    if on("self-harm") or on("violence/graphic"):
        reasons.append("self_harm_violence")
        return "block", reasons

    # 성적 노골성
    if on("sexual/explicit") or sc("sexual/explicit") >= 0.5:
        reasons.append("sexual_explicit")
        return ("block", reasons) if channel in ("post", "comment") else ("soft_block", reasons)

    # 불법
    if sc("illicit") >= 0.5 or sc("illicit/violent") >= 0.4:
        reasons.append("illicit")
        return "block", reasons

    # 욕설/혐오
    if on("harassment") or sc("harassment") >= 0.4:
        reasons.append("harassment")
        return "soft_block", reasons

    # 채널별 임계치
    th = _POLICY.get(channel, {})
    for k, t in th.items():
        if scores.get(k, 0.0) >= t:
            reasons.append(f"{k}_threshold")
            return "soft_block", reasons

    return "allow", reasons


async def guard_text_async(
        text: str,
        *,
        channel: str = "generic",
        skip_api: bool = False
) -> Decision:
    """
    Redis 캐싱을 사용하는 비동기 가드
    """
    # Redis 캐시 확인
    cached = _get_cached_decision(text, channel)
    if cached:
        return cached

    t = _normalize_full(text)

    # ===== 순서 중요! 교육 의도를 가장 먼저 체크 =====
    # 1. 교육 의도 (최우선!)
    if _match_any(EDU_WHITELIST, t):
        decision = Decision(
            action="allow",
            reasons=["edu_whitelist"],
            categories={},
            scores={}
        )
        _set_cached_decision(text, channel, decision)
        return decision

    # 2. 피해 신고/상담
    if _match_any(VICTIM_WHITELIST, t):
        decision = Decision(
            action="allow",
            reasons=["victim_context"],
            categories={},
            scores={}
        )
        _set_cached_decision(text, channel, decision)
        return decision

    # 3. 미성년자 + 성적 요청 (명확한 차단 패턴)
    if _is_minor_sexual_request(t):
        decision = Decision(
            action="block",
            reasons=["minor_sexual_intent"],
            categories={},
            scores={}
        )
        _set_cached_decision(text, channel, decision)
        return decision

    # 4. 명시 차단 패턴
    if _match_any(BLOCK_PATTERNS, t):
        decision = Decision(
            action="block",
            reasons=["block_pattern"],
            categories={},
            scores={}
        )
        _set_cached_decision(text, channel, decision)
        return decision

    # 5. skip_api면 허용
    if skip_api:
        decision = Decision(
            action="allow",
            reasons=["skip_api"],
            categories={},
            scores={}
        )
        return decision

    # 6. OpenAI API 호출 (마지막)
    try:
        cats, scs, _ = await _openai_moderate_async(t)
        action, reasons = _pick_action(cats, scs, channel=channel)
        decision = Decision(action=action, reasons=reasons, categories=cats, scores=scs)
        _set_cached_decision(text, channel, decision)
        return decision
    except Exception as e:
        decision = Decision(
            action="hold",
            reasons=[f"api_error: {e.__class__.__name__}"],
            categories={},
            scores={}
        )
        return decision
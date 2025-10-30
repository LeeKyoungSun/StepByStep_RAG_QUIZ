# app/rag/retrievers/rrf.py
from __future__ import annotations
from collections import defaultdict
from typing import Callable, Iterable, List, Hashable, Any
from langchain_core.documents import Document

def default_key_fn(d: Document) -> Hashable:
    md = d.metadata or {}
    # 가능한 한 충돌이 적은 키를 선택
    return md.get("doc_id") or (md.get("source"), md.get("chunk_id"))

def rrf_fuse(
    lists: Iterable[List[Document]],
    *,
    k: int = 60,            # RRF 상수 (50~100 권장)
    top_k: int = 5,
    key_fn: Callable[[Document], Hashable] = default_key_fn,
) -> List[Document]:
    """
    RRF(Reciprocal Rank Fusion)
    score(d) = Σ 1 / (k + rank_i(d))
    - lists: 각 리트리버의 문서 리스트(랭크 오름차순 정렬 가정)
    - k: RRF 상수
    - top_k: 최종 반환 개수
    - key_fn: 문서 중복 판단 키
    """
    rank_score = defaultdict(float)
    first_seen_order: dict[Hashable, int] = {}
    order_counter = 0

    # 점수 누적
    for lst in lists:
        for rank_idx, d in enumerate(lst):
            key = key_fn(d)
            if key not in first_seen_order:
                first_seen_order[key] = order_counter
                order_counter += 1
            rank_score[key] += 1.0 / (k + (rank_idx + 1))

    # 상위 top_k 키
    items = sorted(rank_score.items(), key=lambda x: (-x[1], first_seen_order[x[0]]))
    keep_keys = set(key for key, _ in items[:top_k])

    # 원본 Document 복원 (안정적 순서 유지)
    merged: List[Document] = []
    used: set[Hashable] = set()

    for lst in lists:
        for d in lst:
            key = key_fn(d)
            if key in keep_keys and key not in used:
                merged.append(d)
                used.add(key)
                if len(merged) >= top_k:
                    return merged

    # 부족하면 풀에서 채우기
    if len(merged) < top_k:
        for d in [d for lst in lists for d in lst]:
            key = key_fn(d)
            if key not in used:
                merged.append(d)
                used.add(key)
                if len(merged) >= top_k:
                    break

    return merged
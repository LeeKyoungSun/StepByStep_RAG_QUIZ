# app/rag/retrievers/hybrid_rrf.py
from __future__ import annotations
import os
from typing import List, Sequence, Optional, Dict, Tuple
from pathlib import Path

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# pydantic v1/v2 호환
try:
    from pydantic import ConfigDict, Field
    _MODEL_CONFIG = {"model_config": ConfigDict(arbitrary_types_allowed=True, extra="allow")}
except Exception:
    from pydantic import Field
    class _Cfg:
        arbitrary_types_allowed = True
        extra = "allow"
    _MODEL_CONFIG = {"Config": _Cfg}

from app.core.stores.faiss_store import FaissStore
from app.core.stores.bm25_store import BM25Store
from app.rag.retrievers.utils import dict_to_doc


def _key_of(r: Dict):
    return r.get("doc_id") or (r.get("source") or r.get("src"), r.get("chunk_id"))


def rrf_weighted(fa_rows: List[Dict], bm_rows: List[Dict], k: int, rrf_k: int, w_f: float, w_b: float) -> List[Dict]:
    scores: Dict[Tuple, float] = {}
    seen: Dict[Tuple, Tuple[int, Dict]] = {}
    order = 0
    for rows, w in ((fa_rows, w_f), (bm_rows, w_b)):
        for i, r in enumerate(rows):
            key = _key_of(r)
            if key not in scores:
                scores[key] = 0.0
                seen[key] = (order, r)
                order += 1
            scores[key] += w * (1.0 / (rrf_k + (i + 1)))
    top = sorted(scores.items(), key=lambda x: (-x[1], seen[x[0]][0]))[:k]
    return [seen[k][1] for k, _ in top]


def _read_dirs_from_env() -> List[str]:
    env_dirs = (os.getenv("BM25_DIRS") or "").strip()
    if env_dirs:
        return [d.strip() for d in env_dirs.split(",") if d.strip()]
    fp = os.getenv("BM25_DIRS_FILE")
    if fp and not os.path.isabs(fp):
       fp = os.path.abspath(fp)
    if fp and Path(fp).exists():
        return [ln.strip() for ln in Path(fp).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return []


class HybridRRF(BaseRetriever):
    """FAISS + BM25를 가중 RRF로 융합하는 Retriever (Pydantic 친화형)"""

    # === Pydantic 필드 선언 (중요: 'bm25_stores' 사용, 'bm25' 아님) ===
    faiss: FaissStore
    bm25_stores: List[BM25Store] = Field(default_factory=list)

    top_k: int = 4
    candidate_k: int = 6
    rrf_k: int = 60
    w_faiss: float = 0.7
    w_bm25: float = 0.3

    locals().update(_MODEL_CONFIG)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1) FAISS 후보
        f_rows = self.faiss.search(query, top_k=self.candidate_k)

        # 2) BM25 후보(모든 샤드에서 후보 합침)
        b_rows: List[Dict] = []
        for st in self.bm25_stores:
            b_rows.extend(st.search(query, top_k=self.candidate_k))

        # 3) 가중 RRF로 최종 top_k 융합
        fused = rrf_weighted(
            fa_rows=f_rows,
            bm_rows=b_rows,
            k=self.top_k,
            rrf_k=self.rrf_k,
            w_f=self.w_faiss,
            w_b=self.w_bm25,
        )
        return [dict_to_doc(r) for r in fused]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def load_hybrid_from_env(
    *,
    faiss_dir: Optional[str] = None,
    bm25_dirs: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
    candidate_k: Optional[int] = None,
) -> HybridRRF:
    """환경변수/인자를 읽어 HybridRRF 인스턴스를 생성하는 팩토리."""
    faiss_dir = faiss_dir or os.getenv("FAISS_DIR", "./data/indexes/merged/faiss")
    fa = FaissStore.load(faiss_dir)

    dirs = list(bm25_dirs) if bm25_dirs is not None else _read_dirs_from_env()
    if not dirs:
        raise ValueError("BM25_DIRS 또는 BM25_DIRS_FILE이 비어 있습니다.")
    bm25s = [BM25Store.load(d) for d in dirs]

    return HybridRRF(
        faiss=fa,
        bm25_stores=bm25s,
        top_k=int(top_k or os.getenv("TOP_K", "4")),
        candidate_k=int(candidate_k or os.getenv("CANDIDATE_K", "6")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        w_faiss=float(os.getenv("W_FAISS", "0.7")),
        w_bm25=float(os.getenv("W_BM25", "0.3")),
    )
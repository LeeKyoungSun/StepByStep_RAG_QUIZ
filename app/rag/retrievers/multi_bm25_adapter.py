import os, json
from typing import List, Sequence, Optional
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.rag.retrievers.rrf import rrf_fuse
from app.rag.retrievers.utils import dict_to_doc
from app.core.stores.bm25_store import BM25Store


def _read_dirs_from_file(path: str) -> List[str]:
    out: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p:
                    out.append(p)
    except FileNotFoundError:
        pass
    return out


def _load_dirs_from_env() -> List[str]:
    # 1) 파일 경로 우선
    file_var = os.getenv("BM25_DIRS_FILE")
    if file_var and os.path.exists(file_var):
        dirs = _read_dirs_from_file(file_var)
        if dirs:
            return dirs

    # 2) JSON 배열 지원 (예: BM25_DIRS_JSON='["a","b","c"]')
    json_var = os.getenv("BM25_DIRS_JSON")
    if json_var:
        try:
            dirs = json.loads(json_var)
            if isinstance(dirs, list) and dirs:
                return [str(d).strip() for d in dirs if str(d).strip()]
        except Exception:
            pass

    # 3) CSV (쉼표 구분)
    csv_var = os.getenv("BM25_DIRS")
    if csv_var:
        dirs = [p.strip() for p in csv_var.split(",") if p.strip()]
        if dirs:
            return dirs

    return []


class MultiBM25Retriever(BaseRetriever):
    # ✅ Pydantic v2 필드 선언 (여기에 정의해야 __init__ 없이도 세팅 가능)
    stores: List[BM25Store] = Field(default_factory=list)
    k: int = 6

    # 임의 타입 허용
    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_dirs(cls, index_dirs: Sequence[str], candidate_k: int = 6) -> "MultiBM25Retriever":
        stores = [BM25Store.load(d) for d in index_dirs]
        return cls(stores=stores, k=candidate_k)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        bags: List[List[Document]] = []
        for st in self.stores:
            rows = st.search(query, top_k=self.k)
            docs = [dict_to_doc(r) for r in rows]
            bags.append(docs)
        rrf_k = int(os.getenv("RRF_K", "60"))
        return rrf_fuse(bags, k=rrf_k, top_k=self.k)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


def load_bm25_retrievers_from_env() -> MultiBM25Retriever:
    dirs = _load_dirs_from_env()
    if not dirs:
        raise ValueError("BM25_DIRS 환경변수가 비어 있습니다.")
    cand_k = int(os.getenv("CANDIDATE_K", "6"))
    return MultiBM25Retriever.from_dirs(dirs, candidate_k=cand_k)
from __future__ import annotations
from typing import List, Sequence, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)
from app.rag.retrievers.rrf import rrf_fuse
from app.rag.retrievers.utils import dict_to_doc
from app.core.stores.bm25_store import BM25Store
import asyncio


class MultiBM25Retriever(BaseRetriever):
    """여러 bm25.pkl을 동시에 검색(논리 병합)하는 LangChain 호환 리트리버"""

    def __init__(self, index_dirs: Sequence[str], candidate_k: int = 6):
        super().__init__()
        # BM25Store.load()가 반드시 store를 return 해야 함!
        self.stores = [BM25Store.load(d) for d in index_dirs]
        self.k = candidate_k

    # --- 동기 ---
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
        # RRF 상수 k=60, 최종 반환 개수 top_k=self.k
        return rrf_fuse(bags, k=60, top_k=self.k)

    # --- 비동기 ---
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        # I/O 바운드 아님 → 스레드로 동기 함수 실행
        return await asyncio.to_thread(self._get_relevant_documents, query)
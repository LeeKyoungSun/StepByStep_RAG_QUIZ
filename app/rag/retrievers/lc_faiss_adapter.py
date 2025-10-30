# app/rag/retrievers/lc_faiss_adapter.py

from typing import List, Any
from langchain_core.documents import Document
from app.core.stores.faiss_store import FaissStore
import asyncio


class FAISSRetriever:
    """
    LangChain-style retriever wrapper around our FaissStore.
    - 사용법: retriever = FAISSRetriever(faiss_dir, top_k=5)
               docs = retriever.get_relevant_documents("피임 방법")
    """

    def __init__(self, faiss_dir: str, top_k: int = 5):
        self.faiss_dir = faiss_dir
        self.top_k = top_k
        self.store = FaissStore.load(faiss_dir)
        self._lock = asyncio.Lock()  # 비동기 안전성 확보용

    # -----------------------------
    # 동기 버전 (필요시 Chain 외부에서도 사용 가능)
    # -----------------------------
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.store.search(query, top_k=self.top_k)
        docs: List[Document] = []
        for r in results:
            meta = {k: v for k, v in r.items() if k not in ["text", "score"]}
            meta["score"] = r["score"]
            docs.append(Document(page_content=r["text"], metadata=meta))
        return docs

    # -----------------------------
    # 비동기 버전 (LangChain 체인 내부용)
    # -----------------------------
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        async with self._lock:
            return await asyncio.to_thread(self.get_relevant_documents, query)
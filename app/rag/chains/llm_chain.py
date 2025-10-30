from typing import List
import glob, os, asyncio
from langchain_core.documents import Document
from app.core.config import cfg
from app.rag.retrievers.rrf import rrf_fuse
from app.rag.retrievers.multi_bm25_adapter import MultiBM25Retriever
from app.rag.prompts.prompts_lc import build_prompt
from openai import AsyncOpenAI

# -----------------------------
# 다중 인덱스 정의
# -----------------------------
BM25_SHARDS = sorted(glob.glob(os.path.join("data/indexes", "*/bm25")))
FAISS_DIR = cfg.FAISS_DIR  # .env에서 병합본 지정됨

# -----------------------------
# RAG 체인
# -----------------------------
class RAGLightHybrid:
    def __init__(self, top_k: int = 4, candidate_k: int = 6, enable_bm25: bool = True, enable_rrf: bool = True):
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.enable_bm25 = enable_bm25
        self.enable_rrf = enable_rrf

        # 리트리버 세팅
        self.faiss_r = FAISSRetriever(FAISS_DIR, top_k=self.candidate_k)
        self.bm25_r = MultiBM25Retriever(BM25_SHARDS, self.candidate_k) if enable_bm25 else None

        # LLM 클라이언트 (비동기 OpenAI)
        self.llm = AsyncOpenAI()

    # -----------------------------
    # 검색 및 LLM 응답 생성
    # -----------------------------
    async def arun(self, question: str):
        #  비동기 검색 (FAISS, BM25 병렬 실행)
        if self.enable_bm25 and self.bm25_r:
            faiss_task = asyncio.create_task(self.faiss_r._aget_relevant_documents(question))
            bm25_task = asyncio.create_task(self.bm25_r._aget_relevant_documents(question))
            faiss_docs, bm25_docs = await asyncio.gather(faiss_task, bm25_task)
        else:
            faiss_docs = await self.faiss_r._aget_relevant_documents(question)
            bm25_docs = []

        #  결과 융합 (RRF or 단일)
        if self.enable_bm25 and self.enable_rrf and bm25_docs:
            fused = rrf_fuse([faiss_docs, bm25_docs], k=self.top_k)
        else:
            fused = faiss_docs[: self.top_k]

        #  프롬프트 구성
        context_str = "\n\n".join(
            [f"[{i+1}] {doc.page_content.strip()}" for i, doc in enumerate(fused)]
        )
        prompt = build_prompt(question, context_str)

        #  LLM 호출
        resp = await self.llm.chat.completions.create(
            model=cfg.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant for sexuality education."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        answer = resp.choices[0].message.content.strip()

        # 5️⃣ 인용 출처 추출
        citations = [doc.metadata.get("source") for doc in fused if doc.metadata.get("source")]

        return {"answer": answer, "citations": citations}
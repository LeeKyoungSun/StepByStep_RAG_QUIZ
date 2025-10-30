from typing import Optional
from fastapi import FastAPI, Body
from pydantic import BaseModel

from app.core.config import cfg
from app.rag.chains.llm_chain import RAGLightHybrid

app = FastAPI(title="StepByStep RAG & QUIZ (Light-Hybrid)", version="1.0.0")

# 앱 기동 시 기본 체인 준비
default_chain = RAGLightHybrid()


class ChatIn(BaseModel):
    message: str
    userId: Optional[str] = None
    top_k: Optional[int] = None
    enable_bm25: Optional[bool] = None
    enable_rrf: Optional[bool] = None
    # (선택) corpus 이름으로 스위칭하고 싶다면 여기에 추가 가능:
    # corpus: Optional[str] = None


class ChatOut(BaseModel):
    answer: str
    citations: list[str] = []


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "light-hybrid",
        "bm25": cfg.ENABLE_BM25,
        "rrf": cfg.ENABLE_RRF,
        "faiss_dir": cfg.FAISS_DIR,
        "bm25_dir": cfg.BM25_DIR,
        "model": cfg.OPENAI_MODEL,
    }


@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn = Body(...)):
    # 요청별 토글/파라미터 반영(없으면 기본값 사용)
    chain = RAGLightHybrid(
        top_k=body.top_k or cfg.TOP_K,
        candidate_k=cfg.CANDIDATE_K,
        enable_bm25=cfg.ENABLE_BM25 if body.enable_bm25 is None else body.enable_bm25,
        enable_rrf=cfg.ENABLE_RRF if body.enable_rrf is None else body.enable_rrf,
    )
    result = await chain.arun(body.message)
    return ChatOut(answer=result["answer"], citations=result["citations"])
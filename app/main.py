# from pathlib import Path
# from dotenv import load_dotenv
# import os
#
# # .env 파일 절대경로 지정
# env_file = Path(__file__).parent.parent / ".env"
# load_dotenv(env_file)
#
# from fastapi import FastAPI
# from app.api.routes_rag import router as rag_router
# from app.routers.quiz import router as quiz_router
#
# app = FastAPI()
#
# @app.get("/healthz")
# def health_check():
#     return {"status": "ok"}
#
# app.include_router(rag_router, prefix="/api")
# app.include_router(quiz_router, prefix="/api/quiz", tags=["quiz"])
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_rag import router as rag_router
from app.routers.quiz import router as quiz_router

app = FastAPI(title="StepByStep RAG & QUIZ", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 BE 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

app.include_router(rag_router, prefix="/api")
app.include_router(quiz_router, prefix="/api/quiz", tags=["quiz"])
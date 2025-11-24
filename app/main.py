from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_rag import router as rag_router
from app.routers.quiz import router as quiz_router
from app.api.routes_cache import router as cache_router


app = FastAPI(title="StepByStep RAG & QUIZ", version="1.0.0")
app.include_router(cache_router, prefix="/api/cache", tags=["cache"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

app.include_router(rag_router, prefix="/api")
app.include_router(quiz_router, prefix="/api/quiz", tags=["quiz"])
app.include_router(cache_router, prefix="/api/cache", tags=["cache"])
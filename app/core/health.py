# app/routers/health.py
from fastapi import APIRouter
import time

router = APIRouter()
START = time.time()

@router.get("/healthz", tags=["health"])
def healthz():
    return {
        "status": "success",
        "message": "ok",
        "data": {
            "uptime": round(time.time() - START, 3),
            "indexReady": True
        }
    }
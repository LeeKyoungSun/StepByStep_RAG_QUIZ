# app/routers/moderation_internal.py
from fastapi import APIRouter
from typing import List
from app.schemas.moderation import (
    TextIn, TextBatchIn, CheckResult, CheckBatchResult, Decision
)
from utils.moderation import guard_text, SAFE_FALLBACK

router = APIRouter()

def _make_result(text: str) -> CheckResult:
    gd = guard_text(text)
    if gd.action == "soft_block":
        return CheckResult(decision=Decision(**gd.__dict__), safe_text=SAFE_FALLBACK, original=text)
    return CheckResult(decision=Decision(**gd.__dict__), original=text)

# 배치 텍스트 검사 (내부)
@router.post("/check-batch", response_model=CheckBatchResult)
def check_batch(body: TextBatchIn):
    results: List[CheckResult] = [_make_result(t) for t in body.texts]
    return CheckBatchResult(results=results)

# RAG 스니펫 사전 필터 (내부)
@router.post("/filter-snippets", response_model=CheckBatchResult)
def filter_snippets(body: TextBatchIn):
    """
    스니펫을 allow만 통과시키고, 나머지는 soft_block 처리로 대체.
    """
    outs: List[CheckResult] = []
    for t in body.texts:
        gd = guard_text(t)
        if gd.action == "allow":
            outs.append(CheckResult(decision=Decision(**gd.__dict__), original=t))
        else:
            outs.append(CheckResult(decision=Decision(**gd.__dict__), safe_text="(안전상 요약 생략)", original=t))
    return CheckBatchResult(results=outs)

# LLM 입력 가드 (내부)
@router.post("/guard-input", response_model=CheckResult)
def guard_input(body: TextIn):
    return _make_result(body.text)

# LLM 출력 가드 (내부)
@router.post("/guard-output", response_model=CheckResult)
def guard_output(body: TextIn):
    return _make_result(body.text)

# 혼합 배치 (내부)
@router.post("/guard-batch", response_model=CheckBatchResult)
def guard_batch(body: TextBatchIn):
    return check_batch(body)
# app/core/moderation/moderation_internal.py
from fastapi import APIRouter
from typing import List
from app.core.moderation.moderation import (
    TextIn, TextBatchIn, CheckResult, CheckBatchResult, Decision,
    make_check_result, SAFE_FALLBACK
)
from app.core.moderation.guard_engine import guard_text

router = APIRouter()

# 배치 검사
@router.post("/check-batch", response_model=CheckBatchResult)
def check_batch(body: TextBatchIn):
    results: List[CheckResult] = []
    for t in body.texts:
        gd: Decision = guard_text(t, channel="chat_input")
        safe = SAFE_FALLBACK if gd.action == "soft_block" else None
        results.append(make_check_result(text=t, decision=gd, safe_text=safe))
    return CheckBatchResult(results=results)

# RAG 스니펫 필터링
@router.post("/filter-snippets", response_model=CheckBatchResult)
def filter_snippets(body: TextBatchIn):
    outs: List[CheckResult] = []
    for t in body.texts:
        gd: Decision = guard_text(t, channel="snippet")
        safe = "(안전상 요약 생략)" if gd.action != "allow" else None
        outs.append(make_check_result(text=t, decision=gd, safe_text=safe))
    return CheckBatchResult(results=outs)

@router.post("/guard-input", response_model=CheckResult)
def guard_input(body: TextIn):
    gd = guard_text(body.text, channel="chat_input")
    safe = SAFE_FALLBACK if gd.action == "soft_block" else None
    return make_check_result(text=body.text, decision=gd, safe_text=safe)

@router.post("/guard-output", response_model=CheckResult)
def guard_output(body: TextIn):
    gd = guard_text(body.text, channel="chat_output")
    safe = SAFE_FALLBACK if gd.action == "soft_block" else None
    return make_check_result(text=body.text, decision=gd, safe_text=safe)

@router.post("/guard-batch", response_model=CheckBatchResult)
def guard_batch(body: TextBatchIn):
    return check_batch(body)
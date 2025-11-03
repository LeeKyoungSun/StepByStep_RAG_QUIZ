from fastapi import APIRouter
from typing import List
from app.core.moderation.moderation import (
    TextIn, TextBatchIn, CheckResult, CheckBatchResult, make_check_result
)

router = APIRouter()

# 배치 검사
@router.post("/check-batch", response_model=CheckBatchResult)
def check_batch(body: TextBatchIn):
    return CheckBatchResult(results=[make_check_result(t) for t in body.texts])

# RAG 스니펫 필터링
@router.post("/filter-snippets", response_model=CheckBatchResult)
def filter_snippets(body: TextBatchIn):
    outs: List[CheckResult] = []
    for t in body.texts:
        r = make_check_result(t)
        if r.decision.action == "allow":
            outs.append(r)
        else:
            r.safe_text = "(안전상 일부 문구 생략)"
            outs.append(r)
    return CheckBatchResult(results=outs)

# 챗봇 입력 가드
@router.post("/guard-input", response_model=CheckResult)
def guard_input(body: TextIn):
    return make_check_result(body.text)

# 챗봇 출력 가드
@router.post("/guard-output", response_model=CheckResult)
def guard_output(body: TextIn):
    return make_check_result(body.text)
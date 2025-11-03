from fastapi import APIRouter, HTTPException
from app.core.moderation.moderation import (
    TextIn, CheckResult, PostGuardIn, CommentGuardIn, make_check_result
)

router = APIRouter()

# 단건 검사 (외부 공개용)
@router.post("/check", response_model=CheckResult)
def check_text(body: TextIn):
    return make_check_result(body.text)

# 게시글 생성/수정 가드
@router.post("/guard-post", response_model=CheckResult)
def guard_post(body: PostGuardIn):
    text = f"{body.title or ''}\n{body.content}".strip()
    r = make_check_result(text)
    if r.decision.action == "block":
        raise HTTPException(status_code=422, detail={"reason": r.decision.reasons})
    # soft_block이면 안전문으로 저장
    if r.decision.action == "soft_block" and r.safe_text:
        body.content = r.safe_text
    return r

# 댓글 생성/수정 가드
@router.post("/guard-comment", response_model=CheckResult)
def guard_comment(body: CommentGuardIn):
    r = make_check_result(body.content)
    if r.decision.action == "block":
        raise HTTPException(status_code=422, detail={"reason": r.decision.reasons})
    if r.decision.action == "soft_block" and r.safe_text:
        body.content = r.safe_text
    return r
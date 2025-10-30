# app/routers/moderation_public.py
from fastapi import APIRouter
from app.schemas.moderation import (
    TextIn, CheckResult, Decision,
    PostGuardIn, CommentGuardIn
)
from utils.moderation import guard_text, SAFE_FALLBACK

router = APIRouter()

def _make_result(text: str) -> CheckResult:
    gd = guard_text(text)
    if gd.action == "soft_block":
        return CheckResult(decision=Decision(**gd.__dict__), safe_text=SAFE_FALLBACK, original=text)
    return CheckResult(decision=Decision(**gd.__dict__), original=text)

# 단건 텍스트 검사(원하면 외부 공개)
@router.post("/check", response_model=CheckResult)
def check_text(body: TextIn):
    return _make_result(body.text)

# 게시물 생성/수정 가드 (외부 공개)
@router.post("/guard-post", response_model=CheckResult)
def guard_post(body: PostGuardIn):
    text = f"{body.title or ''}\n{body.content}".strip()
    return _make_result(text)

# 댓글 생성/수정 가드 (외부 공개)
@router.post("/guard-comment", response_model=CheckResult)
def guard_comment(body: CommentGuardIn):
    return _make_result(body.content)
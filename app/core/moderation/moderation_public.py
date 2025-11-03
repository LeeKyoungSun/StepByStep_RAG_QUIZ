# app/core/moderation/moderation_public.py
from fastapi import APIRouter, HTTPException
from app.core.moderation.moderation import (
    TextIn, CheckResult, Decision, PostGuardIn, CommentGuardIn,
    make_check_result, SAFE_FALLBACK
)
from app.core.moderation.guard_engine import guard_text

router = APIRouter()

@router.post("/check", response_model=CheckResult)
def check_text(body: TextIn):
    gd: Decision = guard_text(body.text, channel="chat_input")
    safe = SAFE_FALLBACK if gd.action == "soft_block" else None
    return make_check_result(text=body.text, decision=gd, safe_text=safe)

@router.post("/guard-post", response_model=CheckResult)
def guard_post(body: PostGuardIn):
    text = f"{body.title or ''}\n{body.content}".strip()
    gd: Decision = guard_text(text, channel="post")
    if gd.action == "block":
        raise HTTPException(status_code=422, detail={"decision": gd.model_dump()})
    safe = SAFE_FALLBACK if gd.action == "soft_block" else None
    return make_check_result(text=text, decision=gd, safe_text=safe)

@router.post("/guard-comment", response_model=CheckResult)
def guard_comment(body: CommentGuardIn):
    gd: Decision = guard_text(body.content, channel="comment")
    if gd.action == "block":
        raise HTTPException(status_code=422, detail={"decision": gd.model_dump()})
    safe = SAFE_FALLBACK if gd.action == "soft_block" else None
    return make_check_result(text=body.content, decision=gd, safe_text=safe)
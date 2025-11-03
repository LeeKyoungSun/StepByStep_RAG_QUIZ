# app/core/moderation/moderation.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.core.moderation.guard_engine import check_with_openai, SAFE_FALLBACK  # ⬅️ 추가

# ===== 모델 =====
class Decision(BaseModel):
    action: str               # allow | soft_block | hold | block
    reasons: List[str] = []
    categories: Dict[str, Any] = {}
    scores: Dict[str, float] = {}

class TextIn(BaseModel):
    text: str = Field(..., description="검사할 텍스트")

class TextBatchIn(BaseModel):
    texts: List[str]

class CheckResult(BaseModel):
    decision: Decision
    safe_text: Optional[str] = None
    original: Optional[str] = None

class CheckBatchResult(BaseModel):
    results: List[CheckResult]

class PostGuardIn(BaseModel):
    title: Optional[str] = None
    content: str

class CommentGuardIn(BaseModel):
    content: str

# ===== 공용 헬퍼 =====
def guard_text(text: str, *, channel: str = "chat_input") -> Decision:
    """
    channel: chat_input | chat_output | snippet | post | comment
    """
    return check_with_openai(text, channel=channel)

def make_check_result(text: str, *, channel: str) -> CheckResult:
    dec = guard_text(text, channel=channel)
    if dec.action == "soft_block":
        return CheckResult(decision=dec, safe_text=SAFE_FALLBACK, original=text)
    return CheckResult(decision=dec, original=text)
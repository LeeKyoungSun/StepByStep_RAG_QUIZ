# app/core/moderation/moderation.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# 안전 대체문(공용 상수) — guard_engine 이걸 참조
SAFE_FALLBACK = "민감하거나 유해할 수 있는 내용이라 자세한 설명은 생략할게. 안전한 범위에서 다시 질문해줘."

# ===== 공용 데이터 모델 =====
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
    safe_text: Optional[str] = None   # soft_block 시 대체문
    original: Optional[str] = None

class CheckBatchResult(BaseModel):
    results: List[CheckResult]

class PostGuardIn(BaseModel):
    title: Optional[str] = None
    content: str

class CommentGuardIn(BaseModel):
    content: str

# ===== 공용 헬퍼 =====
def make_check_result(*, text: str, decision: Decision, safe_text: Optional[str]=None) -> CheckResult:
    """
    라우터에서 공통으로 쓰는 Result 포매터.
    """
    return CheckResult(decision=decision, safe_text=safe_text, original=text)
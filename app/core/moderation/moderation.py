# app/schemas/moderation.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# 공통
class Decision(BaseModel):
    action: str               # allow | soft_block | hold | block
    reasons: List[str] = []
    categories: Dict[str, Any] = {}
    scores: Dict[str, float] = {}

# 단건/배치 입력
class TextIn(BaseModel):
    text: str = Field(..., description="검사할 텍스트")

class TextBatchIn(BaseModel):
    texts: List[str]

# 결과 (단건)
class CheckResult(BaseModel):
    decision: Decision
    safe_text: Optional[str] = None  # soft_block 시 대체문
    original: Optional[str] = None

# 결과 (배치)
class CheckBatchResult(BaseModel):
    results: List[CheckResult]

# 게시물/댓글 스키마
class PostGuardIn(BaseModel):
    title: Optional[str] = None
    content: str

class CommentGuardIn(BaseModel):
    content: str
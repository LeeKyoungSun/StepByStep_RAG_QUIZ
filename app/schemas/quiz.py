# app/schemas/quiz.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# -------------------------------
# 공통 상수
# -------------------------------
MAX_CHOICES = 4
POINTS_ON_CORRECT = 20          # 내부 1점 = 외부 20포인트
QUIZ_SESSION_TTL_MIN = 30       # (세션 TTL, 현재 DB 기반이라 의미 없음)

# -------------------------------
# 1) 키워드 목록
# -------------------------------
class Keyword(BaseModel):
    key: str
    label: str
    sampleTopics: List[str] = []


# -------------------------------
# 2) 퀴즈 세트 생성 (응답)
# -------------------------------
class QuizItem(BaseModel):
    itemId: str                 # DB question_id (문항 고유 ID)
    type: str                   # "situation" | "concept"
    question: str
    choices: List[str]
    references: Optional[List[Dict[str, Any]]] = None


class QuizGetResponse(BaseModel):
    quizId: str                 # attempt.id (DB 기준)
    mode: str
    keyword: Optional[str] = None
    total: int
    items: List[QuizItem]


# -------------------------------
# 3) 보기 선택 및 제출
# -------------------------------
class SubmitAnswerRequest(BaseModel):
    quizId: str
    itemId: str                 # DB question_id 문자열
    choiceIndex: int = Field(ge=0, le=MAX_CHOICES - 1)


class SubmitAnswerResponse(BaseModel):
    correct: bool               # 정오 여부
    correctIndex: int           # 정답 인덱스(0~3)
    explanation: Optional[str] = None
    earnedPoints: int           # 이번 문항으로 얻은 포인트
    balance: Optional[int] = None   # 현재 잔액 (포인트 시스템 연동 시 사용)
    resultId: str               # quizId (=attempt.id)


# -------------------------------
# 4) 결과 조회
# -------------------------------
class ResultItem(BaseModel):
    itemId: str
    yourChoice: int
    correctIndex: int
    correct: bool
    earnedPoints: int
    question: Optional[str] = None
    choices: Optional[List[str]] = None
    explanation: Optional[str] = None


class QuizResultResponse(BaseModel):
    resultId: str
    total: int
    correctCount: int
    earnedPointsTotal: int
    items: List[ResultItem]
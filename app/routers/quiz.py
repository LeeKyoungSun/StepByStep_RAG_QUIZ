# app/routers/quiz.py
"""
퀴즈 API 라우터
- 키워드 검색
- 퀴즈 생성 (RAG 기반)
- 답안 제출
- 결과 조회
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.schemas.quiz import (
    Keyword, QuizItem, QuizGetResponse,
    SubmitAnswerRequest, SubmitAnswerResponse,
    QuizResultResponse, ResultItem,
)
from app.crud.quiz import (
    create_attempt_with_items,
    submit_choice,
    read_result,
)

# 퀴즈 생성 엔진
from app.scenarios.service import get_service
from app.scenarios.keyword_rules import match_keywords

router = APIRouter()


# ---------- DB 의존성 ----------
def get_db():
    """DB 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- 1) 키워드 목록 ----------
@router.get("/keywords", response_model=Dict[str, List[Keyword]])
def get_keywords(
        q: Optional[str] = None,
        limit: int = Query(50, ge=1, le=200),
):
    """
    키워드 목록 조회

    - q: 검색어 (옵션)
    - limit: 최대 결과 수
    """
    items = match_keywords(q or "", limit)

    # Keyword 스키마로 변환
    keywords = [
        Keyword(
            key=item["key"],
            label=item["label"],
            sampleTopics=item.get("sampleTopics", [])
        )
        for item in items
    ]

    return {"items": keywords}


# ---------- 2) 퀴즈 세트 생성 ----------
@router.get("", response_model=QuizGetResponse)
def create_quiz(
        mode: str = Query(..., description="by_keyword | random"),
        keyword: Optional[str] = None,
        n: int = Query(5, ge=1, le=10),
        user_id: int = Query(1, description="임시 사용자 ID"),
        db: Session = Depends(get_db),
):
    """
    퀴즈 생성 (개선: 중복 방지)

    - mode: by_keyword (키워드 기반) 또는 random (랜덤)
    - keyword: 키워드 (mode=by_keyword일 때 필수)
    - n: 문제 수 (1-10)
    - user_id: 사용자 ID (임시)
    """
    # 파라미터 검증
    if mode not in {"by_keyword", "random"}:
        raise HTTPException(
            400,
            detail={"error": {"code": "INVALID_PARAM", "message": "mode must be by_keyword|random"}}
        )

    if mode == "by_keyword" and not keyword:
        raise HTTPException(
            400,
            detail={"error": {"code": "INVALID_PARAM", "message": "keyword required for by_keyword mode"}}
        )

    # 퀴즈 생성 서비스 (매번 새로 생성 - 메모리 관리)
    svc = get_service()

    # RAG로 스니펫 가져오기
    if mode == "by_keyword":
        snips = svc.pick_by_keyword(keyword, svc.cfg.topk)
    else:
        snips = svc.random_snippets(svc.cfg.topk)

    if not snips:
        raise HTTPException(
            404,
            detail={"error": {"code": "NO_CONTENT", "message": f"No content found for keyword: {keyword}"}}
        )

    # 퀴즈 항목 생성
    raw_items = []
    for i in range(n):
        try:
            # 각 문제마다 다른 스니펫 조합 사용
            item_snips = snips[i % len(snips):] + snips[:i % len(snips)]

            item = svc.make_quiz_item(
                keyword=keyword,
                snips=item_snips[:6],  # 각 문제당 6개 스니펫
                force_type=None,
                concept_topic=None
            )
            raw_items.append(item)
        except Exception as e:
            print(f"[경고] 문제 생성 실패 ({i + 1}/{n}): {e}")
            continue

    if not raw_items:
        raise HTTPException(
            500,
            detail={"error": {"code": "GENERATION_FAILED", "message": "Failed to generate quiz items"}}
        )

    # DB에 저장
    attempt, item_to_qid, _ = create_attempt_with_items(
        db,
        user_id=user_id,
        scenario_title=keyword or "랜덤",
        items=[
            {
                "question": it["question"],
                "choices": it["choices"],
                "answer_index": it["correct_index"],
                "explanation": it.get("explanation", "")
            }
            for it in raw_items
        ]
    )
    db.commit()

    # 응답 구성
    items = []
    for i, it in enumerate(raw_items, start=1):
        qid = item_to_qid[f"it_{i:02d}"]
        items.append(QuizItem(
            itemId=str(qid),
            type=it.get("type", "situation"),
            question=it["question"],
            choices=it["choices"],
            references=it.get("sources") or None,
        ))

    return QuizGetResponse(
        quizId=str(attempt.id),
        mode=mode,
        keyword=keyword,
        total=len(items),
        items=items
    )


# ---------- 3) 보기 선택 및 제출 ----------
@router.post("/answer", response_model=SubmitAnswerResponse)
def post_answer(
        req: SubmitAnswerRequest,
        db: Session = Depends(get_db)
):
    """
    답안 제출 및 채점

    - quizId: 퀴즈 시도 ID
    - itemId: 문항 ID
    - choiceIndex: 선택한 인덱스 (0-3)
    """
    try:
        attempt_id = int(req.quizId)
        question_id = int(req.itemId)
    except ValueError:
        raise HTTPException(
            400,
            detail={"error": {"code": "INVALID_PARAM", "message": "IDs must be numeric strings"}}
        )

    # 채점
    correct, correct_idx, explanation, earned = submit_choice(
        db,
        attempt_id=attempt_id,
        question_id=question_id,
        choice_index=req.choiceIndex
    )
    db.commit()

    return SubmitAnswerResponse(
        correct=correct,
        correctIndex=correct_idx,
        explanation=explanation or "해설이 없습니다.",
        earnedPoints=20 if earned > 0 else 0,  # 외부 포인트 스케일
        balance=0,  # 포인트 시스템 미구현
        resultId=req.quizId
    )


# ---------- 4) 결과 조회 ----------
@router.get("/results/{resultId}", response_model=QuizResultResponse)
def get_results(
        resultId: str,
        db: Session = Depends(get_db)
):
    """
    퀴즈 결과 조회

    - resultId: 퀴즈 시도 ID
    """
    try:
        attempt_id = int(resultId)
    except ValueError:
        raise HTTPException(
            400,
            detail={"error": {"code": "INVALID_PARAM", "message": "resultId must be numeric"}}
        )

    data = read_result(db, attempt_id)

    if not data:
        raise HTTPException(
            404,
            detail={"error": {"code": "NOT_FOUND", "message": "Result not found"}}
        )

    # 응답 구성
    items: List[ResultItem] = []
    for r in data["items"]:
        items.append(ResultItem(
            itemId=str(r["itemId"]),
            yourChoice=r["yourChoice"],
            correctIndex=r["correctIndex"],
            correct=r["correct"],
            earnedPoints=r["earnedPoints"] * 20,  # 외부 포인트 스케일
            question=r.get("question"),
            choices=r.get("choices"),
            explanation=r.get("explanation"),
        ))

    return QuizResultResponse(
        resultId=data["resultId"],
        total=data["total"],
        correctCount=data["correctCount"],
        earnedPointsTotal=data["earnedPointsTotal"] * 20,
        items=items
    )
# app/crud/quiz.py
"""
퀴즈 CRUD 작업
- 해설(explanation) 지원
- 정답 위치 랜덤 지원
- 멱등성 보장
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select, update

from app.models.quiz import (
    QuizScenario, QuizQuestion, QuizOption,
    QuizAttempt, QuizResponse
)


# ---------- 시나리오 보조 ----------
def get_or_create_scenario(db: Session, title: str) -> QuizScenario:
    """시나리오 조회 또는 생성"""
    row = db.execute(
        select(QuizScenario).where(QuizScenario.title == title)
    ).scalar_one_or_none()

    if row:
        return row

    row = QuizScenario(title=title)
    db.add(row)
    db.flush()
    return row


# ---------- 퀴즈 생성(문항/보기 저장) ----------
def create_attempt_with_items(
        db: Session,
        user_id: int,
        scenario_title: str,
        items: List[Dict]  # [{question, choices[4], answer_index, explanation}]
) -> Tuple[QuizAttempt, Dict[str, int], Dict[str, int]]:
    """
    퀴즈 시도 및 문항 생성

    Args:
        db: DB 세션
        user_id: 사용자 ID
        scenario_title: 시나리오 제목
        items: 문항 리스트

    Returns:
        (attempt, itemId->question_id, itemId->correct_index)
    """
    scenario = get_or_create_scenario(db, scenario_title or "자동생성 시나리오")

    attempt = QuizAttempt(
        user_id=user_id,
        scenario_id=scenario.id,
        score_max=len(items),
        score_total=0,
        started_at=datetime.utcnow(),
        status="IN_PROGRESS",
    )
    db.add(attempt)
    db.flush()

    item_to_qid: Dict[str, int] = {}
    answers: Dict[str, int] = {}

    for i, it in enumerate(items, start=1):
        #  해설 필드 추가
        explanation = it.get("explanation") or it.get("correct_text") or ""

        q = QuizQuestion(
            stem=it.get("question") or "",
            correct_text=explanation,  # 해설 저장
            scenario_id=scenario.id,
        )
        db.add(q)
        db.flush()

        # 보기 저장 (A-D 라벨 고정)
        choices = it.get("choices") or []
        ai = int(it.get("answer_index", 0))
        answers[f"it_{i:02d}"] = ai

        for idx, text in enumerate(choices[:4]):
            label = "ABCD"[idx]
            opt = QuizOption(
                is_correct=(idx == ai),  #  정답 위치가 랜덤일 수 있음
                label=label,
                text=text or "",
                question_id=q.id,
            )
            db.add(opt)

        item_to_qid[f"it_{i:02d}"] = q.id

    db.flush()
    return attempt, item_to_qid, answers


# ---------- 제출 & 채점 ----------
def submit_choice(
        db: Session,
        attempt_id: int,
        question_id: int,
        choice_index: int
) -> Tuple[bool, int, Optional[str], int]:
    """
    선택지 제출 및 채점

    Args:
        db: DB 세션
        attempt_id: 시도 ID
        question_id: 문항 ID
        choice_index: 선택한 인덱스 (0-3)

    Returns:
        (correct, correct_index, explanation, earned_points)
    """
    # 이미 제출했는지 체크 (멱등성)
    existed = db.execute(
        select(QuizResponse).where(
            (QuizResponse.attempt_id == attempt_id) &
            (QuizResponse.question_id == question_id)
        )
    ).scalar_one_or_none()

    if existed:
        # 이미 제출한 경우: earned=0, 기존 정답 기준 반환
        correct_idx = _get_correct_index(db, question_id)
        explanation = _get_explanation(db, question_id)
        return (bool(existed.is_correct), correct_idx, explanation, 0)

    # 정답 인덱스 계산
    correct_idx = _get_correct_index(db, question_id)
    is_ok = (choice_index == correct_idx)

    # 사용자가 고른 option_id
    opt_id = _get_option_id_by_index(db, question_id, choice_index)

    # 응답 저장
    resp = QuizResponse(
        attempt_id=attempt_id,
        question_id=question_id,
        is_correct=is_ok,
        score=1 if is_ok else 0,
        option_id=opt_id,
        text_answer=None,
        created_at=datetime.utcnow(),
    )
    db.add(resp)

    # 점수합 갱신
    if is_ok:
        db.execute(
            update(QuizAttempt)
            .where(QuizAttempt.id == attempt_id)
            .values(score_total=(QuizAttempt.score_total + 1))
        )

    db.flush()

    #  해설 반환
    explanation = _get_explanation(db, question_id)
    return (is_ok, correct_idx, explanation, 1 if is_ok else 0)


def _get_correct_index(db: Session, question_id: int) -> int:
    """정답 인덱스 조회 (label A-D 기준 오름차순)"""
    rows = db.execute(
        select(QuizOption)
        .where(QuizOption.question_id == question_id)
        .order_by(QuizOption.label.asc())
    ).scalars().all()

    for i, o in enumerate(rows):
        if o.is_correct:
            return i
    return 0


def _get_option_id_by_index(db: Session, question_id: int, idx: int) -> Optional[int]:
    """인덱스로 option_id 조회"""
    rows = db.execute(
        select(QuizOption.id)
        .where(QuizOption.question_id == question_id)
        .order_by(QuizOption.label.asc())
    ).scalars().all()

    if 0 <= idx < len(rows):
        return rows[idx]
    return None


def _get_explanation(db: Session, question_id: int) -> Optional[str]:
    """문항의 해설 조회"""
    q = db.get(QuizQuestion, question_id)
    return q.correct_text if q else None


def _index_of_option(db: Session, question_id: int, option_id: Optional[int]) -> int:
    """option_id를 인덱스로 변환"""
    if option_id is None:
        return -1

    rows = db.execute(
        select(QuizOption.id)
        .where(QuizOption.question_id == question_id)
        .order_by(QuizOption.label.asc())
    ).scalars().all()

    try:
        return list(rows).index(option_id)
    except ValueError:
        return -1


# ---------- 결과 조회 ----------
def read_result(db: Session, attempt_id: int) -> Dict:
    """
    퀴즈 결과 조회

    Args:
        db: DB 세션
        attempt_id: 시도 ID

    Returns:
        결과 딕셔너리
    """
    attempt = db.get(QuizAttempt, attempt_id)
    if not attempt:
        return {}

    # 실제 응시한 문항 조회
    resp_rows = db.execute(
        select(QuizResponse, QuizQuestion.stem, QuizQuestion.correct_text)
        .join(QuizQuestion, QuizQuestion.id == QuizResponse.question_id)
        .where(QuizResponse.attempt_id == attempt_id)
    ).all()

    items = []
    correct_cnt = 0

    for resp, stem, explanation in resp_rows:
        cidx = _get_correct_index(db, resp.question_id)
        ok = bool(resp.is_correct)

        if ok:
            correct_cnt += 1

        #  선택지 조회 (옵션)
        choices = _get_choices(db, resp.question_id)

        items.append({
            "itemId": str(resp.question_id),
            "yourChoice": _index_of_option(db, resp.question_id, resp.option_id),
            "correctIndex": cidx,
            "correct": ok,
            "earnedPoints": 1 if ok else 0,
            "question": stem,
            "choices": choices,  #  선택지 추가
            "explanation": explanation,  #  해설 추가
        })

    return {
        "resultId": str(attempt_id),
        "total": len(items),
        "correctCount": correct_cnt,
        "earnedPointsTotal": correct_cnt,  # 1점 체계
        "items": items,
    }


def _get_choices(db: Session, question_id: int) -> List[str]:
    """문항의 선택지 조회"""
    rows = db.execute(
        select(QuizOption.text)
        .where(QuizOption.question_id == question_id)
        .order_by(QuizOption.label.asc())
    ).scalars().all()

    return list(rows)
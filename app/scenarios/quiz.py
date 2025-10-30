# app/routers/quiz.py
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Header
from typing import Optional, List, Dict
from uuid import uuid4
from datetime import datetime, timedelta

from app.deps import get_current_user
from app.schemas.common import User
from app.schemas.quiz import (
    Keyword, QuizItem, QuizGetResponse,
    SubmitAnswerRequest, SubmitAnswerResponse,
    ResultItem, QuizResultResponse,
    POINTS_ON_CORRECT, QUIZ_SESSION_TTL_MIN
)

# 반드시 네 엔진과 프롬프트 사용
try:
    # StepByStep_AI/scenarios/{service.py, keyword_rules.py}
    from scenarios import service as scenario_service  # type: ignore
    from scenarios import keyword_rules               # type: ignore
except Exception as e:
    raise RuntimeError(f"[INIT] scenarios 모듈 로드 실패: {e}")

try:
    # utils/prompts.py 의 시나리오 전용 프롬프트 (필요시 사용)
    from utils.prompts import SCENARIO_PROMPT, USER_TMPL  # type: ignore
except Exception as e:
    raise RuntimeError(f"[INIT] utils.prompts 로드 실패: {e}")

router = APIRouter()

# -------------------------
# Service Resolver (단일 정의)
# -------------------------
_SVC = None

def _resolve_svc():
    # 1) 명시 인스턴스(svc/engine)가 이미 있으면 사용
    for name in ("svc", "engine"):
        inst = getattr(scenario_service, name, None)
        if inst is not None:
            return inst

    # 2) 팩토리 우선(get_service 등)
    for fn in ("get_service", "build_service", "create_service", "make_service", "init_service"):
        f = getattr(scenario_service, fn, None)
        if callable(f):
            return f()

    # 3) 클래스 직접 생성 (Config 있으면 같이 전달)
    for cls_name in ("ScenarioService", "Service"):
        cls = getattr(scenario_service, cls_name, None)
        if cls is not None:
            Cfg = getattr(scenario_service, "Config", None)
            cfg = Cfg() if callable(Cfg) else None
            try:
                return cls(cfg) if cfg is not None else cls()
            except TypeError:
                try:
                    return cls(config=cfg) if cfg is not None else cls()
                except Exception:
                    pass

    # 4) 최후: 모듈 함수 make_quiz/grade_answer 사용
    if hasattr(scenario_service, "make_quiz") and hasattr(scenario_service, "grade_answer"):
        return scenario_service

    raise RuntimeError(
        "scenarios.service 에서 엔진 인스턴스를 찾을 수 없습니다. "
        "svc/engine/get_service/ScenarioService 중 하나를 제공하세요."
    )

def _get_svc():
    global _SVC
    if _SVC is None:
        _SVC = _resolve_svc()
    return _SVC

# -------------------------
# 인메모리 저장소(데모)
# -------------------------
class _Session:
    def __init__(self, quiz_id: str, user_id: int, items: List[QuizItem], answers: Dict[str, int]):
        self.quiz_id = quiz_id
        self.user_id = user_id
        self.items = {it.itemId: it for it in items}
        self.answers = answers                 # itemId -> 정답 인덱스
        self.answered: Dict[str, int] = {}     # itemId -> 사용자가 고른 인덱스
        self.created_at = datetime.utcnow()

_sessions: Dict[str, _Session] = {}  # quizId -> session
_balances: Dict[int, int] = {}       # userId -> 포인트 (데모용)

def _get_balance(uid: int) -> int:
    return _balances.get(uid, 0)

def _add_points(uid: int, delta: int) -> int:
    _balances[uid] = _get_balance(uid) + delta
    return _balances[uid]

def _ensure_alive(sess: _Session):
    if datetime.utcnow() - sess.created_at > timedelta(minutes=QUIZ_SESSION_TTL_MIN):
        raise HTTPException(410, detail={"error": {"code": "SESSION_EXPIRED", "message": "Quiz session expired"}})

# -------------------------
# 엔진 어댑터
# -------------------------
def _engine_get_keywords(q: Optional[str], limit: int) -> List[Keyword]:
    # keyword_rules: get_keywords() 또는 KEYWORDS/keywords 상수 지원
    if hasattr(keyword_rules, "get_keywords"):
        raw = keyword_rules.get_keywords()
    elif hasattr(keyword_rules, "KEYWORDS"):
        raw = getattr(keyword_rules, "KEYWORDS")
    elif hasattr(keyword_rules, "keywords"):
        raw = getattr(keyword_rules, "keywords")
    else:
        raise HTTPException(
            500,
            detail={"error": {"code": "ENGINE_MISSING", "message": "keyword_rules.get_keywords/KEYWORDS 둘 다 없음"}}
        )

    items: List[Keyword] = []
    for it in raw:
        if isinstance(it, dict):
            key = it.get("key") or it.get("id") or it.get("name")
            label = it.get("label") or it.get("name") or key
            sample = it.get("sampleTopics") or it.get("samples") or []
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            key, label = it[0], it[1]
            sample = list(it[2]) if len(it) >= 3 and isinstance(it[2], (list, tuple)) else []
        else:
            key = str(it); label = str(it); sample = []
        if key and label:
            items.append(Keyword(key=key, label=label, sampleTopics=sample))

    if q:
        items = [k for k in items if q in (k.label or "") or q in (k.key or "")]
    return items[:max(1, min(limit, 200))]

def _engine_generate_quiz(mode: str, keyword: Optional[str], n: int) -> tuple[List[QuizItem], Dict[str, int]]:
    svc = _get_svc()
    make_quiz_fn = getattr(svc, "make_quiz", None)
    if make_quiz_fn is None:
        raise HTTPException(500, detail={"error": {"code": "ENGINE_MISSING", "message": "service.make_quiz 미구현"}})

    try:
        data = make_quiz_fn(mode=mode, keyword=keyword, n=n)
    except TypeError:
        data = make_quiz_fn(mode, keyword, n)
    except Exception as e:
        raise HTTPException(500, detail={"error": {"code": "ENGINE_FAIL", "message": str(e)}})

    if not isinstance(data, list):
        raise HTTPException(500, detail={"error": {"code": "ENGINE_PAYLOAD_INVALID", "message": "make_quiz는 리스트를 반환해야 함"}})

    items: List[QuizItem] = []
    answers: Dict[str, int] = {}

    for i, it in enumerate(data):
        if not isinstance(it, dict) or "choices" not in it or "answer_index" not in it:
            raise HTTPException(500, detail={"error": {"code": "ENGINE_ITEM_INVALID", "message": f"{i}번 항목 형식 오류"}})
        item_id = f"it_{i+1:02d}"
        items.append(
            QuizItem(
                itemId=item_id,
                type=it.get("type") or "situation",
                question=it.get("question") or "",
                choices=it.get("choices") or [],
                references=it.get("sources") or None,
            )
        )
        answers[item_id] = int(it.get("answer_index", -1))

    return items, answers

def _engine_grade(quiz_id: str, item_id: str, choice_index: int, sess: _Session) -> tuple[bool, int, str]:
    svc = _get_svc()
    grade_fn = getattr(svc, "grade_answer", None)
    if grade_fn is None:
        raise HTTPException(500, detail={"error": {"code": "ENGINE_MISSING", "message": "service.grade_answer 미구현"}})
    try:
        res = grade_fn(quiz_id=quiz_id, item_id=item_id, choice_index=choice_index)
    except TypeError:
        res = grade_fn(quiz_id, item_id, choice_index)
    except Exception as e:
        raise HTTPException(500, detail={"error": {"code": "ENGINE_FAIL", "message": str(e)}})

    try:
        return bool(res["correct"]), int(res["correctIndex"]), str(res.get("explanation") or "")
    except Exception:
        raise HTTPException(500, detail={"error": {"code": "ENGINE_PAYLOAD_INVALID", "message": "grade_answer 반환 형식 오류"}})

# -------------------------
# 유틸: 인증(게이트웨이/X-User-Id or JWT)
# -------------------------
def _resolve_uid(x_user_id: Optional[str], user: Optional[User]) -> int:
    if user is not None:
        return user.userId
    if x_user_id:
        try:
            return int(x_user_id)
        except ValueError:
            raise HTTPException(401, detail={"error": {"code": "UNAUTHORIZED", "message": "invalid X-User-Id"}})
    raise HTTPException(401, detail={"error": {"code": "UNAUTHORIZED", "message": "need X-User-Id or JWT"}})

# -------------------------
# 1) 키워드 목록  GET /api/quiz/keywords
# -------------------------
@router.get("/keywords", response_model=Dict[str, List[Keyword]])
def get_keywords(
    q: Optional[str] = None,
    limit: int = 50,
    x_user_id: Optional[str] = Header(default=None, convert_underscores=False),
    user: Optional[User] = Depends(get_current_user),
):
    _ = _resolve_uid(x_user_id, user)  # 인증 요구(명세 O)
    items = _engine_get_keywords(q, limit)
    return {"items": items}

# -------------------------
# 2) 퀴즈 세트 생성  GET /api/quiz
# -------------------------
@router.get("", response_model=QuizGetResponse)
def create_quiz(
    mode: str = Query(..., description="by_keyword | random"),
    keyword: Optional[str] = None,
    n: int = Query(5, ge=1, le=10),
    x_user_id: Optional[str] = Header(default=None, convert_underscores=False),
    user: Optional[User] = Depends(get_current_user),
):
    uid = _resolve_uid(x_user_id, user)

    if mode not in {"by_keyword", "random"}:
        raise HTTPException(400, detail={"error": {"code": "INVALID_PARAM", "message": "mode must be by_keyword|random"}})
    if mode == "by_keyword" and not keyword:
        raise HTTPException(400, detail={"error": {"code": "INVALID_PARAM", "message": "keyword required"}})

    items, answers = _engine_generate_quiz(mode, keyword, n)
    quiz_id = f"qz_{uuid4().hex[:6]}"
    _sessions[quiz_id] = _Session(quiz_id, uid, items, answers)

    return QuizGetResponse(quizId=quiz_id, mode=mode, keyword=keyword, total=len(items), items=items)

# -------------------------
# 3) 보기 선택 및 제출  POST /api/quiz/answer
# -------------------------
@router.post("/answer", response_model=SubmitAnswerResponse)
def submit_answer(
    req: SubmitAnswerRequest,
    x_user_id: Optional[str] = Header(default=None, convert_underscores=False),
    user: Optional[User] = Depends(get_current_user),
):
    uid = _resolve_uid(x_user_id, user)

    sess = _sessions.get(req.quizId)
    if not sess or sess.user_id != uid:
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "Quiz not found"}})

    _ensure_alive(sess)

    if req.itemId not in sess.items:
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "Item not found"}})

    if req.itemId in sess.answered:
        raise HTTPException(409, detail={"error": {"code": "ALREADY_ANSWERED", "message": "This item was already submitted"}})

    correct, correct_idx, explanation = _engine_grade(req.quizId, req.itemId, req.choiceIndex, sess)
    sess.answered[req.itemId] = req.choiceIndex

    earned = POINTS_ON_CORRECT if correct else 0
    balance = _add_points(uid, earned) if earned else _get_balance(uid)

    return SubmitAnswerResponse(
        correct=correct,
        correctIndex=correct_idx,
        explanation=explanation or None,
        earnedPoints=earned,
        balance=balance,
        resultId=req.quizId
    )

# -------------------------
# 4) 결과 조회  GET /api/quiz/results/{resultId}
# -------------------------
@router.get("/results/{resultId}", response_model=QuizResultResponse)
def get_result(
    resultId: str = Path(..., description="= quizId"),
    x_user_id: Optional[str] = Header(default=None, convert_underscores=False),
    user: Optional[User] = Depends(get_current_user),
):
    uid = _resolve_uid(x_user_id, user)

    sess = _sessions.get(resultId)
    if not sess or sess.user_id != uid:
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "Result not found"}})

    items: List[ResultItem] = []
    total = len(sess.items)
    correct_cnt = 0
    earned_total = 0

    for iid, it in sess.items.items():
        your = sess.answered.get(iid, -1)
        cidx = sess.answers[iid]
        ok = (your == cidx)
        earned = POINTS_ON_CORRECT if ok else 0
        if ok:
            correct_cnt += 1
            earned_total += earned

        items.append(ResultItem(
            itemId=iid, yourChoice=your, correctIndex=cidx, correct=ok, earnedPoints=earned,
            question=it.question, choices=it.choices, explanation=None
        ))

    return QuizResultResponse(
        resultId=resultId,
        total=total,
        correctCount=correct_cnt,
        earnedPointsTotal=earned_total,
        items=items
    )
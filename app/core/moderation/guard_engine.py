# app/core/moderation/guard_engine.py
from __future__ import annotations
import re, os
from typing import Dict, Any, List, Tuple
import time, json, hashlib, pathlib
import unicodedata
from openai import OpenAI
from pydantic import BaseModel
from app.core.moderation.moderation import Decision  # 모델만 참조 (순환 import 방지)

# =========================
# 설정 & 공용 메시지
# =========================
_OAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("MODERATION_MODEL", "omni-moderation-latest")
DEFAULT_CHANNEL = "generic"

SAFE_FALLBACK = (
    "내용 일부가 안전하지 않아 요약·완화된 표현으로 대체했어.\n"
    "위험하거나 불편한 상황이면 보건교사·믿을 수 있는 어른과 상의하고,\n"
    "긴급 시 112, 상담은 청소년전화 1388/자살예방 1393로 바로 도움받아줘."
)

BLOCK_REPLY = (
    "이 요청은 안전 정책상 처리할 수 없어. "
    "대신 건강·안전 관점에서 궁금한 점을 구체적으로 물어봐줘.\n"
    "긴급하면 112, 상담은 1388/1393에 연락하자."
)

# =========================
# 규칙: 패턴/정책
# =========================
CATEGORY_MAP = {
    "sexual.explicit":  r"(야동|포르노|노골적|섹스\s*자세|오르가즘\s*(늘|키)|체위|야사|야짤)",
    "sexual.minors":    r"(미성년자.*성관계|학생과\s*성관계|청소년.*음란|n\s*번방)",
    "sexual.services":  r"(조건\s*만남|조건만남|매춘|성\s*매매|유흥알바)",
    "self_harm":        r"(자해(하는|방법)|죽고\s*싶|극단적\s*선택)",
    "violence.assault": r"(강간|성폭행|데이트\s*폭력|스쿨\s*폭력)",
    "illegal.drugs":    r"(필로폰|메스암페타민|대마초|마약).*(사는|파는|구매|판매)",
    "pii.contact":      r"((\b010[-\s]?\d{4}[-\s]?\d{4}\b)|카톡\s*아이디|인스타\s*@\w+|디엠\s*주세요|주\s*소:\s*)",
    "medical.hazard":   r"(약\s*처방\s*없이\s*구매|과다\s*복용)",
}

# 명시 차단 패턴 (요청형 불법/미성년자/자해 방법 등)
BLOCK_PATTERNS = [
    CATEGORY_MAP["sexual.services"],
    r"(미성년자.*(만나|관계|성관계)|학생과\s*성관계)",
    CATEGORY_MAP["illegal.drugs"],
    r"(자해하는\s*법|자살하는\s*법)",
    r"(불법\s*촬영|몰카).*(해|방법|하는\s*법)",
]

# 피해·신고/상담 의도 화이트리스트 (허용 + 안전 안내 유도)
VICTIM_WHITELIST = [
    r"(성폭행|강간|강제추행|불법\s*촬영|디지털\s*성범죄|스토킹|협박|학대).*(당했|피해|신고|도움)",
    r"(피해).*(성폭력|불법\s*촬영|스토킹|학대|협박)",
]

# 미성년자 표현(다양한 표기 지원)
_MINOR_RE_LIST = [
    re.compile(r"\b(1[0-7])\s*살\b"),                # 10~17살
    re.compile(r"\b(만\s*1[0-7]\s*세)\b"),          # 만 14세 등
    re.compile(r"(초등학생|중학생|고등학생|미성년|미자)"),
    re.compile(r"(중[1-3]|고[1-3])"),
    # 붙임말 변형
    re.compile(r"(1[0-7])\s*살인데"),
    re.compile(r"(1[0-7])살인데"),
]

# 성적 행위/요청 의도 (은어 포함, 필요시 확장)
_SEXUAL_INTENT_RE = re.compile(
    r"(성관계|관계해|관계하자|섹스|ㅅㅅ|야스|자세히\s*알려줘|자세히\s*가르쳐|만나서\s*하자|야한\s*사진|수위\s*사진|노출)",
    re.I
)

# 교육 의도 화이트리스트(모더 호출 전 선허용, 단 명확 금지패턴 예외)
EDU_WHITELIST = [
    r"(콘돔|피임약|IUD|임플란트|성병|성매개감염|동의|의사소통|피임|검사|상담|생리|월경|주기|성교육|예방|부작용|보건|안전)"
]
# 채널별 임계치
_POLICY: Dict[str, Dict[str, float]] = {
    "chat_input":   {"sexual/minors": 0.5, "self-harm": 0.3},
    "snippet":      {"sexual/minors": 0.3, "sexual/explicit": 0.4},
    "chat_output":  {"sexual/minors": 0.2, "sexual/explicit": 0.3},
    "post":         {"sexual/minors": 0.2, "sexual/explicit": 0.3},
    "comment":      {"sexual/minors": 0.2, "sexual/explicit": 0.3},
}
# URL/파일 링크 탐지 (확장자 포함)
URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b)",  # URL/이메일
    re.IGNORECASE,
)

FILE_EXT_RE = re.compile(
    r"\.(?:jpg|jpeg|png|gif|bmp|webp|mp4|avi|mkv|mov|zip|rar|7z|tgz|gz|apk|exe|torrent)\b",
    re.IGNORECASE,
)

MOD_LOG_FILE = os.getenv("MOD_LOG_FILE", "logs/moderation.log.jsonl")

# =========================
# 유틸
# =========================
def _normalize(s: str) -> str:
    return (s or "").replace("\u200b", "").replace("\ufeff", "").strip()

def _match_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

def _contains(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

def _is_educational_context(text: str) -> bool:
    t = _normalize(text)
    return _match_any(EDU_WHITELIST, t)

def _is_victim_context(text: str) -> bool:
    t = _normalize(text)
    return _match_any(VICTIM_WHITELIST, t)

def _soft_sanitize(text: str) -> str:
    # (필요시 사용) 과도한 표현을 완곡어로 대체
    text = re.sub(r"(야동|포르노)", "유해한 성적 콘텐츠", text, flags=re.I)
    text = re.sub(r"(체위|자세)", "노골적 행위 묘사", text, flags=re.I)
    text = re.sub(CATEGORY_MAP["pii.contact"], "(연락처 등 민감정보 제거)", text, flags=re.I)
    return text

def _normalize_full(s: str) -> str:
    # 숫자/문자 혼용, 전각/반각, 제로폭 문자 등 정규화
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u200b", "").replace("\ufeff", "")
    return s.strip()

def _strip_spaces(s: str) -> str:
    return re.sub(r"\s+", "", s)

def _mentions_minor(s_norm: str, s_compact: str) -> bool:
    return any(rx.search(s_norm) or rx.search(s_compact) for rx in _MINOR_RE_LIST)

def _is_minor_sexual_request(text: str) -> bool:
    """미성년자 언급 + 성적 행위 유도/요청이 함께 있을 때 True"""
    s = _normalize_full(text)
    s_compact = _strip_spaces(s)
    return _mentions_minor(s, s_compact) and (
        _SEXUAL_INTENT_RE.search(s) or _SEXUAL_INTENT_RE.search(s_compact)
    )
def _ensure_parent(path: str):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def _redact(s: str, keep: int = 120) -> str:
    s = (s or "").replace("\n", " ")
    return s[:keep] + ("…" if len(s) > keep else "")

def _log_moderation(*, text: str, decision: Decision, channel: str, user_id: str | None = None):
    """JSONL 로깅 (원문 저장하지 않고, 앞부분 120자만 부분 저장 + hash)"""
    try:
        _ensure_parent(MOD_LOG_FILE)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "channel": channel,
            "userId": user_id,
            "action": decision.action,
            "reasons": decision.reasons,
            "categories": decision.categories,
            "scores": decision.scores,
            "text_hash": _sha1(text),
            "text_head": _redact(text, 120),
        }
        with open(MOD_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # 로깅 에러는 무시 (서비스 흐름 방해 X)
        pass

# =========================
# OpenAI Moderation 호출
# =========================
def _openai_moderate(text: str) -> Tuple[Dict[str, bool], Dict[str, float], bool]:
    resp = _OAI.moderations.create(model=_MODEL, input=text)
    res = resp.results[0]
    cats: Dict[str, bool] = dict(res.categories)
    scs: Dict[str, float] = dict(res.category_scores)
    flagged: bool = bool(res.flagged)
    return cats, scs, flagged

# =========================
# 점수/카테고리 → 액션
# =========================
def _pick_action(categories: Dict[str, bool], scores: Dict[str, float], *, channel: str) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    def on(cat): return bool(categories.get(cat, False))
    def sc(cat): return float(scores.get(cat, 0.0))

    # 강력 차단
    if on("sexual/minors"):
        reasons.append("미성년자 관련 성적 콘텐츠"); return "block", reasons
    if on("self-harm") or on("violence/graphic") or on("hate/threatening"):
        reasons.append("자해/타해·폭력 선동/협박"); return "block", reasons

    # 성적 노골성(성인) – 채널별
    if on("sexual/explicit") or sc("sexual/explicit") >= 0.5:
        reasons.append("노골적 성적 표현")
        return ("block", reasons) if channel in ("post", "comment") else ("soft_block", reasons)

    # 불법/마약 점수 높으면 차단
    if sc("illicit") >= 0.5 or sc("illicit/violent") >= 0.4:
        reasons.append("불법/유해 요청"); return "block", reasons

    # 욕설/혐오(경미)
    if on("harassment") or on("hate") or sc("harassment") >= 0.4:
        reasons.append("욕설/비하 표현"); return "soft_block", reasons

    # 채널 임계치 기반 soft
    th = _POLICY.get(channel, {})
    for k, t in th.items():
        if scores.get(k, 0.0) >= t:
            reasons.append(f"{k}:{scores[k]:.2f}>={t:.2f}")
    if reasons:
        return "soft_block", reasons

    return "allow", reasons

# =========================
# 공개 API: guard_text
# =========================
def guard_text(text: str, *, channel: str = DEFAULT_CHANNEL, user_id: str | None = None) -> Decision:
    """
    단일 진입점:
      0) 피해/신고 의도면 허용(안내 유도)
      1) 명시 차단 패턴 즉시 block
      2) 교육 의도면 선허용 (단, 명확 금지 패턴 예외)
      3) OpenAI moderation → 채널별 정책
    """
    t = _normalize_full(text)
    decision: Decision | None = None

    # 0a) 미성년자+성적행위 선차단 (화이트리스트보다 우선) ⬅ NEW
    if _is_minor_sexual_request(t):
        return Decision(
            action="block",
            reasons=["sexual_minors_intent"],
            categories={},
            scores={}
        )

    # 0b) 피해 신고/상담 의도 → 허용(후속 단계에서 안전 안내)
    if _is_victim_context(t):
        decision = Decision(action="allow", reasons=["victim_context_allow"], categories={}, scores={})
        _log_moderation(text=text, decision=decision, channel=channel, user_id=user_id)
        return Decision(action="allow", reasons=["victim_context_allow"], categories={}, scores={})

    # 0c) URL/파일 링크 감지
    if URL_RE.search(t) or FILE_EXT_RE.search(t):
        reason = "url_or_file_link"
        if channel in ("post", "comment"):  # 게시판은 보수적으로 차단
            decision = Decision(action="block", reasons=[reason], categories={}, scores={})
        else:  # 챗봇 입력/출력/스니펫은 완화
            decision = Decision(action="soft_block", reasons=[reason], categories={}, scores={})
        _log_moderation(text=text, decision=decision, channel=channel, user_id=user_id)
        return decision

    # 1) 명시 차단 패턴
    if _match_any(BLOCK_PATTERNS, t):
        return Decision(action="block", reasons=["explicit_block_pattern"], categories={}, scores={})

    # 2) 교육 의도 선허용
    if _is_educational_context(t):
        return Decision(action="allow", reasons=["edu_whitelist_pre"], categories={}, scores={})

    # 3) OpenAI moderation
    try:
        cats, scs, _ = _openai_moderate(t)
        action, reasons = _pick_action(cats, scs, channel=channel)
        return Decision(action=action, reasons=reasons, categories=cats, scores=scs)
    except Exception as e:
        # API 오류 시 보수적으로 hold
        return Decision(action="hold", reasons=[f"moderation_error: {e.__class__.__name__}"], categories={}, scores={})

# 외부에서 import할 수 있게 alias 제공
def is_educational_intent(text: str) -> bool:
    """외부용: 교육 의도 문장 여부 판단"""
    return _is_educational_context(text)

# =========================
# (옵션) 배치용 타입 (필요 시)
# =========================
class GuardDecision(BaseModel):
    action: str                 # allow | soft_block | hold | block
    reasons: List[str] = []
    categories: Dict[str, Any] = {}
    scores: Dict[str, float] = {}
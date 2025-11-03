# utils/cleaning.py
import re, hashlib, unicodedata
from typing import Tuple, Iterable, List, Dict, Set

# ── 간단 영어/한국어 불용어 (필요 시 확장)
EN_STOPWORDS: Set[str] = {
    "the","a","an","of","to","in","and","or","for","on","at","by","with","is","are",
    "was","were","be","been","being","this","that","those","these","it","its","as",
    "from","but","not","no","if","then","than","so","such","into","about","over",
    "can","could","should","would","may","might","will","shall","do","does","did",
    "have","has","had","i","you","he","she","we","they","them","their","our","your"
}
KO_STOPWORDS: Set[str] = {
    "그리고","또한","그러나","하지만","즉","및","등","또","더","점","수","것","등등","때","중",
    "이","그","저","는","은","이란","이라는","의","가","을","를","도","로","에서","에게","까지","부터",
    "하다","했다","된다","된다면","때문","위해","대한","대해","하지","않다","있다","없다"
}

BULLETS = r"[■□◆◇●○▪︎▸►•·※★☆▶▷◀◁❖❗️❕✱†‡•]+"
ZWS = r"[\u200B-\u200F\uFEFF\u2060]"

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# ── 라인 유틸
def _iter_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        # 좌우 공백만 우선 제거(중간 공백은 후처리)
        yield line.strip("\r").rstrip()

def _fix_hyphen_join(lines: Iterable[str]) -> Iterable[str]:
    """줄끝 하이픈으로 끊긴 영단어/숫자 복원: 'develop-\nment' -> 'development'"""
    buf = ""
    for line in lines:
        if buf:
            line = buf + line.lstrip()
            buf = ""
        if re.search(r"[A-Za-z0-9]-$", line) and not line.endswith("--"):
            buf = line[:-1]  # 하이픈 제거 후 다음 줄과 결합
            continue
        yield line
    if buf:
        yield buf

def _strip_headers_footers(lines: Iterable[str]) -> Iterable[str]:
    """
    페이지 번호/머릿말/꼬릿말/URL/라인 장식/TOC 잔재 제거
    (OCR/교재 공통 노이즈 패턴 보강)
    """
    page_num = re.compile(r"^\s*(page\s*)?\d{1,4}\s*$", re.I)
    url = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")
    dashes = re.compile(r"^[-=_~]{3,}$")
    header_like = re.compile(r"^\s*(chapter\s+\d+|contents?|table of contents|index)\s*$", re.I)
    copyright_like = re.compile(r"©|copyright", re.I)

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if (page_num.match(s) or dashes.match(s) or url.search(s)
            or header_like.match(s) or copyright_like.search(s)):
            continue
        yield line

def _normalize_unicode(text: str) -> str:
    # 전각/호환 문자 정규화
    text = unicodedata.normalize("NFKC", text)
    # 비표시 제로폭 제거
    text = re.sub(ZWS, "", text)
    # 흔한 OCR 따옴표 통일
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    # OCR 특수 잔재 정리
    text = text.replace("BO0I", "").replace("BOOI", "")
    # 중복 공백 축소
    text = re.sub(r"[ \t]+", " ", text)
    # 윈도우 개행 정리
    text = re.sub(r"\r\n|\r", "\n", text)
    return text

def _drop_short_noise(lines: Iterable[str], min_len: int = 2) -> Iterable[str]:
    for line in lines:
        s = line.strip()
        if len(s) < min_len:
            continue
        # 알파벳/숫자/한글이 하나도 없으면 제거
        if not re.search(r"[A-Za-z0-9가-힣]", s):
            continue
        yield line

def _collapse_bullets(text: str) -> str:
    # 과도한 불릿들은 ' • '로 축약
    text = re.sub(BULLETS, " • ", text)
    # 불릿 뒤 과도 공백 정리
    text = re.sub(r"(?:\s*•\s*){2,}", " • ", text)
    return text

# ── 불용어 기반 토크나이즈 (BM25 참고/정제 통계)
def _tokenize_en(text: str) -> List[str]:
    toks = []
    for w in re.findall(r"[A-Za-z]+", text.lower()):
        if len(w) <= 1 or w in EN_STOPWORDS:
            continue
        toks.append(w)
    return toks

def _tokenize_ko(text: str) -> List[str]:
    toks = []
    for w in re.findall(r"[가-힣]{2,}", text):
        if w in KO_STOPWORDS:
            continue
        toks.append(w)
    return toks

def guess_lang(text: str) -> str:
    en = len(re.findall(r"[A-Za-z]", text))
    ko = len(re.findall(r"[가-힣]", text))
    if ko and en:
        return "mixed" if abs(ko - en) / max(1, ko + en) < 0.7 else ("ko" if ko > en else "en")
    return "ko" if ko > 0 else "en"

# ── 공개 API
def clean_text(raw: str) -> Tuple[str, Dict]:
    """
    OCR 텍스트 정제(강화):
      - 유니코드 정규화(NFKC), 제로폭·이상문자 제거
      - 하이픈 줄바꿈 복원
      - 머릿/꼬릿말, 페이지번호, URL, 라인장식, 저작권 문구 제거
      - 불릿/잡기호 축약, 공백/개행 정리
      - 3회 이상 연속 개행을 2회로 축소
    반환: (정제 텍스트, 통계 dict)
    """
    original_len = len(raw)
    text = _normalize_unicode(raw)
    lines = list(_iter_lines(text))
    lines = list(_fix_hyphen_join(lines))
    lines = list(_strip_headers_footers(lines))
    lines = list(_drop_short_noise(lines))
    cleaned = "\n".join(lines)
    cleaned = _collapse_bullets(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    info: Dict = {
        "orig_len": original_len,
        "clean_len": len(cleaned),
        "ratio": round((len(cleaned) / original_len) if original_len else 1.0, 4),
        "en_tokens": len(_tokenize_en(cleaned)),
        "ko_tokens": len(_tokenize_ko(cleaned)),
        "lines": len(lines),
        "lang_guess": guess_lang(cleaned),
    }
    return cleaned, info

def clean_ocr_text(raw: str) -> str:
    cleaned, _ = clean_text(raw)
    return cleaned

def clean_for_bm25(raw: str) -> str:
    en = _tokenize_en(raw)
    ko = _tokenize_ko(raw)
    return " ".join(en + ko)
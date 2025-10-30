# SCSC/scenario/service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import Counter, deque
from pathlib import Path
import json, random, re, os, time, inspect

from openai import OpenAI

from utils.prompts import SCENARIO_PROMPT, USER_TMPL
from scenarios.keyword_rules import match_keywords

from utils.faiss_store import FaissStore
from utils.bm25_store import BM25Store


# -------------------------------
# OpenAI 호출 헬퍼 (SDK 버전 안전)
# -------------------------------
def _supports_response_format(client) -> bool:
    try:
        sig = inspect.signature(client.responses.create)
        return "response_format" in sig.parameters
    except Exception:
        return False


def call_llm_json(client: OpenAI, prompt: str, model: str, temperature: float = 0.0) -> str:
    """
    가능한 경우 responses + response_format 사용,
    아니면 responses, 마지막 폴백으로 chat.completions 사용.
    항상 'JSON 문자열'을 반환.
    """
    # 1) responses + response_format
    try:
        if hasattr(client, "responses") and _supports_response_format(client):
            r = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return getattr(r, "output_text", None) or r.output[0].content[0].text.value

        # 2) responses (response_format 미지원)
        if hasattr(client, "responses"):
            r = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
            )
            return getattr(r, "output_text", None) or r.output[0].content[0].text.value
    except TypeError:
        pass
    except Exception:
        pass

    # 3) chat.completions 폴백
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return r.choices[0].message.content or "{}"


# -------------------------------
# 개념 토픽 풀
# -------------------------------
CONCEPT_MAP: Dict[str, List[str]] = {
    "성병": [
        "HPV 정의", "HPV 유형", "HPV 백신",
        "헤르페스 특징", "헤르페스 증상", "헤르페스 재발",
        "클라미디아 증상", "클라미디아 무증상 가능성",
        "임질 증상",
        "매독 1기 특징", "매독 2기 특징", "매독 3기 특징",
        "HIV 전파", "HIV 검사", "HIV 치료",
        "B형간염 전파", "B형간염 예방접종",
        "트리코모나스 특징", "트리코모나스 치료",
        "무증상 가능성", "잠복기 개념", "검사 권장 시점",
        "성병 감염 경로", "성병 전파 방식",
        "성병 검사 방법", "성병 검사 주기",
        "자가키트 활용", "익명검사 활용",
        "파트너 통보", "동시 치료", "재감염 가능성", "치료 완료 기준",
        "콘돔의 성병 예방 효과",
        "구강 성교 예방(덴탈댐)", "항문 성교 예방(콘돔)",
        "백신으로 예방 가능한 감염",
        "목욕탕 전파 오해", "화장실 좌변기 전파 오해",
        "키스만으로 전파 오해", "항생제 남용 위험", "항생제 내성 위험",
    ],
    "피임": [
        "콘돔 개봉", "콘돔 꼭지 공기 빼기", "콘돔 착용 순서", "콘돔 탈착",
        "콘돔 보관법", "콘돔 파손 원인",
        "질외사정 실패율", "질외사정 전분비액 위험",
        "사후피임약 복용 시점", "사후피임약 효과", "사후피임약 부작용",
        "경구피임약 복용법", "경구피임약 복용 누락 대처", "경구피임약 부작용",
        "IUD 구리 장단점", "IUD 호르몬 장단점", "IUD 부작용",
        "피임 패치 작용", "피임 패치 대상",
        "피임 주사 작용", "피임 주사 대상",
        "피임 임플란트 작용", "피임 임플란트 대상",
        "질정 사용법", "질정 실패율",
        "다이어프램 사용법", "다이어프램 실패율",
        "피임 실패 시 응급피임", "피임 실패 시 임신 가능성 평가", "피임 실패 상담",
        "피임과 성병 예방의 차이", "이중 보호", "피임 의사소통 전략",
    ],
    "생리": [
        "월경 주기", "배란", "개인차 원인", "불규칙 원인",
        "가임기 계산 한계", "가임기 계산 오해",
        "월경통 자기관리", "월경통 약물", "월경통 경고 신호",
        "PMS 특징", "PMDD 특징", "PMS/PMDD 대처",
        "생리대 사용", "탐폰 사용", "생리컵 사용",
        "교체 주기", "위생 관리",
        "스팟팅 원인", "주기 변화 원인", "스트레스 영향", "체중 변화 영향", "약물 영향",
        "초경 안내", "사춘기 변화",
        "수영 시 용품 선택", "체육 시 용품 선택",
        "과다 월경 상담 기준", "과소 월경 상담 기준",
    ],
    "경계/동의": [
        "동의 원칙: 자유", "동의 원칙: 명확성", "동의 원칙: 구체성", "동의 원칙: 가역성",
        "취중 동의 무효", "압박 관계 동의 무효", "권력관계 동의 무효",
        "경계 설정 방법", "의사표현 문장 예시",
        "거절 뒤 대화", "관계 존중",
        "디지털 동의: 사진 촬영", "디지털 동의: 영상 공유",
        "동의의 지속성", "동의의 철회",
    ],
    "관계/의사소통": [
        "감정 인식", "감정 표현", "경청 스킬",
        "나-메시지", "비난 대신 구체적 요청",
        "갈등 해결: 사실/감정/요청 분리",
        "연애 의사결정", "연애 상호 존중",
        "개인정보 공유 범위", "비밀보장",
        "질문 스킬", "확인 스킬", "확증 편향 줄이기",
    ],
    "온라인/디지털": [
        "디지털 성범죄: 불법촬영/유포/협박",
        "사진 요구 거절 문장", "차단", "증거 보존",
        "신고 112", "상담 1366", "디지털 성범죄 지원단",
        "2단계 인증", "비밀번호 관리",
        "저작권/초상권 기본", "유해물 신고",
    ],
    "임신/출산": [
        "임신 가능성", "가임기 오해 바로잡기",
        "임신 테스트기 시점/판독",
        "임신 초기 증상/확인 절차",
        "임신중절 정보 접근/상담",
        "의료기관 찾기/비밀보장", "임신 안전",
    ],
    "건강/상담": [
        "학교 보건실/보건교사 활용",
        "청소년 친화 의료기관 찾기",
        "비밀보장/동반자 동의",
        "불안/우울/정신건강과 성",
        "헬프라인 112/1366/위기대응",
        "상담 준비: 증상 기록/질문 리스트",
    ],
    "신체 변화": [
        "사춘기 2차 성징/개인차",
        "음경/고환/포경/몽정/발기",
        "유방 발달/브라 선택",
        "체모/목소리/피부 변화 관리",
        "신체 이미지/자기존중감",
    ],
    "외모/자기이미지": [
        "체중/체형과 건강",
        "다이어트 오해", "여드름/피부 관리",
        "미디어 보정/필터 인식", "외모 괴롭힘 대처",
    ],
    "자위/욕구": [
        "성적 욕구/자위의 정상성",
        "오해 교정", "프라이버시/위생/디지털 안전",
        "콘텐츠 선택/경계 설정",
    ],
}

SCENARIO_BACKGROUNDS = [
    "수업 끝나고 복도에서 대화 중", "동아리 활동 쉬는 시간", "단체 채팅에서 의견을 나누는 중",
    "공원 벤치에서 이야기하는 중", "보건실 상담 대기 중", "온라인 메신저 대화 중",
    "급식 줄에서 잡담 중", "조별 과제 회의 중", "등굣길 버스 안", "도서관 자습 중",
    "체육 시간 팀 활동 전", "학급 게시판 앞", "학교 축제 준비 중",
    "청소년상담복지센터 대기실", "보건소 예약 전화 전", "주말 스터디 카페",
]

SCENARIO_ENDINGS = [
    "너라면 어떻게 할래?", "지금 선택할 행동은 무엇일까?", "어떤 말부터 꺼낼래?",
    "먼저 확인해야 할 것은 무엇일까?", "누구와 상의해볼까?", "가장 안전한 선택은 무엇일까?",
    "네가 취할 수 있는 다음 한 걸음은?", "상대를 존중하면서 뭐라고 말하겠어?",
]

CONCEPT_FORMS = [
    "{topic}는 무엇일까?",
    "다음 중 {topic}의 특징으로 옳은 것은?",
    "{topic}에 대한 설명으로 맞는 것을 골라.",
    "{topic} 예방 또는 관리 방법으로 올바른 것은?",
]

RECENT_MAX = 20


@dataclass
class Config:
    index_root: str = "SCSC/indexes"  # window/ qna 루트 상위
    topk: int = 6
    max_context_chars: int = 1600
    gen_model: str = "gpt-4o-mini"


class ScenarioService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.faiss = self._try_load_faiss()
        self.bm25 = self._try_load_bm25()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._pool = self._load_pool()
        self._keywords = None
        self._recent_questions = deque(maxlen=RECENT_MAX)
        self._topic_cycle: Dict[str, List[str]] = {}

    # ---------- 상태 유틸 ----------
    def _mk_facts(self, snips: List[Dict[str, Any]], max_chars: int = 900) -> str:
        """스니펫을 시험용 FACTS로 압축."""
        facts, used = [], 0
        for s in snips:
            t = (s.get("text") or "").strip()
            if not t:
                continue
            # 문장 1~2개만 추출
            for one in re.split(r"[.!?]\s+", t)[:2]:
                one = " ".join(one.split())
                if len(one) < 12:
                    continue
                if used + len(one) + 3 > max_chars:
                    return "\n".join(f"- {f}" for f in facts)
                facts.append(one)
                used += len(one) + 3
        return "\n".join(f"- {f}" for f in facts) if facts else ""

    @staticmethod
    def _unify_register(text: str) -> str:
        if not isinstance(text, str):
            return text
        t = re.sub(r"요[.!?]?$", "", text)
        t = (t.replace("십시오", "요")
               .replace("하세요", "해")
               .replace("해주세요", "해줘"))
        return re.sub(r"\s+", " ", t).strip()

    @staticmethod
    def _too_similar(a: str, b: str, th: float = 0.7) -> bool:
        ta = set(a.replace(",", " ").replace(".", " ").split())
        tb = set(b.replace(",", " ").replace(".", " ").split())
        if not ta or not tb:
            return False
        j = len(ta & tb) / max(1, len(ta | tb))
        return j >= th

    def _push_recent(self, q: str) -> bool:
        for prev in self._recent_questions:
            if self._too_similar(prev, q):
                return False
        self._recent_questions.append(q)
        return True

    def _next_concept_topic(self, kw: str) -> Optional[str]:
        pool = CONCEPT_MAP.get(kw or "", [])
        if not pool:
            return None
        if kw not in self._topic_cycle or not self._topic_cycle[kw]:
            arr = pool[:]
            random.shuffle(arr)
            self._topic_cycle[kw] = arr
        return self._topic_cycle[kw].pop()

    def _scenario_hint(self) -> str:
        bg = random.choice(SCENARIO_BACKGROUNDS)
        end = random.choice(SCENARIO_ENDINGS)
        return f"[배경] {bg}\n[마무리 질문] {end}"

    # ---------- 디버그용: 근거 보기 ----------
    def show_sources(self, item: Dict[str, Any]) -> str:
        def _src_key(d: Dict[str, Any]) -> str:
            return d.get("source") or d.get("src") or d.get("source_path") or d.get("doc") or ""

        def _cid(v: Any) -> str:
            return str(v) if v is not None else ""

        out = []
        refs = item.get("sources", []) if isinstance(item, dict) else []
        if not refs:
            return "(근거 없음)"

        want = {(_src_key(r), _cid(r.get("chunk_id"))) for r in refs}

        for s in self._pool:
            got_src = _src_key(s)
            got_cid = _cid(s.get("chunk_id"))
            if (got_src, got_cid) in want:
                out.append(f"[{got_src} #{got_cid}] {s.get('text', '')}")
        return "\n".join(out) if out else "(근거 없음)"

    # ---------- 인덱스 로드 ----------
    def _try_load_faiss(self):
        root = Path(self.cfg.index_root)
        candidates = list(root.glob("**/*_mac")) + list(root.glob("**/*_window"))
        for d in candidates:
            try:
                return FaissStore.load(str(d))
            except Exception:
                continue
        return None

    def _try_load_bm25(self):
        root = Path(self.cfg.index_root)
        candidates = list(root.glob("**/*_mac")) + list(root.glob("**/*_window"))
        for d in candidates:
            try:
                return BM25Store.load(str(d))
            except Exception:
                continue
        return None

    def _load_pool(self) -> List[Dict[str, Any]]:
        """인덱스에서 전체 스니펫 풀 1회 구축"""
        pool: List[Dict[str, Any]] = []
        for p in Path(self.cfg.index_root).glob("**/meta.json"):
            arr = json.loads(p.read_text(encoding="utf-8"))
            for row in arr:
                txt = row.get("text") or row.get("chunk_text") or ""
                if not txt:
                    continue
                kws = row.get("keywords") or match_keywords(txt)
                pool.append({
                    "text": " ".join(txt.split()),
                    "source": row.get("src") or row.get("source") or p.parent.name,
                    "chunk_id": row.get("chunk_id"),
                    "keywords": kws,
                })
        return pool

    # ---------- 검색 ----------
    def search(self, query: str, topk: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if self.faiss:
            results += self.faiss.search(query, top_k=topk)
        if self.bm25:
            results += self.bm25.search(query, top_k=topk)
        if not results:
            return []
        # 간단 RRF
        def k(s): return f"{s.get('source')}#{s.get('chunk_id')}"
        scored: Dict[str, float] = {}
        for rank, s in enumerate(results, start=1):
            scored[k(s)] = scored.get(k(s), 0.0) + 1.0 / (60 + rank)
        uniq = {k(s): s for s in results}
        ranked = sorted(uniq.values(), key=lambda s: scored[k(s)], reverse=True)
        return ranked[:topk]

    def random_snippets(self, topk: int) -> List[Dict[str, Any]]:
        if not self._pool:
            return []
        return random.sample(self._pool, k=min(topk, len(self._pool)))

    # ---------- 키워드 ----------
    def keywords(self, limit: int = 40):
        WHITELIST = (
            "피임", "생리", "연애", "외모", "신체 변화",
            "젠더", "관계/의사소통", "경계/동의", "온라인/디지털",
            "성병/검사", "임신/출산", "자위/욕구", "건강/상담",
        )
        c = Counter()
        for s in self._pool:
            for kw in s.get("keywords", ()):
                c[kw] += 1
        ordered = [(k, c[k]) for k in WHITELIST if k in c]
        if not ordered:
            ordered = sorted(c.items(), key=lambda x: x[1], reverse=True)
        return [{"keyword": k, "count": v} for k, v in ordered[:limit]]

    def pick_by_keyword(self, keyword: str, topk: int):
        cand = [s for s in self._pool if keyword in s.get("keywords", [])]
        if not cand:
            return []
        random.shuffle(cand)
        if self.faiss:
            query = f"{keyword} 원칙 개념 예방 검사 특징 사례"
            ranked = self.faiss.search(query, top_k=topk * 3)
            allow = {(s["source"], s.get("chunk_id")) for s in cand}
            ranked = [r for r in ranked if (r.get("source"), r.get("chunk_id")) in allow]
            random.shuffle(ranked)
            if ranked:
                return ranked[:topk]
        return cand[:topk]

    # ---------- 내부 유틸 ----------
    def _concept_snippets(self, keyword: str, topic: str, topk: int) -> List[Dict[str, Any]]:
        def _variants(t: str) -> List[str]:
            t2 = re.sub(r"[\(\)]", " ", t)
            toks = [t, t2]
            if "HPV" in t.upper(): toks += ["HPV", "인유두종", "인유두종바이러스"]
            if "HIV" in t.upper(): toks += ["HIV", "AIDS", "에이즈"]
            return list(dict.fromkeys([re.sub(r"\s+", " ", x).strip() for x in toks if x.strip()]))

        pats = [re.compile(re.escape(v), re.IGNORECASE) for v in _variants(topic)]
        hits = []
        for s in self._pool:
            if keyword and keyword not in (s.get("keywords") or []):
                continue
            txt = s.get("text") or ""
            if any(p.search(txt) for p in pats):
                hits.append(s)
                if len(hits) >= topk:
                    break
        if len(hits) >= max(2, topk // 2):
            random.shuffle(hits)
            return hits[:topk]

        q = f"{topic} 정의 특징 전파 경로 예방 검사 설명 근거"
        found = self.search(q, topk=topk * 2)
        if keyword and found:
            f2 = [r for r in found if keyword in (r.get("keywords") or [])]
            found = f2 or found
        random.shuffle(found)
        return (found or hits or self.random_snippets(topk))[:topk]

    @staticmethod
    def _normalize_choices(choices: List[str]) -> List[str]:
        norm = []
        for c in choices:
            if not isinstance(c, str):
                continue
            cc = re.sub(r"^\s*[A-Da-d]\s*[\.\):]\s*", "", c).strip()
            cc = re.sub(r"\s+", " ", cc)
            if cc:
                norm.append(cc)
        seen, out = set(), []
        for c in norm:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    @staticmethod
    def _looks_bad_choices(choices: List[str]) -> bool:
        if len(choices) != 4:
            return True
        if any(len(c.strip()) < 8 for c in choices):
            return True
        norm = [re.sub(r"\s+", " ", c.strip()) for c in choices]
        if len(set(norm)) < 4:
            return True
        bad_phrases = ["무시하고", "강제로", "즉시 관계", "피임 없이", "아무 준비 없이"]
        if sum(any(p in c for p in bad_phrases) for c in norm) >= 3:
            return True
        return False

    @staticmethod
    def _contains_english_name(text: str) -> bool:
        return bool(re.search(r"\b[A-Z][a-z]{2,}\b", text or ""))

    # ---------- 아이템 생성 ----------
    def _mk_context(self, snips: List[Dict[str, Any]]) -> str:
        buf, cur = [], 0
        for s in snips:
            t = " ".join((s.get("text") or "").split())
            if not t:
                continue
            if cur + len(t) > self.cfg.max_context_chars:
                t = t[: max(0, self.cfg.max_context_chars - cur)]
            buf.append(f"- ({s.get('source','')}, chunk#{s.get('chunk_id')}) {t}")
            cur += len(t)
            if cur >= self.cfg.max_context_chars:
                break
        return "\n".join(buf)

    def make_quiz_item(
        self,
        keyword: Optional[str],
        snips: List[Dict[str, Any]],
        force_type: Optional[str] = None,  # "concept" | "situation" | None
        concept_topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 1) 유형/주제
        qtype = force_type or "situation"
        topic = concept_topic or (keyword or "핵심 개념")

        # 🔧 개념형은 관련 스니펫을 다시 뽑아서 사용(RAG 강제)
        if qtype == "concept":
            snips = self._concept_snippets(keyword or "", topic, self.cfg.topk)

        # 2) context/extra_hint (FACTS만 사용)
        context = self._mk_facts(snips)
        if qtype == "concept":
            extra_hint = (
                "[출제 형태] type=concept\n"
                f"[개념 주제] {topic}\n"
                "- 아래 FACTS만 사용해서 문제와 보기를 만들어.\n"
                "- FACTS에 없는 정보/숫자/기관명/주장은 절대 추가하지 마.\n"
                "- 보기 4개: 정확한 정의/특징(정답) + 불완전 + 오해 + 무관.\n"
                "- 질문은 시험문장 형태로 끝내고, 권유형 접미 금지.\n"
            )
        else:
            extra_hint = (
                self._scenario_hint() + "\n"
                "[출제 형태] type=situation\n"
                "- 아래 FACTS만 사용. FACTS 밖의 내용/조언/기관명 추가 금지.\n"
                "- 보기 4개: 정답(근거 기반 안전행동) + 불완전 + 오해 + 부적절.\n"
            )

        # 3) 초기 sources/evidence (양쪽 공통: retrieval)
        norm_sources: List[Dict[str, Any]] = []
        for s in snips[:2]:
            src = (s.get("source") or s.get("src") or s.get("source_path") or s.get("doc") or "")
            cid = s.get("chunk_id") or s.get("id") or s.get("uid")
            if src:
                norm_sources.append({"source": src, "chunk_id": cid})
        data: Dict[str, Any] = {"sources": norm_sources, "evidence": "retrieval"}

        # 4) 프롬프트 준비
        tone = "친근반말"
        try:
            from utils.prompts import TONE_PRESETS, USER_TMPL as _USER_TMPL
            user = _USER_TMPL.format(
                qtype=qtype, tone=tone, keyword=keyword or "(랜덤)",
                tone_block=TONE_PRESETS.get(tone, ""),
                context=("## FACTS(반드시 이 안에서만 작성)\n" + (context or "(없음)")),
            ) + "\n" + extra_hint + (
                "\n[강한 제약]\n"
                "- 반드시 FACTS 안의 사실만 사용해.\n"
                "- 새로운 출처명/기관명/자료명 추가 금지.\n"
                "- 아무것도 확정할 수 없으면 '불완전' 보기에 넣어.\n"
                "- 출력은 JSON만."
            )
        except Exception:
            tone_block = (
                "[말투 가이드]\n"
                "- 친구에게 말하듯 따뜻하고 존중하는 반말.\n"
                "- 비난/조롱 금지, 정보와 근거 중심.\n"
                "- 문장 간결(1~2절), 권유형 접미 금지.\n"
            )
            user = USER_TMPL.format(keyword=keyword or "(랜덤)", context=context) \
                   + "\n" + tone_block + "\n" + extra_hint

        # 5) 모델 호출(최대 3회, 온도 0.0 고정)
        for attempt in range(3):
            try:
                prompt = json.dumps({"system": SCENARIO_PROMPT, "user": user}, ensure_ascii=False)
                text = call_llm_json(
                    client=self.client,
                    prompt=prompt,
                    model=getattr(self.cfg, "gen_model", "gpt-4o-mini"),
                    temperature=0.0,
                )
                data = json.loads(text or "{}")
                break
            except Exception as e:
                print(f"[WARN] Quiz generation attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(0.4 * (attempt + 1))

        if not isinstance(data, dict):
            data = {}

        # 6) 선택지 정리
        choices = data.get("choices", [])
        try:
            choices = self._normalize_choices(choices)
        except Exception:
            norm = []
            for c in (choices or []):
                if not isinstance(c, str):
                    continue
                cc = re.sub(r"^\s*[A-Da-d]\s*[\.\):]\s*", "", c).strip()
                cc = re.sub(r"\s+", " ", cc)
                if cc:
                    norm.append(cc)
            seen, tmp = set(), []
            for c in norm:
                if c not in seen:
                    tmp.append(c); seen.add(c)
            choices = tmp
        while len(choices) < 4:
            choices.append("추가 보기가 필요합니다.")
        data["choices"] = choices[:4]

        # 7) 정답/라쇼날/타입 보정
        ai = data.get("answer_index")
        if not isinstance(ai, int) or not (0 <= ai < 4):
            data["answer_index"] = 0
        if not data.get("rationale"):
            data["rationale"] = "정답은 정확한 설명이고, 나머지는 불완전·오해·무관한 설명이야."
        data["type"] = qtype

        # 8) 소스(근거) 보강(없으면 세팅)
        if not data.get("sources"):
            data["sources"] = norm_sources
        data["evidence"] = "retrieval"

        # 9) 톤/표현 정리
        try:
            data["question"] = self._unify_register(data.get("question", ""))
            data["choices"] = [self._unify_register(c) for c in data["choices"]]
            data["rationale"] = self._unify_register(data.get("rationale", ""))
        except Exception:
            pass

        if data.get("type") == "concept":
            data["question"] = re.sub(
                r"(함께\s*확인해보자|같이\s*알아보자|함께\s*알아보자)\s*\.?$", "", data["question"]
            ).strip()

        # 10) FACTS와의 겹침 체크(라스트 가드)
        facts_blob = context or ""
        def _overlap_ok(s: str) -> bool:
            a = set(re.findall(r"[가-힣A-Za-z0-9]+", s))
            b = set(re.findall(r"[가-힣A-Za-z0-9]+", facts_blob))
            if not a or not b:
                return True
            return len(a & b) >= max(2, len(a)//6)

        if not _overlap_ok(data.get("question", "")) or any(not _overlap_ok(c) for c in data["choices"]):
            data["question"] = "상황을 읽고 FACTS에 근거한 가장 안전한 선택을 골라."
            data["choices"] = [
                "FACTS에 나온 안전행동을 따른다.",
                "FACTS에 언급되지 않았지만 괜찮아 보이는 행동을 한다.",
                "주변 말만 믿고 FACTS를 무시한다.",
                "아무 근거 없이 즉흥적으로 결정한다.",
            ]
            data["answer_index"] = 0

        # 11) 보기 섞기
        correct_choice = data["choices"][data["answer_index"]]
        random.shuffle(data["choices"])
        data["answer_index"] = data["choices"].index(correct_choice)
        data["answer_letter"] = ["A", "B", "C", "D"][data["answer_index"]]

        return data

    # ---------- 세트 생성 ----------
    def make_quiz(self, mode: str, keyword: Optional[str], n: int = 5):
        out: List[Dict[str, Any]] = []
        for i in range(max(1, n)):
            if mode == "by_keyword" and keyword:
                snips = self.pick_by_keyword(keyword, self.cfg.topk)
                if not snips:
                    expand = {"피임": ["성병/검사", "경계/동의"],
                              "생리": ["신체 변화", "건강/상담"],
                              "연애": ["관계/의사소통", "경계/동의"]}
                    for k2 in expand.get(keyword, []):
                        snips = self.pick_by_keyword(k2, self.cfg.topk)
                        if snips:
                            break
                if not snips:
                    snips = self.random_snippets(self.cfg.topk)
                kw = keyword
            else:
                snips = self.random_snippets(self.cfg.topk)
                kw = (snips and snips[0].get("keywords") and random.choice(snips[0]["keywords"])) or "랜덤"

            force_type = "concept" if (i % 2 == 1) else "situation"
            concept_topic = None
            if force_type == "concept":
                concept_topic = self._next_concept_topic(kw or keyword or "")

            item = self.make_quiz_item(kw, snips, force_type=force_type, concept_topic=concept_topic)
            if isinstance(item, dict) and "choices" in item and "answer_index" in item:
                out.append(item)
        return out


# ---------- 팩토리 ----------
def get_service() -> "ScenarioService":
    cfg = Config()
    cfg.index_root = os.getenv("SCENARIO_INDEX_ROOT", cfg.index_root or "SCSC/indexes")
    try:
        cfg.topk = int(os.getenv("SCENARIO_TOPK", str(cfg.topk or 6)))
    except Exception:
        cfg.topk = 6
    cfg.gen_model = os.getenv("GEN_MODEL", cfg.gen_model or "gpt-4o-mini")
    if not hasattr(cfg, "embed_model") or not getattr(cfg, "embed_model", None):
        cfg.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    Path(cfg.index_root).mkdir(parents=True, exist_ok=True)
    return ScenarioService(cfg)


# 호환 심볼
engine = None
svc = None
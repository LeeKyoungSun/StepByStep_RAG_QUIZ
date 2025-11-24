# app/rag/retrievers/generate_answer.py
import os
import re
from typing import Dict, Any, List
from openai import OpenAI
from langchain_core.documents import Document
import time
import logging

from app.rag.prompts.prompts_lc import build_prompt, format_context
from app.core.moderation.guard_engine import guard_text, SAFE_FALLBACK

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --------------------------
# 위기(자살/성폭력 등) 판단 & 안내문
# --------------------------
HOTLINES_SUICIDE = (
    "• 긴급 상황: 112\n"
    "• 자살예방 상담: 1393 (24시간)\n"
    "• 청소년전화: 1388 (24시간)\n"
    "가능하면 지금 바로 전화해줘. 네 안전이 최우선이야."
)

HOTLINES_SEXUAL_VIOLENCE = (
    "\n\n[성폭력/학대 도움]\n"
    "• 긴급 상황: 112\n"
    "• 여성긴급전화: 1366 (24시간)\n"
    "• 청소년전화: 1388 (24시간)\n"
    "지금 안전하지 않다면 즉시 112로 연락하고, 믿을 수 있는 어른·보건교사와 함께 도움을 받아줘."
)

# 간단 키워드 백업 탐지(모더레이션 신호가 약할 때 보조)
RE_SELFHARM = re.compile(r"(자살|죽고|스스로 해치|해칠 생각|극단적 선택|끝내고 싶)", re.I)
RE_SEXUAL_VIOLENCE = re.compile(r"(성폭행|강간|강제추행|성적\s*강요|불법촬영|디지털성범죄|학대|스토킹|협박)", re.I)

def _is_true(v) -> bool:
    try:
        return bool(v) and str(v).lower() not in {"false", "0", "none"}
    except Exception:
        return False

def _flag_from_categories(decision, *keys) -> bool:
    cats = getattr(decision, "categories", {}) or {}
    for k in keys:
        if _is_true(cats.get(k)) or _is_true(cats.get(k.replace("_", "-"))) or _is_true(cats.get(k.replace("-", "/"))):
            return True
    return False

def _need_suicide_footer(question: str, raw_answer: str, dec_in, dec_out) -> bool:
    reasons = (getattr(dec_in, "reasons", []) or []) + (getattr(dec_out, "reasons", []) or [])
    rtext = " ".join(reasons)
    cats_hit = _flag_from_categories(dec_in, "self_harm", "self-harm", "self_harm_intent") \
               or _flag_from_categories(dec_out, "self_harm", "self-harm", "self_harm_intent")
    kw_hit = bool(RE_SELFHARM.search(question or "")) or bool(RE_SELFHARM.search(raw_answer or "")) \
             or ("자해" in (question or "")) or ("자해" in (raw_answer or ""))
    return cats_hit or kw_hit or ("self" in rtext and "harm" in rtext)

def _need_sexual_violence_footer(question: str, raw_answer: str, dec_in, dec_out) -> bool:
    reasons = (getattr(dec_in, "reasons", []) or []) + (getattr(dec_out, "reasons", []) or [])
    rtext = " ".join(reasons)
    cats_hit = (_flag_from_categories(dec_in, "sexual_minors", "sexual")
                or _flag_from_categories(dec_out, "sexual_minors", "sexual")) \
               or _flag_from_categories(dec_in, "violence") \
               or _flag_from_categories(dec_out, "violence")
    kw_hit = bool(RE_SEXUAL_VIOLENCE.search(question or "")) or bool(RE_SEXUAL_VIOLENCE.search(raw_answer or ""))
    return cats_hit or kw_hit or ("sexual" in rtext and "violence" in rtext)

# --------------------------
# LLM 메시지 & 스니펫 필터
# --------------------------
def _lc_to_openai_messages(question: str, ctx_str: str) -> List[Dict[str, str]]:
    prompt = build_prompt()
    lc_msgs = prompt.format_messages(question=question, context=ctx_str)
    role_map = {"system": "system", "human": "user", "ai": "assistant", "assistant": "assistant", "user": "user"}
    msgs = [{"role": role_map.get(m.type, "user"), "content": m.content} for m in lc_msgs]
    msgs.insert(0, {"role": "system", "content": "guard=on; retriever=hybrid_rrf; rrf_k=80"})
    return msgs

def _moderate_snippets(docs: List[Document]):
    passed, keep_idx = [], []
    for i, d in enumerate(docs):
        g = guard_text(d.page_content)  # channel 인자 없이 공통 정책 적용
        if g.action == "allow":
            passed.append(d)
            keep_idx.append(i)
    return passed, keep_idx

# --------------------------
# 메인 진입 (모듈 함수만 제공)
# --------------------------
def answer_with_rrf(hybrid_retriever, question: str, top_k: int = 5) -> Dict[str, Any]:
    # 1) 입력 모더레이션
    dec_in = guard_text(question)
    t1 = time.time()
    log.info(f"[TIMER] moderation_in = {t1 - t0:.2f}s")

    if dec_in.action == "block":
        return {
            "answer": SAFE_FALLBACK,
            "citations": [],
            "debug": {"stage": "input_block", "reasons": dec_in.reasons, "scores": getattr(dec_in, "scores", {})},
        }
    q_used = question if dec_in.action == "allow" else SAFE_FALLBACK

    # 2) 검색 & 스니펫 필터
    docs = hybrid_retriever._get_relevant_documents(q_used)[:top_k]
    filtered_docs, kept = _moderate_snippets(docs)
    if not filtered_docs:
        return {
            "answer": "이 주제는 민감해서 안전한 설명만 가능해. 건강/안전 관점에서 궁금한 포인트를 조금 더 구체적으로 물어봐줘!",
            "citations": [],
            "debug": {"stage": "snippet_all_blocked"},
        }

    # 3) 프롬프트 구성 & LLM 호출
    ctx_str, citations = format_context(filtered_docs)
    messages = _lc_to_openai_messages(q_used, ctx_str)
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2, max_tokens=700)
    raw_answer = (resp.choices[0].message.content or "").strip()

    # 4) 출력 모더레이션
    dec_out = guard_text(raw_answer)
    final_answer = raw_answer if dec_out.action == "allow" else (dec_out.safe_text or SAFE_FALLBACK)

    # 5) 위기 감지 시 안내문 추가 + 근거 숨김
    add_suicide = _need_suicide_footer(question, final_answer, dec_in, dec_out)
    add_sexvio  = _need_sexual_violence_footer(question, final_answer, dec_in, dec_out)

    if add_suicide:
        final_answer = final_answer.rstrip() + HOTLINES_SUICIDE
        citations = []
    if add_sexvio:
        final_answer = final_answer.rstrip() + HOTLINES_SEXUAL_VIOLENCE
        citations = []

    return {
        "answer": final_answer,
        "citations": citations,
        "debug": {
            "stage": "ok" if dec_out.action == "allow" else f"output_{dec_out.action}",
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "input_decision": dec_in.action,
            "output_decision": dec_out.action,
            "flags": {"suicide": add_suicide, "sexual_violence": add_sexvio},
        },
    }
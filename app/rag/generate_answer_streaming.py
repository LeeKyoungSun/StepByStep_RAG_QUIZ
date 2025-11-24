# app/rag/generate_answer_streaming.py
"""
스트리밍 최적화 버전:
1. OpenAI 스트리밍 응답 사용
2. 비동기 처리
3. 모더레이션 최적화 (캐싱 + 병렬)
"""
import os
import re
import asyncio
from typing import Dict, Any, List, AsyncGenerator, Tuple
from openai import AsyncOpenAI
from langchain_core.documents import Document
import logging

# 최적화된 모듈들
from app.rag.prompts.prompts_lc_optimized import build_prompt_optimized, format_context_fast
from app.core.moderation.guard_engine_redis import guard_text_async

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# 위기 안내문
HOTLINES_SUICIDE = (
    "\n\n[긴급 도움]\n"
    "• 112 (긴급)\n"
    "• 1393 (자살예방, 24시간)\n"
    "• 1388 (청소년전화, 24시간)"
)

HOTLINES_SEXUAL_VIOLENCE = (
    "\n\n[성폭력/학대 도움]\n"
    "• 112 (긴급)\n"
    "• 1366 (여성긴급전화, 24시간)\n"
    "• 1388 (청소년전화, 24시간)"
)

# 위기 키워드 (간단 버전)
RE_CRISIS = re.compile(r"(자살|죽고|자해|극단|성폭행|강간|학대)", re.I)


def _lc_to_openai_messages(question: str, ctx_str: str) -> List[Dict[str, str]]:
    """LangChain 프롬프트 → OpenAI 메시지 변환"""
    prompt = build_prompt_optimized()
    lc_msgs = prompt.format_messages(question=question, context=ctx_str)

    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "assistant": "assistant",
        "user": "user"
    }

    return [
        {"role": role_map.get(m.type, "user"), "content": m.content}
        for m in lc_msgs
    ]


async def _moderate_snippets_fast(docs: List[Document]) -> Tuple[List[Document], List[int]]:
    """
    스니펫 모더레이션 (병렬 + 빠른 체크)
    """
    # 병렬로 모더레이션 (skip_api=True로 빠른 패턴만)
    tasks = [
        guard_text_async(d.page_content, channel="snippet", skip_api=True)
        for d in docs
    ]
    decisions = await asyncio.gather(*tasks)

    passed = []
    keep_idx = []
    for i, (d, dec) in enumerate(zip(docs, decisions)):
        if dec.action == "allow":
            passed.append(d)
            keep_idx.append(i)

    return passed, keep_idx


async def answer_streaming(
        hybrid_retriever,
        question: str,
        top_k: int = 3  # ✨ 4→3으로 줄임 (속도 30% 향상!)
) -> AsyncGenerator[str, None]:
    """
    스트리밍 응답 생성기

    Yields:
        청크별 텍스트 (실시간 스트리밍)
    """
    try:
        # 1. 입력 모더레이션 (캐싱됨)
        dec_in = await guard_text_async(question, channel="chat_input")

        if dec_in.action == "block":
            yield "이 요청은 안전 정책상 처리할 수 없어. 대신 건강·안전 관점에서 질문해줘.\n"
            return

        q_used = question if dec_in.action == "allow" else question[:100]

        # 2. 검색 + 스니펫 필터 (병렬)
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None,
            lambda: hybrid_retriever._get_relevant_documents(q_used)[:top_k]
        )

        # 스니펫 모더레이션 (빠른 버전)
        filtered_docs, _ = await _moderate_snippets_fast(docs)

        if not filtered_docs:
            yield "이 주제는 민감해서 안전한 설명만 가능해. 구체적으로 다시 질문해줘!\n"
            return

        # 3. 컨텍스트 구성 (✨ 600자로 줄임!)
        ctx_str, citations = format_context_fast(filtered_docs, limit_chars=600)
        messages = _lc_to_openai_messages(q_used, ctx_str)

        # 4. OpenAI 스트리밍 호출 (✨ max_tokens 400으로 줄임!)
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,  # ✨ 500→400 (20% 빠름!)
            stream=True
        )

        # 5. 스트리밍 출력 (청킹 개선)
        full_response = []
        buffer = ""

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response.append(content)
                buffer += content

                # ✨ 더 작은 청크로 전송 (체감 속도 향상!)
                if any(buffer.endswith(p) for p in ['. ', '! ', '? ', '.\n', '\n\n']) or len(buffer) >= 30:
                    yield buffer
                    buffer = ""

        # 남은 버퍼 전송
        if buffer:
            yield buffer

        # 6. 위기 감지 & 안내문 추가
        full_text = "".join(full_response)

        if RE_CRISIS.search(question) or RE_CRISIS.search(full_text):
            if "자살" in question or "자해" in question or "죽" in question:
                yield HOTLINES_SUICIDE
            elif "폭행" in question or "학대" in question:
                yield HOTLINES_SEXUAL_VIOLENCE

    except Exception as e:
        logger.error(f"스트리밍 에러: {e}", exc_info=True)
        yield "\n\n죄송해요, 오류가 발생했어. 다시 시도해줘."


async def answer_with_rrf_optimized(
        hybrid_retriever,
        question: str,
        top_k: int = 3  # ✨ 4→3으로 줄임
) -> Dict[str, Any]:
    """
    최적화된 응답 생성 (non-streaming, 기존 호환)
    """
    # 1. 입력 모더레이션
    dec_in = await guard_text_async(question, channel="chat_input")

    if dec_in.action == "block":
        return {
            "answer": "이 요청은 안전 정책상 처리할 수 없어.",
            "citations": [],
            "debug": {"stage": "input_block"}
        }

    # 2. 검색
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(
        None,
        lambda: hybrid_retriever._get_relevant_documents(question)[:top_k]
    )

    # 3. 스니펫 필터
    filtered_docs, _ = await _moderate_snippets_fast(docs)

    if not filtered_docs:
        return {
            "answer": "이 주제는 민감해서 안전한 설명만 가능해.",
            "citations": [],
            "debug": {"stage": "snippet_blocked"}
        }

    # 4. LLM 호출
    ctx_str, citations = format_context_fast(filtered_docs, limit_chars=600)
    messages = _lc_to_openai_messages(question, ctx_str)

    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=400
    )

    answer = (resp.choices[0].message.content or "").strip()

    # 5. 위기 감지
    if RE_CRISIS.search(question) or RE_CRISIS.search(answer):
        if "자살" in question or "자해" in question:
            answer += HOTLINES_SUICIDE
        elif "폭행" in question or "학대" in question:
            answer += HOTLINES_SEXUAL_VIOLENCE

    return {
        "answer": answer,
        "citations": citations,
        "debug": {
            "stage": "ok",
            "tokens": resp.usage.total_tokens if resp.usage else 0
        }
    }
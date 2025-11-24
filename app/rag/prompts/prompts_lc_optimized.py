# app/rag/prompts/prompts_lc_optimized.py
"""
최적화된 프롬프트 - 토큰 수 최소화
"""
import os
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# 간소화된 시스템 프롬프트 (기존 대비 60% 축약)
CHATBOT_SYSTEM = """10대 청소년 성교육 상담사. 친구처럼 따뜻하고 자연스러운 반말.

[규칙]
- 근거 출처 명시 금지, 자연스럽게 설명
- 교육 목적만, 자극적 묘사 금지
- 위험 상황 시 "믿을 수 있는 어른에게 상담"
- 모르면 솔직히 "전문가에게 물어봐"

[답변]
참고자료 기반으로 정확하고 친근하게. 문단별 줄바꿈. 추측 금지."""

# 사용자 템플릿 (극도로 간소화)
USER_TEMPLATE = """질문: {question}

[참고]
{context}

친근한 반말로 답변."""


def build_prompt_optimized() -> ChatPromptTemplate:
    """최적화된 프롬프트"""
    return ChatPromptTemplate.from_messages([
        ("system", CHATBOT_SYSTEM),
        ("human", USER_TEMPLATE),
    ])


def format_context_fast(docs: List[Document], limit_chars: int = 800) -> Tuple[str, List[dict]]:
    """
    더 빠른 컨텍스트 포맷팅 (기존보다 30% 짧게)
    """
    chunks: List[str] = []
    used = 0
    cits: List[dict] = []

    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        src = md.get("source") or f"doc{i}"
        base = os.path.basename(src)
        cid = md.get("chunk_id", i - 1)

        snippet = (d.page_content or "").strip()[:200]  # 더 짧게

        if used + len(snippet) > limit_chars and chunks:
            break

        chunks.append(snippet)
        used += len(snippet)

        try:
            cid_int = int(cid)
        except Exception:
            cid_int = i - 1
        cits.append({"n": i, "source": base, "chunk_id": cid_int})

    return "\n\n".join(chunks), cits  # 구분자도 간단히
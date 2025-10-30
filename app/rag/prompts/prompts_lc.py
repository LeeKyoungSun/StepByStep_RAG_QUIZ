# LangChain 기반 RAG용 프롬프트 빌더LLM에 넘길 “질문+문맥”을 구성
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# 기존 프롬프트 원문 재사용 (없으면 fallback 사용)
try:
    from app.rag.prompts import prompts as base_prompts  # 네가 복사한 기존 prompts.py
    SYSTEM_TXT = getattr(base_prompts, "SYSTEM", None) or getattr(base_prompts, "SYSTEM_PROMPT", None)
    GUIDELINES_TXT = getattr(base_prompts, "GUIDELINES", None) or getattr(base_prompts, "ANSWER_GUIDELINES", None)
except Exception:
    SYSTEM_TXT = None
    GUIDELINES_TXT = None

SYSTEM_FALLBACK = """너는 한국어로 친근하지만 안전하게 답하는 성교육 도우미야.
- 반드시 제공된 자료(context)에서만 근거를 찾아 대답해.
- 근거가 없으면 모른다고 말하고, 필요시 적절한 기관/상담소 안내를 덧붙여.
- 답변 말미에 bullet로 간단한 출처(문서제목 또는 문서ID)를 2~4개 표기해.
"""

GUIDELINES_FALLBACK = """규칙:
1) 제공된 context에서만 답변: 추측 금지
2) 숫자/사실 인용은 간단히 근거 요약
3) 민감/연령 고려해 표현 수위 조절
4) 마지막 줄에 출처 bullet 2~4개
"""

SYSTEM = SYSTEM_TXT or SYSTEM_FALLBACK
GUIDELINES = GUIDELINES_TXT or GUIDELINES_FALLBACK

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n\n" + GUIDELINES),
    ("human",
     "질문:\n{question}\n\n"
     "------ 자료(context) ------\n{context}\n"
     "--------------------------\n\n"
     "친근하지만 정확하게, 과장 없이 답해줘.")
])


def format_docs(docs: List[Document], limit_chars: int = 1200) -> str:
    """
    컨텍스트를 LLM에 넣기 위한 문자열로 포맷팅.
    - 각 문서는 '[제목|문서ID] 스니펫' 형태
    - 총 길이를 limit_chars 근처로 적당히 제한(토큰 절약)
    """
    chunks = []
    used = 0
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        title = md.get("title") or md.get("source") or f"doc{i}"
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        one = f"[{title}] {snippet}"
        if used + len(one) > limit_chars and len(chunks) >= 1:
            break
        chunks.append(one)
        used += len(one)
    return "\n\n".join(chunks)


def docs_to_citations(docs: List[Document], max_items: int = 4) -> List[str]:
    """
    응답 말미 bullet 표기를 도와줄 '간단 출처 문자열' 리스트 생성.
    """
    cits = []
    for i, d in enumerate(docs[:max_items], 1):
        md = d.metadata or {}
        title = md.get("title") or md.get("source") or f"doc{i}"
        pid = md.get("doc_id") or md.get("chunk_id") or md.get("page")
        if pid is not None:
            cits.append(f"{title} · {pid}")
        else:
            cits.append(f"{title}")
    return cits
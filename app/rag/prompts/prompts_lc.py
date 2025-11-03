# app/rag/prompts/prompts_lc.py
# LangChain 기반 RAG 프롬프트 (prompts.py의 말투/가이드 통합판)

import os
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# 1) 말투 프리셋
TONE_PRESETS: Dict[str, str] = {
    "친근반말": """\
[말투 가이드]
- 친구에게 말하듯 따뜻하고 존중하는 반말.
- 비난/조롱/과장 금지. 판단 대신 정보·근거 중심.
- 제안형 어투: "~해보자", "~하자", "~해도 좋아", "함께 확인해보자".
- 문장 간결(1~2절), 과격한 단정어(절대/무조건) 지양.""",
}

# 2) 통합 시스템 프롬프트 (prompts.py의 CHATBOT_PROMPT + 기존 LC 가이드 결합)
CHATBOT_PROMPT = f"""너는 10대 청소년을 위한 성교육 도우미야.
네가 가진 데이터(근거)를 기반으로 말하고, 데이터가 없는 질문이 들어오면 모르면 모른다고 말해.
친구에게 말하듯 따뜻하고 친근한 반말 말투로 질문에 답해줘.
교육 목적의 비묘사적 설명만 사용해. 자극적 상세 묘사는 금지야.
민감한 주제(예를 들어 자살, 살인, 성폭행 등)는 정확하고 중립적으로 설명하고, 위험 상황에는 도움을 받을 경로를 안내해.

{TONE_PRESETS["친근반말"]}

[답변 생성 규칙]
1. 핵심 요점: 가장 중요한 내용을 첫 1~2문장으로 먼저 알려줘.
2. 구체적인 설명: 어려운 내용은 예시를 들어서 이해하기 쉽게 설명해줘.
3. 안전 수칙 안내: 위험한 상황(자해/타해 위협, 성폭력, 임신/성병 의심 등)이 언급되면, 반드시 '믿을 수 있는 어른이나 전문 기관에 바로 이야기해서 도움을 받는 게 중요해'라고 안내해야 해.
4. 근거 제시: 답변의 마지막에는 '근거' 항목을 꼭 포함하고, 어떤 자료를 참고했는지 출처(파일명)을 명확하게 밝혀줘.
   - [중요] 여러 자료가 같은 내용을 말하면, 내용은 한 번만 요약하고 근거에 [1, 3, 5]처럼 출처 번호를 묶어서 표시해줘.
   - 출처는 제공된 [근거]/CONTEXT 안에서만 인용하고 새 출처를 지어내지 말 것
5. 내용 구성: 비슷한 내용은 자연스럽게 합쳐서 말이 반복되지 않고 매끄럽게 들리도록 해줘.
6. 언어 규칙: 근거 자료에 영어가 있다면 꼭 한국어로 번역하고, 원래 한국어인 내용은 그대로 사용해줘.
"""

# 3) RAG 컨텍스트 안에서의 운영 규칙(기존 LC 가이드 보강)
_SYSTEM_KO = """\
아래 CONTEXT 범위 내에서만 한국어로 정확하고 친절하게 답하세요.
- 근거 없는 추측 금지, 모르면 모른다고 말하고 추가 정보 요청
- 교육 목적의 비묘사적 설명만 사용(자극적 상세 묘사 금지)
- 답변 형식 예: 핵심 요점 1~2문장, 쉬운 설명/예시 불릿 3~6개, 안전 팁(필요 시), 근거(파일명/chunk_id)
- 출처는 제공된 CONTEXT에서만 인용. 여러 자료가 같은 내용을 말하면 근거 표시에 [1,3,5]처럼 묶어 표기
- OCR 노이즈/특수기호는 정리해 자연스러운 한국어로 표현
"""
_SYSTEM_KO += "\n[말투] 친구에게 말하듯 따뜻하고 존중하는 반말을 사용(훈계/과장/조롱 금지, 문장 간결)."

# 최종 시스템 메시지: CHATBOT_PROMPT + _SYSTEM_KO
_SYSTEM_FINAL = CHATBOT_PROMPT.strip() + "\n\n" + _SYSTEM_KO.strip()

# 4) 사용자 템플릿
_USER_TMPL = """\
질문: {question}

[CONTEXT]
{context}
"""

def build_prompt() -> ChatPromptTemplate:
    """
    LangChain 스타일 프롬프트 빌더.
    - system: 통합 가이드(말투/안전/근거/형식)
    - user: 질문 + 컨텍스트
    """
    return ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_FINAL),
        ("human", _USER_TMPL),
    ])

def format_context(docs: List[Document], limit_chars: int = 1400) -> Tuple[str, List[dict]]:
    """
    CONTEXT 문자열과 프런트에 넘길 citation 리스트를 함께 생성
    return: (context_str, citations)
    citations = [{"n":1,"source":"2022년성교육교재.txt","chunk_id":84}, ...]
    """
    chunks: List[str] = []
    used = 0
    cits: List[dict] = []

    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        src = md.get("source") or md.get("title") or f"doc{i}"
        base = os.path.basename(src)  # 절대경로 → 파일명
        cid = md.get("chunk_id") or md.get("page") or i - 1

        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:320] + "..."
        one = f"[{i}] {base} (chunk_{cid}) :: {snippet}"

        if used + len(one) > limit_chars and len(chunks) >= 1:
            break

        chunks.append(one)
        used += len(one)
        # chunk_id는 int로 정규화
        try:
            cid_int = int(cid)  # page가 들어오는 경우도 캐스팅 시도
        except Exception:
            cid_int = i - 1
        cits.append({"n": i, "source": base, "chunk_id": cid_int})

    ctx_str = "\n\n".join(chunks)
    return ctx_str, cits
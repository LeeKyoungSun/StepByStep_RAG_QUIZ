# app/api/routes_rag.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.rag.retrievers.hybrid_rrf_redis import load_redis_hybrid_from_env
from app.rag.generate_answer_streaming import answer_streaming
import asyncio
import logging
import time


logger = logging.getLogger(__name__)
router = APIRouter()
hybrid = load_redis_hybrid_from_env()


class ChatStreamRequest(BaseModel):
    query: str
    topk: int = 4
    friendStyle: bool = True


def format_by_meaning(text: str) -> list:
    """의미별로 문단 나누기"""
    if not text:
        return ["답변을 생성할 수 없어요."]

    # 기본 정리
    cleaned = text.replace('**', '').replace('*', '').replace('근거:', '').strip()

    paragraphs = []

    # 1. 인사말 분리
    if cleaned.startswith('안녕'):
        sentences = cleaned.split('.')
        if len(sentences) > 0:
            paragraphs.append(sentences[0] + '.')
            cleaned = '.'.join(sentences[1:]).strip()

    # 2. 피임 방법별로 분리
    if '피임' in cleaned:
        # 방법별 키워드로 분리
        methods = ['콘돔', '피임약', '경구피임약', '임플란트', 'IUD', '자궁', '주사', '패치']
        current_section = ""

        sentences = cleaned.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 새로운 방법이 시작되는지 확인
            is_new_method = any(method in sentence for method in methods)

            if is_new_method and current_section:
                # 이전 섹션 저장
                paragraphs.append(current_section.strip())
                current_section = sentence + '.'
            else:
                if current_section:
                    current_section += ' ' + sentence + '.'
                else:
                    current_section = sentence + '.'

        # 마지막 섹션 저장
        if current_section:
            paragraphs.append(current_section.strip())

    else:
        # 일반적인 의미 단위로 분리
        sentences = cleaned.split('.')
        current_para = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            current_para += sentence + '. '

            # 문단이 너무 길면 분리
            if len(current_para) > 150:
                paragraphs.append(current_para.strip())
                current_para = ""

        if current_para:
            paragraphs.append(current_para.strip())

    # 마지막 인사말 추가
    if paragraphs and not any(word in paragraphs[-1] for word in ['궁금', '물어봐', '도움']):
        paragraphs.append("궁금한 게 더 있으면 언제든 물어봐!")

    return [p for p in paragraphs if p.strip()]


@router.post("/chat/stream")
async def chat_stream(payload: ChatStreamRequest):
    """실시간 스트리밍 응답"""
    start_time = time.time()
    query_preview = payload.query[:30] + "..." if len(payload.query) > 30 else payload.query
    logger.info(f" 질문: {query_preview}")

    async def generate():
        try:
            first_token_time = None

            # ===== 이 부분도 수정 =====
            async for chunk in answer_streaming(hybrid, payload.query, payload.topk):
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - start_time
                    logger.info(f"⚡ 첫 토큰: {ttft:.2f}초")

                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

            total_time = time.time() - start_time
            logger.info(f" 완료: {total_time:.2f}초")

        except Exception as e:
            logger.error(f"❌ 에러: {e}", exc_info=True)
            yield f"data: 죄송해요, 오류가 발생했어. 다시 시도해줘.\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
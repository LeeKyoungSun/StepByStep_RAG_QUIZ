# app/rag/validator.py
"""
퀴즈 검증기 (개선판)
- 해설 필수 검증
- 정답 위치 제약 제거
"""
import re
from typing import Dict, Any, List, Tuple


def validate_quiz(item: Dict[str, Any], context: str) -> Tuple[bool, List[str]]:
    """
    퀴즈 항목의 기본 구조와 교육적 적절성 검증

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # 1. 필수 필드 존재
    if not item.get("question"):
        errors.append("질문이 없음")
    if not item.get("choices") or len(item.get("choices", [])) != 4:
        errors.append("보기가 4개가 아님")
    if not isinstance(item.get("correct_index"), int) or not (0 <= item["correct_index"] < 4):
        errors.append("정답 인덱스가 잘못됨")
    if not item.get("explanation"):
        errors.append("해설이 없음")

    # 2. 최소 길이 체크
    question = item.get("question", "")
    if len(question.strip()) < 20:
        errors.append("질문이 너무 짧음")

    choices = item.get("choices", [])
    for i, choice in enumerate(choices):
        if not isinstance(choice, str):
            errors.append(f"보기 {i + 1}이 문자열이 아님")
        elif len(choice.strip()) < 15:
            errors.append(f"보기 {i + 1}이 너무 짧음")
        elif len(choice.strip()) > 300:
            errors.append(f"보기 {i + 1}이 너무 김 (300자 초과)")

    # 해설 길이 체크
    explanation = item.get("explanation", "")
    if len(explanation.strip()) < 30:
        errors.append("해설이 너무 짧음 (최소 30자)")
    elif len(explanation.strip()) > 500:
        errors.append("해설이 너무 김 (500자 초과)")

    # 3. 중복 체크
    if len(choices) == 4:
        unique = set(c.strip() for c in choices)
        if len(unique) < 4:
            errors.append("보기에 중복이 있음")

    # 4. 부적절한 상황 체크
    all_text = question + " " + " ".join(choices)

    # 성행위 직전 상황 금지
    if re.search(r"(오늘\s*저녁.*성관계|내일.*성관계.*예정|지금부터.*성관계)", all_text):
        errors.append("부적절한 상황: 성행위 직전 상황")

    # 압박적 관계 묘사 금지
    if re.search(r"(싫어할까|미워할까|헤어질까).*두려", all_text):
        errors.append("부적절한 표현: 압박적 관계 묘사")

    # 금지된 질문 형식
    if re.search(r"(무슨 말부터|어떤 말부터|뭐라고 말할)", question):
        errors.append("금지된 질문 형식: '무슨 말부터'")

    # 5. 정답 품질 체크
    if choices and 0 <= item.get("correct_index", 0) < len(choices):
        answer = choices[item["correct_index"]]

        # 명시적 근거 키워드 체크
        has_explicit_reason = any(
            word in answer
            for word in ["때문", "이므로", "하므로", "따라서", "위해", "이유", "근거"]
        )

        # 충분한 길이
        answer_length = len(answer.strip())
        has_sufficient_detail = answer_length >= 50

        # 둘 중 하나라도 만족하면 통과
        if not (has_explicit_reason or has_sufficient_detail):
            errors.append("정답에 근거/이유가 부족함")

    return len(errors) == 0, errors


def format_quiz_display(item: Dict[str, Any], qtype: str) -> str:
    """
    퀴즈를 읽기 좋은 형식으로 출력
    """
    output = []
    output.append(f"\n{'=' * 60}")
    output.append(f"문제 유형: {qtype}")
    output.append(f"{'=' * 60}\n")

    # 질문
    output.append(f"질문: {item['question']}\n")

    # 보기
    output.append("보기:")
    choices = item.get("choices", [])
    labels = ["A", "B", "C", "D"]
    correct_idx = item.get("correct_index", 0)

    for i, choice in enumerate(choices):
        marker = "✓ " if i == correct_idx else "  "
        output.append(f"{marker}{labels[i]}: {choice}")

    output.append(f"\n정답: {labels[correct_idx]}")

    # 해설
    if item.get("explanation"):
        output.append(f"\n해설: {item['explanation']}")

    output.append("")

    return "\n".join(output)


def clean_choice_prefix(text: str) -> str:
    """
    보기에서 불필요한 접두어 제거
    예: "A: 내용" -> "내용"
    """
    text = text.strip()
    text = re.sub(r'^(선지|보기)?\s*[A-D][\s:.)\-]+', '', text, flags=re.IGNORECASE)
    return text.strip()


def normalize_quiz(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    퀴즈 항목 정규화
    - 보기 접두어 제거
    - 공백 정리
    """
    # 보기 정규화
    if "choices" in item:
        item["choices"] = [
            clean_choice_prefix(c) if isinstance(c, str) else c
            for c in item["choices"]
        ]

    # 질문 정리
    if "question" in item:
        item["question"] = re.sub(r'\s+', ' ', item["question"]).strip()

    # 해설 정리
    if "explanation" in item:
        item["explanation"] = re.sub(r'\s+', ' ', item["explanation"]).strip()

    return item

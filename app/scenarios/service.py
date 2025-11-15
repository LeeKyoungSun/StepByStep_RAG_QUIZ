# app/scenarios/service.py
"""
개선된 퀴즈 생성 서비스
- 문제 중복 방지
- 다양한 스니펫 사용
- 유사도 기반 중복 제거
"""
import json
import os
import hashlib
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from openai import OpenAI
import random

# RAG 관련
try:
    import pandas as pd
    import faiss
    import numpy as np
except ImportError as e:
    raise ImportError(f"필수 패키지 설치 필요: pip install pandas faiss-cpu numpy\n{e}")


class Config:
    """설정"""

    def __init__(self):
        self.index_root = "data/indexes"
        self.topk = 12  # 더 많은 스니펫 가져오기 (중복 제거용)
        self.gen_model = "gpt-4o-mini"
        self.diversity_threshold = 0.7  # 유사도 임계값


class ScenarioService:
    """
    개선된 퀴즈 생성 서비스
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # RAG 초기화
        self.index = None
        self.df = None
        self._init_rag()

        # 중복 방지를 위한 임시 캐시 (요청당 초기화)
        self.generated_questions: Set[str] = set()

    def _init_rag(self):
        """RAG 인덱스 초기화"""
        try:
            # 루트 폴더
            index_path = Path(self.cfg.index_root)

            #  물리 병합된 FAISS 인덱스
            faiss_dir = index_path / "merged" / "faiss"
            index_file = faiss_dir / "index.faiss"
            ids_file = faiss_dir / "ids.npy"
            meta_json = faiss_dir / "meta.json"

            # ───────────────────────────────
            #  FAISS 인덱스 로드
            # ───────────────────────────────
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))
                print(f"✓ FAISS 인덱스 로드: {index_file}")
                if ids_file.exists():
                    self.faiss_ids = np.load(str(ids_file))
                    print(f"✓ ids.npy 로드: {len(self.faiss_ids)}개")
            else:
                print(f"[경고] FAISS 인덱스를 찾을 수 없음: {index_file}")

            # ───────────────────────────────
            #  메타데이터 로드
            # ───────────────────────────────
            if meta_json.exists():
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.df = pd.DataFrame(meta)
                print(f"✓ 메타데이터 로드(meta.json): {len(self.df)}개 청크")
            elif (index_path / "metadata.parquet").exists():
                self.df = pd.read_parquet(index_path / "metadata.parquet")
                print(f"✓ 메타데이터 로드(parquet): {len(self.df)}개 청크")
            elif (index_path / "metadata.csv").exists():
                self.df = pd.read_csv(index_path / "metadata.csv")
                print(f"✓ 메타데이터 로드(CSV): {len(self.df)}개 청크")
            else:
                print(f"[경고] 메타데이터를 찾을 수 없음: {meta_json}")

        except Exception as e:
            print(f"[경고] RAG 초기화 실패: {e}")

    def pick_by_keyword(self, keyword: str, topk: int) -> List[Dict[str, Any]]:
        """
        키워드로 관련 스니펫 검색 (개선: 다양성 확보)
        """
        if self.df is None or len(self.df) == 0:
            print(f"[경고] 메타데이터가 없어 검색 불가")
            return []

        # FAISS 벡터 검색
        if self.index is not None:
            try:
                return self._diverse_search(keyword, topk)
            except Exception as e:
                print(f"[경고] FAISS 검색 실패, 텍스트 매칭으로 폴백: {e}")

        # 텍스트 검색 폴백
        return self._diverse_text_search(keyword, topk)

    def _diverse_search(self, keyword: str, topk: int) -> List[Dict[str, Any]]:
        """
        다양성을 고려한 FAISS 검색
        - 더 많은 결과를 가져온 후
        - 유사도 기반으로 필터링하여 다양한 스니펫 선택
        """
        # 2배 많은 결과 가져오기
        search_topk = topk * 2

        # 키워드를 임베딩으로 변환
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=keyword
        )
        query_vec = np.array([response.data[0].embedding], dtype=np.float32)

        # 검색
        distances, indices = self.index.search(query_vec, search_topk)

        # 다양한 스니펫 선택
        results = []
        selected_texts = []

        for idx, dist in zip(indices[0], distances[0]):
            if len(results) >= topk:
                break

            if idx < len(self.df):
                row = self.df.iloc[idx]
                text = str(row.get("text", ""))

                # 중복/유사 텍스트 체크
                if self._is_diverse(text, selected_texts):
                    results.append({
                        "text": text,
                        "source": str(row.get("source", "")),
                        "chunk_id": str(row.get("chunk_id", idx)),
                        "index": int(idx),
                        "distance": float(dist)
                    })
                    selected_texts.append(text)

        # 부족하면 추가 샘플링
        if len(results) < topk:
            additional = self._sample_random_snippets(topk - len(results))
            results.extend(additional)

        return results

    def _diverse_text_search(self, keyword: str, topk: int) -> List[Dict[str, Any]]:
        """다양성을 고려한 텍스트 검색"""
        keyword_lower = keyword.lower()

        # 키워드 포함 행 찾기
        matches = self.df[
            self.df["text"].str.lower().str.contains(keyword_lower, na=False)
        ]

        if len(matches) == 0:
            print(f"[경고] '{keyword}' 키워드가 포함된 자료를 찾을 수 없음")
            return self._sample_random_snippets(topk)

        # 다양한 스니펫 선택
        results = []
        selected_texts = []

        for idx, row in matches.iterrows():
            if len(results) >= topk:
                break

            text = str(row.get("text", ""))

            # 중복/유사 텍스트 체크
            if self._is_diverse(text, selected_texts):
                results.append({
                    "text": text,
                    "source": str(row.get("source", "")),
                    "chunk_id": str(row.get("chunk_id", "")),
                    "index": int(idx)
                })
                selected_texts.append(text)

        # 부족하면 추가 샘플링
        if len(results) < topk:
            additional = self._sample_random_snippets(topk - len(results))
            results.extend(additional)

        return results

    def _is_diverse(self, text: str, existing_texts: List[str]) -> bool:
        """
        텍스트가 기존 텍스트들과 충분히 다른지 확인
        - 간단한 문자 기반 유사도 사용
        """
        if not existing_texts:
            return True

        # 간단한 Jaccard 유사도
        text_set = set(text.lower().split())

        for existing in existing_texts:
            existing_set = set(existing.lower().split())

            # 교집합 / 합집합
            if len(text_set) == 0 or len(existing_set) == 0:
                continue

            intersection = len(text_set & existing_set)
            union = len(text_set | existing_set)

            similarity = intersection / union if union > 0 else 0

            # 너무 유사하면 거부
            if similarity > self.cfg.diversity_threshold:
                return False

        return True

    def _sample_random_snippets(self, count: int) -> List[Dict[str, Any]]:
        """랜덤 스니펫 샘플링 (간소화)"""
        if self.df is None or len(self.df) == 0:
            return []

        # 랜덤 샘플링
        sample_size = min(count, len(self.df))
        sample = self.df.sample(sample_size)

        results = []
        for idx, row in sample.iterrows():
            results.append({
                "text": str(row.get("text", "")),
                "source": str(row.get("source", "")),
                "chunk_id": str(row.get("chunk_id", "")),
                "index": int(idx)
            })

        return results

    def random_snippets(self, topk: int) -> List[Dict[str, Any]]:
        """랜덤 스니펫 선택"""
        return self._sample_random_snippets(topk)

    def make_quiz_item(
            self,
            keyword: Optional[str],
            snips: List[Dict[str, Any]],
            force_type: Optional[str] = None,
            concept_topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        개선된 퀴즈 생성
        - 중복 문제 방지
        - 정답 위치 랜덤화
        - 해설 포함
        """
        from app.rag.prompts.prompts import get_prompt, QUESTION_TYPES
        from app.rag.validator import validate_quiz, normalize_quiz

        qtype = force_type or random.choice(["situation", "concept"])
        topic = concept_topic or keyword or "핵심 개념"

        # 정답 위치 랜덤 선택
        target_position = random.randint(0, 3)

        # 질문 형식 랜덤 선택
        question_type = None
        if qtype == "concept":
            question_type = random.choice(QUESTION_TYPES)

        # FACTS 생성 (다양한 스니펫 사용)
        context = self._mk_facts(snips, max_chars=1200)

        # 이전 생성 문제 정보
        previous_questions_hint = ""
        if self.generated_questions:
            # 최근 3개 문제 샘플
            recent = list(self.generated_questions)[-3:]
            previous_questions_hint = (
                    "\n\n**중요: 다음 문제들과 중복되지 않도록 완전히 다른 상황/개념으로 출제:**\n"
                    + "\n".join([f"- {q[:80]}..." for q in recent])
            )

        # LLM 호출
        max_retry = 1
        previous_errors = []

        for attempt in range(max_retry):
            try:
                # 프롬프트 생성
                error_feedback = ""
                if previous_errors:
                    error_feedback = f"\n\n**이전 오류**: {', '.join(previous_errors[-2:])}"

                prompt = get_prompt(
                    qtype=qtype,
                    keyword=keyword or "성교육",
                    topic=topic,
                    context=context,
                    correct_position=target_position,
                    question_type=question_type
                ) + error_feedback + previous_questions_hint

                # LLM 호출
                response = self._call_llm(prompt)
                item = self._parse_json(response)

                # 정규화
                item = normalize_quiz(item)

                # 중복 체크
                question_hash = self._hash_question(item["question"])
                if question_hash in self.generated_questions:
                    print(f"[시도 {attempt + 1}/{max_retry}] 중복 문제 감지, 재생성")
                    previous_errors.append("중복 문제")
                    continue

                # 정답 위치 강제 조정
                actual_position = item.get("correct_index", 0)
                if actual_position != target_position:
                    print(f"[자동 수정] 정답 위치: {actual_position} → {target_position}")
                    item = self._fix_answer_position(item, actual_position, target_position)

                # 검증
                is_valid, errors = validate_quiz(item, context)

                if is_valid:
                    # 메타데이터 추가
                    item["type"] = qtype
                    item["keyword"] = keyword
                    item["topic"] = topic
                    item["question_type"] = question_type
                    item["sources"] = self._extract_sources(snips)
                    item["correct_label"] = ["A", "B", "C", "D"][item["correct_index"]]

                    # 캐시에 추가
                    self.generated_questions.add(question_hash)

                    return item
                else:
                    previous_errors = errors
                    print(f"[시도 {attempt + 1}/{max_retry}] 검증 실패: {', '.join(errors)}")

            except Exception as e:
                print(f"[시도 {attempt + 1}/{max_retry}] 생성 실패: {e}")
                previous_errors.append(str(e))

        # 폴백
        print(f"[경고] {max_retry}회 시도 실패. 기본 문제 생성")
        return self._create_fallback_item(qtype, keyword, topic, snips, target_position)

    def _hash_question(self, question: str) -> str:
        """문제 해시 생성 (중복 체크용)"""
        # 공백과 특수문자 제거 후 해시
        normalized = "".join(question.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _fix_answer_position(
            self,
            item: Dict[str, Any],
            current_pos: int,
            target_pos: int
    ) -> Dict[str, Any]:
        """정답 위치를 강제로 조정"""
        choices = item.get("choices", [])
        if len(choices) != 4:
            return item

        # 보기 순서 변경
        new_choices = choices.copy()

        # current_pos에 있는 정답을 target_pos로 이동
        answer_choice = new_choices.pop(current_pos)
        new_choices.insert(target_pos, answer_choice)

        item["choices"] = new_choices
        item["correct_index"] = target_pos

        return item

    def _extract_sources(self, snips: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """소스 정보 추출"""
        sources = []
        for s in snips[:2]:
            sources.append({
                "source": str(s.get("source", "")),
                "chunk_id": str(s.get("chunk_id", ""))
            })
        return sources

    def _call_llm(self, prompt: str) -> str:
        """LLM 호출"""
        try:
            r = self.client.responses.create(
                model=self.cfg.gen_model,
                input=prompt,
                response_format={"type": "json_object"},
                max_output_tokens=1000
            )
            return r.output[0].content[0].text.value
        except:
            r = self.client.chat.completions.create(
                model=self.cfg.gen_model,
                messages=[
                    {"role": "system", "content": "JSON 형식으로만 답변하세요."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return r.choices[0].message.content

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """JSON 파싱"""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)

    def _mk_facts(self, snips: List[Dict[str, Any]], max_chars: int = 1200) -> str:
        """RAG 스니펫을 FACTS 형식으로 변환"""
        facts = []
        total_chars = 0

        for snip in snips:
            text = snip.get("text", "")
            if total_chars + len(text) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    text = text[:remaining] + "..."
                else:
                    break

            facts.append(f"- {text}")
            total_chars += len(text)

        return "\n".join(facts)

    def _create_fallback_item(
            self,
            qtype: str,
            keyword: Optional[str],
            topic: str,
            snips: List[Dict[str, Any]],
            correct_position: int
    ) -> Dict[str, Any]:
        """폴백: LLM 실패 시 기본 문제 생성"""
        choices_template = [
            f"{topic}에 대한 부정확한 설명이다.",
            f"{topic}의 일부 특징만 언급한 불완전한 설명이다.",
            f"{topic}와 혼동하기 쉬운 잘못된 설명이다.",
            f"{topic}에 대한 정확한 설명으로, 주요 특징과 효과를 포함하고 있다. 이는 교육적으로 검증된 정보이므로 신뢰할 수 있다."
        ]

        # 정답을 지정된 위치로 이동
        correct_answer = choices_template.pop(3)
        choices_template.insert(correct_position, correct_answer)

        question = f"{topic}에 대한 설명으로 옳은 것은? (생성 ID: {random.randint(1000, 9999)})"

        # 캐시에 추가
        question_hash = self._hash_question(question)
        self.generated_questions.add(question_hash)

        return {
            "type": qtype,
            "question": question,
            "choices": choices_template,
            "correct_index": correct_position,
            "correct_label": ["A", "B", "C", "D"][correct_position],
            "explanation": f"{topic}에 대한 정확한 이해는 안전한 성생활의 기초입니다. 올바른 정보를 바탕으로 현명한 선택을 할 수 있습니다.",
            "keyword": keyword,
            "topic": topic,
            "is_fallback": True,
            "sources": self._extract_sources(snips)
        }


def get_service() -> ScenarioService:
    """서비스 팩토리"""
    cfg = Config()
    cfg.index_root = os.getenv("SCENARIO_INDEX_ROOT", "data/indexes")
    cfg.topk = int(os.getenv("SCENARIO_TOPK", "12"))  # 더 많이 가져오기
    cfg.gen_model = os.getenv("GEN_MODEL", "gpt-4o-mini")
    cfg.diversity_threshold = float(os.getenv("DIVERSITY_THRESHOLD", "0.7"))
    return ScenarioService(cfg)
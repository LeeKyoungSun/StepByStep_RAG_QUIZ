#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
퀴즈 생성 및 DB 저장 스크립트
- 정답 위치 랜덤화
- 해설 포함
- RDS 직접 저장
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from app.scenarios.service import get_service
from app.rag.validator import format_quiz_display


def save_to_json(items: List[Dict[str, Any]], filepath: str) -> None:
    """JSON 파일로 저장"""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"✓ JSON 저장: {out_path}")


def save_to_database(items: List[Dict[str, Any]], scenario_title: str) -> None:
    """
    RDS에 퀴즈 저장

    DB 구조:
    - quiz_scenario: 시나리오 메타데이터
    - quiz_question: 질문 (stem, correct_text, scenario_id)
    - quiz_option: 선택지 (text, label, is_correct, question_id)
    """
    try:
        from app.db import SessionLocal
        from app.models.quiz import QuizScenario, QuizQuestion, QuizOption

        db = SessionLocal()

        try:
            # 1. 시나리오 생성
            scenario = QuizScenario(title=scenario_title)
            db.add(scenario)
            db.flush()  # scenario.id 생성

            print(f"✓ 시나리오 생성: {scenario.title} (ID: {scenario.id})")

            # 2. 각 문제 저장
            for idx, item in enumerate(items, 1):
                # 질문 생성
                question = QuizQuestion(
                    stem=item["question"],
                    correct_text=item.get("explanation", ""),
                    scenario_id=scenario.id
                )
                db.add(question)
                db.flush()  # question.id 생성

                # 선택지 생성
                labels = ["A", "B", "C", "D"]
                correct_idx = item["correct_index"]

                for i, choice_text in enumerate(item["choices"]):
                    option = QuizOption(
                        text=choice_text,
                        label=labels[i],
                        is_correct=(i == correct_idx),
                        question_id=question.id
                    )
                    db.add(option)

                print(f"  [{idx}/{len(items)}] 문제 저장: {item['question'][:50]}...")

            # 3. 커밋
            db.commit()
            print(f"\n✓ DB 저장 완료: {len(items)}개 문제")
            print(f"  - Scenario ID: {scenario.id}")
            print(f"  - 제목: {scenario_title}")

        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    except ImportError as e:
        print(f"[오류] DB 모듈을 찾을 수 없습니다: {e}")
        print("  - app.db.SessionLocal 확인")
        print("  - app.models.quiz 확인")
    except Exception as e:
        print(f"[오류] DB 저장 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="성교육 퀴즈 생성")
    parser.add_argument("--mode", choices=["random", "by_keyword"], default="by_keyword")
    parser.add_argument("--keyword", default="피임", help="검색 키워드")
    parser.add_argument("--num", "-n", type=int, default=5, help="생성할 문제 수")
    parser.add_argument("--force_type", choices=["situation", "concept"], default=None,
                        help="문제 유형 강제 (없으면 랜덤)")
    parser.add_argument("--concept_topic", default=None, help="개념형 주제")
    parser.add_argument("--preview", action="store_true", default=True, help="생성 즉시 출력")
    parser.add_argument("--save_json", action="store_true", help="JSON 파일로 저장")
    parser.add_argument("--out", default="data/quiz_output.json", help="JSON 저장 경로")
    parser.add_argument("--save_db", action="store_true", help="RDS에 저장")
    parser.add_argument("--scenario_title", default=None, help="시나리오 제목 (DB 저장 시)")

    args = parser.parse_args()

    # 서비스 초기화
    print("=" * 60)
    print("퀴즈 생성 시작")
    print("=" * 60)
    print(f"키워드: {args.keyword}")
    print(f"문제 수: {args.num}")
    print(f"유형: {args.force_type or '랜덤'}")
    print("=" * 60)

    svc = get_service()

    # 퀴즈 생성
    results = []
    for i in range(args.num):
        print(f"\n[{i + 1}/{args.num}] 생성 중...")

        # 스니펫 가져오기
        if args.mode == "by_keyword" and args.keyword:
            snips = svc.pick_by_keyword(args.keyword, svc.cfg.topk)
        else:
            snips = svc.random_snippets(svc.cfg.topk)

        if not snips:
            print(f"[경고] 관련 자료를 찾을 수 없습니다: {args.keyword}")
            continue

        # 문제 생성
        item = svc.make_quiz_item(
            keyword=args.keyword,
            snips=snips,
            force_type=args.force_type,
            concept_topic=args.concept_topic
        )

        if item:
            results.append(item)

            # 즉시 출력
            if args.preview:
                print(format_quiz_display(item, item.get("type", "situation")))

    # 결과 요약
    print(f"\n{'=' * 60}")
    print(f"생성 완료: {len(results)}/{args.num}개")
    print(f"{'=' * 60}\n")

    if not results:
        print("[경고] 생성된 문제가 없습니다.")
        return

    # 정답 위치 분포 출력
    correct_positions = [item["correct_index"] for item in results]
    print("정답 위치 분포:")
    for label, idx in zip(["A", "B", "C", "D"], range(4)):
        count = correct_positions.count(idx)
        print(f"  {label}: {count}개")
    print()

    # JSON 저장
    if args.save_json:
        save_to_json(results, args.out)

    # DB 저장
    if args.save_db:
        scenario_title = args.scenario_title or f"{args.keyword} 퀴즈"
        save_to_database(results, scenario_title)


if __name__ == "__main__":
    main()

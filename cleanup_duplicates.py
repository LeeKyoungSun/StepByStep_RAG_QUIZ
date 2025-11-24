#!/usr/bin/env python3
"""
중복 퀴즈 감지 및 삭제 스크립트

사용법:
  python cleanup_duplicates.py --dry-run  # 중복 확인만
  python cleanup_duplicates.py --delete   # 중복 삭제
  python cleanup_duplicates.py --interactive  # 하나씩 확인하며 삭제
"""
import argparse
import hashlib
from collections import defaultdict
from typing import List, Dict, Set
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
import os
from datetime import datetime

# 모델 import
import sys

sys.path.append('/home/claude')
from app.models.quiz import QuizQuestion, QuizScenario, QuizAttempt, QuizResponse
from app.db import SessionLocal


def hash_question(stem: str) -> str:
    """질문 해시 생성 (중복 체크용)"""
    # 공백과 특수문자 제거 후 해시
    normalized = "".join(stem.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def find_duplicates(db: Session, similarity_threshold: float = 0.9) -> Dict[str, List[int]]:
    """
    중복 질문 찾기

    Returns:
        {question_hash: [question_id1, question_id2, ...]}
    """
    print(" 중복 질문 검색 중...")

    # 모든 질문 조회
    questions = db.query(QuizQuestion).all()
    print(f"총 {len(questions)}개 질문 분석 중...")

    # 해시별로 그룹화
    hash_to_ids: Dict[str, List[int]] = defaultdict(list)

    for q in questions:
        q_hash = hash_question(q.stem)
        hash_to_ids[q_hash].append(q.id)

    # 중복만 필터링 (2개 이상)
    duplicates = {
        h: ids for h, ids in hash_to_ids.items()
        if len(ids) > 1
    }

    return duplicates


def analyze_duplicates(db: Session, duplicates: Dict[str, List[int]]):
    """중복 통계 분석"""
    print("\n" + "=" * 80)
    print(" 중복 분석 결과")
    print("=" * 80)

    total_duplicates = sum(len(ids) - 1 for ids in duplicates.values())
    unique_questions = len(duplicates)

    print(f" 중복된 고유 질문 수: {unique_questions}개")
    print(f" 삭제 대상 질문 수: {total_duplicates}개")
    print(f" 유지할 질문 수: {unique_questions}개")

    # 상위 10개 중복 질문 표시
    print("\n 중복이 많은 질문 Top 10:")
    print("-" * 80)

    sorted_dups = sorted(
        duplicates.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]

    for i, (q_hash, ids) in enumerate(sorted_dups, 1):
        # 첫 번째 질문 내용 가져오기
        first_q = db.query(QuizQuestion).filter(
            QuizQuestion.id == ids[0]
        ).first()

        stem_preview = first_q.stem[:80] + "..." if len(first_q.stem) > 80 else first_q.stem
        print(f"{i:2d}. [{len(ids)}개 중복] {stem_preview}")
        print(f"    IDs: {ids[:5]}{'...' if len(ids) > 5 else ''}")

    return total_duplicates, unique_questions


def get_question_usage(db: Session, question_id: int) -> Dict:
    """질문 사용 정보 조회"""
    # 응답 수
    response_count = db.query(func.count(QuizResponse.question_id)).filter(
        QuizResponse.question_id == question_id
    ).scalar()

    # 시도 수 (해당 질문이 포함된 시도)
    attempt_count = db.query(func.count(func.distinct(QuizResponse.attempt_id))).filter(
        QuizResponse.question_id == question_id
    ).scalar()

    # 질문 정보
    question = db.query(QuizQuestion).filter(
        QuizQuestion.id == question_id
    ).first()

    return {
        'id': question_id,
        'response_count': response_count,
        'attempt_count': attempt_count,
        'scenario_id': question.scenario_id if question else None
    }


def select_keep_question(db: Session, question_ids: List[int]) -> int:
    """
    중복 질문 중 유지할 질문 선택

    기준:
    1. 가장 많이 사용된 질문 (응답 수)
    2. 가장 오래된 질문 (ID가 작은)
    """
    usages = [get_question_usage(db, qid) for qid in question_ids]

    # 응답이 있는 질문 우선
    with_responses = [u for u in usages if u['response_count'] > 0]

    if with_responses:
        # 가장 많이 사용된 질문
        keep = max(with_responses, key=lambda x: (x['response_count'], -x['id']))
    else:
        # 모두 사용 안 됨 → 가장 오래된 질문 (ID가 작은)
        keep = min(usages, key=lambda x: x['id'])

    return keep['id']


def delete_duplicates(
        db: Session,
        duplicates: Dict[str, List[int]],
        dry_run: bool = True,
        interactive: bool = False
) -> int:
    """
    중복 질문 삭제

    Args:
        db: DB 세션
        duplicates: 중복 질문 맵
        dry_run: True면 실제 삭제 안 함
        interactive: True면 하나씩 확인하며 삭제

    Returns:
        삭제된 질문 수
    """
    deleted_count = 0

    print("\n" + "=" * 80)
    print(f"{' [DRY RUN] 삭제 시뮬레이션' if dry_run else '❌ 중복 질문 삭제 중...'}")
    print("=" * 80)

    for q_hash, ids in duplicates.items():
        # 유지할 질문 선택
        keep_id = select_keep_question(db, ids)
        delete_ids = [qid for qid in ids if qid != keep_id]

        # 질문 내용 가져오기
        keep_q = db.query(QuizQuestion).filter(QuizQuestion.id == keep_id).first()
        stem_preview = keep_q.stem[:80] + "..." if len(keep_q.stem) > 80 else keep_q.stem

        print(f"\n 질문: {stem_preview}")
        print(f"   총 {len(ids)}개 중복 발견")
        print(f"   유지: ID {keep_id}")
        print(f"   삭제: {delete_ids}")

        # Interactive 모드
        if interactive and not dry_run:
            response = input("   이 질문들을 삭제하시겠습니까? (y/N): ")
            if response.lower() != 'y':
                print("   ️  건너뜀")
                continue

        # 삭제 실행
        if not dry_run:
            for del_id in delete_ids:
                try:
                    # CASCADE 삭제 (options, responses 모두 삭제됨)
                    question = db.query(QuizQuestion).filter(
                        QuizQuestion.id == del_id
                    ).first()

                    if question:
                        db.delete(question)
                        deleted_count += 1
                        print(f"   삭제됨: ID {del_id}")
                except Exception as e:
                    print(f"   삭제 실패: ID {del_id} - {e}")

            # 커밋
            try:
                db.commit()
                print(f"    커밋 완료")
            except Exception as e:
                print(f"   ✗ 커밋 실패: {e}")
                db.rollback()
        else:
            deleted_count += len(delete_ids)

    return deleted_count


def cleanup_empty_scenarios(db: Session, dry_run: bool = True) -> int:
    """
    질문이 없는 빈 시나리오 삭제
    """
    print("\n" + "=" * 80)
    print(f"{' [DRY RUN] 빈 시나리오 확인' if dry_run else '🗑️  빈 시나리오 삭제 중...'}")
    print("=" * 80)

    # 질문이 없는 시나리오 찾기
    empty_scenarios = db.query(QuizScenario).filter(
        ~QuizScenario.questions.any()
    ).all()

    print(f"빈 시나리오 {len(empty_scenarios)}개 발견")

    if not dry_run:
        for scenario in empty_scenarios:
            # 사용된 적 있는지 확인 (attempt가 있는지)
            attempt_count = db.query(func.count(QuizAttempt.id)).filter(
                QuizAttempt.scenario_id == scenario.id
            ).scalar()

            if attempt_count > 0:
                print(f"     건너뜀: Scenario #{scenario.id} (사용 기록 {attempt_count}개)")
            else:
                db.delete(scenario)
                print(f"   ✓ 삭제됨: Scenario #{scenario.id} '{scenario.title}'")

        db.commit()
        print(f" 커밋 완료")

    return len(empty_scenarios)


def main():
    parser = argparse.ArgumentParser(description="중복 퀴즈 정리 도구")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="실제 삭제 없이 시뮬레이션만 수행"
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help="중복 질문 삭제 실행"
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help="하나씩 확인하며 삭제"
    )
    parser.add_argument(
        '--clean-scenarios',
        action='store_true',
        help="빈 시나리오도 함께 삭제"
    )

    args = parser.parse_args()

    # 기본은 dry-run
    dry_run = not args.delete

    print(" 중복 퀴즈 정리 도구")
    print("=" * 80)
    print(f"모드: {'🔍 DRY RUN (시뮬레이션)' if dry_run else ' DELETE (실제 삭제)'}")
    print(f"Interactive: {'✓' if args.interactive else '✗'}")
    print("=" * 80)

    # DB 연결
    db = SessionLocal()

    try:
        # 1. 중복 찾기
        duplicates = find_duplicates(db)

        if not duplicates:
            print("\n 중복 질문이 없습니다!")
            return

        # 2. 통계 분석
        total_dups, unique = analyze_duplicates(db, duplicates)

        # 3. 삭제 확인
        if not dry_run and not args.interactive:
            print(f"\n  경고: {total_dups}개의 질문이 삭제됩니다!")
            response = input("계속하시겠습니까? (yes/no): ")
            if response.lower() != 'yes':
                print(" 취소됨")
                return

        # 4. 중복 삭제
        deleted = delete_duplicates(db, duplicates, dry_run, args.interactive)

        # 5. 빈 시나리오 삭제
        if args.clean_scenarios:
            cleanup_empty_scenarios(db, dry_run)

        # 6. 결과 요약
        print("\n" + "=" * 80)
        print(" 완료")
        print("=" * 80)
        print(f"중복 그룹: {len(duplicates)}개")
        print(f"{'삭제 예정' if dry_run else '삭제 완료'}: {deleted}개")
        print(f"유지: {len(duplicates)}개")

        if dry_run:
            print("\n 실제 삭제하려면: python cleanup_duplicates.py --delete")

    finally:
        db.close()


if __name__ == "__main__":
    main()
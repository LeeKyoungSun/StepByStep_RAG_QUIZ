-- ============================================================================
-- 중복 퀴즈 정리 SQL 스크립트
-- ============================================================================

-- 1. 중복 질문 확인 (같은 질문이 여러 개)
-- ============================================================================
SELECT
    stem,
    COUNT(*) as duplicate_count,
    GROUP_CONCAT(id ORDER BY id) as question_ids,
    MIN(id) as keep_id,
    GROUP_CONCAT(id ORDER BY id) as delete_ids
FROM quiz_question
GROUP BY stem
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC
LIMIT 20;

-- 2. 중복 질문 통계
-- ============================================================================
SELECT
    '총 질문 수' as metric,
    COUNT(*) as value
FROM quiz_question

UNION ALL

SELECT
    '고유 질문 수' as metric,
    COUNT(DISTINCT stem) as value
FROM quiz_question

UNION ALL

SELECT
    '중복 질문 그룹 수' as metric,
    COUNT(*) as value
FROM (
    SELECT stem
    FROM quiz_question
    GROUP BY stem
    HAVING COUNT(*) > 1
) as duplicates

UNION ALL

SELECT
    '삭제 대상 질문 수' as metric,
    SUM(cnt - 1) as value
FROM (
    SELECT COUNT(*) as cnt
    FROM quiz_question
    GROUP BY stem
    HAVING COUNT(*) > 1
) as dup_counts;

-- 3. 응답이 많은 질문 확인 (삭제하면 안 되는 질문)
-- ============================================================================
SELECT
    q.id,
    q.stem,
    COUNT(DISTINCT r.attempt_id) as attempt_count,
    COUNT(r.question_id) as response_count
FROM quiz_question q
LEFT JOIN quiz_response r ON q.id = r.question_id
GROUP BY q.id, q.stem
HAVING response_count > 0
ORDER BY response_count DESC
LIMIT 20;

-- 4. 중복 질문 중 유지할 질문 선택 로직
-- ============================================================================
-- 각 중복 그룹에서 유지할 질문 선택
-- 기준: 1) 응답이 있는 질문, 2) 가장 오래된 질문 (ID가 작은)

WITH duplicate_groups AS (
    SELECT
        stem,
        id,
        ROW_NUMBER() OVER (PARTITION BY stem ORDER BY
            -- 응답이 있으면 우선
            (SELECT COUNT(*) FROM quiz_response WHERE question_id = quiz_question.id) DESC,
            -- 같으면 ID가 작은 것
            id ASC
        ) as rn
    FROM quiz_question
    WHERE stem IN (
        SELECT stem
        FROM quiz_question
        GROUP BY stem
        HAVING COUNT(*) > 1
    )
)
SELECT
    stem,
    id as question_id,
    CASE WHEN rn = 1 THEN ' KEEP' ELSE ' DELETE' END as action,
    (SELECT COUNT(*) FROM quiz_response WHERE question_id = duplicate_groups.id) as response_count
FROM duplicate_groups
ORDER BY stem, rn;

-- 5. 중복 질문 삭제 (주의: 실행 전 백업 필수!)
-- ============================================================================
--  경고: 아래 쿼리는 실제 데이터를 삭제합니다!
-- 실행 전 반드시 데이터베이스 백업

-- 5-1. 백업 테이블 생성 (권장)
CREATE TABLE IF NOT EXISTS quiz_question_backup AS
SELECT * FROM quiz_question;

-- 5-2. 중복 질문 삭제 (응답이 없는 중복만 삭제)
-- 각 중복 그룹에서 첫 번째(가장 오래된) 질문만 남기고 나머지 삭제
DELETE q
FROM quiz_question q
WHERE q.id IN (
    SELECT id FROM (
        SELECT
            id,
            stem,
            ROW_NUMBER() OVER (PARTITION BY stem ORDER BY
                (SELECT COUNT(*) FROM quiz_response WHERE question_id = quiz_question.id) DESC,
                id ASC
            ) as rn
        FROM quiz_question
        WHERE stem IN (
            SELECT stem
            FROM quiz_question
            GROUP BY stem
            HAVING COUNT(*) > 1
        )
    ) as dup
    WHERE rn > 1
    AND NOT EXISTS (
        -- 응답이 있는 질문은 삭제하지 않음
        SELECT 1 FROM quiz_response
        WHERE question_id = dup.id
    )
);

-- 5-3. 더 안전한 방법: 단계별 삭제
-- Step 1: 삭제 대상 확인
SELECT
    q.id,
    q.stem,
    q.scenario_id,
    (SELECT COUNT(*) FROM quiz_response WHERE question_id = q.id) as has_responses
FROM quiz_question q
WHERE q.id IN (
    SELECT id FROM (
        SELECT
            id,
            ROW_NUMBER() OVER (PARTITION BY stem ORDER BY id) as rn
        FROM quiz_question
        WHERE stem IN (
            SELECT stem FROM quiz_question
            GROUP BY stem HAVING COUNT(*) > 1
        )
    ) as dup
    WHERE rn > 1
)
AND NOT EXISTS (
    SELECT 1 FROM quiz_response WHERE question_id = q.id
)
ORDER BY q.stem, q.id;

-- Step 2: 위 결과 확인 후 실제 삭제 (응답 없는 중복만)
-- DELETE FROM quiz_question WHERE id IN (...);

-- 6. 빈 시나리오 정리
-- ============================================================================

-- 6-1. 질문이 없는 시나리오 확인
SELECT
    s.id,
    s.title,
    (SELECT COUNT(*) FROM quiz_question WHERE scenario_id = s.id) as question_count,
    (SELECT COUNT(*) FROM quiz_attempt WHERE scenario_id = s.id) as attempt_count
FROM quiz_scenario s
WHERE NOT EXISTS (
    SELECT 1 FROM quiz_question WHERE scenario_id = s.id
)
ORDER BY s.id;

-- 6-2. 질문도 없고 사용 기록도 없는 시나리오 삭제
DELETE FROM quiz_scenario
WHERE id IN (
    SELECT s.id
    FROM quiz_scenario s
    WHERE NOT EXISTS (
        SELECT 1 FROM quiz_question WHERE scenario_id = s.id
    )
    AND NOT EXISTS (
        SELECT 1 FROM quiz_attempt WHERE scenario_id = s.id
    )
);

-- 7. 검증 쿼리
-- ============================================================================

-- 중복이 남아있는지 확인
SELECT
    'Remaining Duplicates' as check_type,
    COUNT(*) as count
FROM (
    SELECT stem
    FROM quiz_question
    GROUP BY stem
    HAVING COUNT(*) > 1
) as dups;

-- 데이터 무결성 확인
SELECT
    'Orphaned Options' as check_type,
    COUNT(*) as count
FROM quiz_option
WHERE question_id NOT IN (SELECT id FROM quiz_question);

SELECT
    'Orphaned Responses' as check_type,
    COUNT(*) as count
FROM quiz_response
WHERE question_id NOT IN (SELECT id FROM quiz_question);

-- ============================================================================
-- 사용 예시
-- ============================================================================

-- 1. 먼저 중복 확인 (섹션 1 실행)
-- 2. 통계 확인 (섹션 2 실행)
-- 3. 응답이 많은 질문 확인 (섹션 3 실행)
-- 4. 삭제 대상 확인 (섹션 5-3 Step 1 실행)
-- 5. 백업 생성 (섹션 5-1 실행)
-- 6. 삭제 실행 (섹션 5-2 또는 5-3 Step 2 실행)
-- 7. 빈 시나리오 정리 (섹션 6 실행)
-- 8. 검증 (섹션 7 실행)
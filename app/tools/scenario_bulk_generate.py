# app/tools/scenario_bulk_generate.py
import os, json, argparse
from pathlib import Path
from app.scenarios.service import get_service

def main():
    p = argparse.ArgumentParser(description="시나리오 퀴즈 대량 생성 CLI")
    p.add_argument("--mode", choices=["random", "by_keyword"], default="random")
    p.add_argument("--keyword", default=None, help="--mode=by_keyword일 때 키워드")
    p.add_argument("-n", "--num", type=int, default=10, help="생성 문항 수")
    p.add_argument("-o", "--out", default="scenarios.jsonl", help="저장 경로(.jsonl)")
    args = p.parse_args()

    svc = get_service()
    data = svc.make_quiz(mode=args.mode, keyword=args.keyword, n=args.num)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] {len(data)}개 생성 → {out_path.resolve()}")

if __name__ == "__main__":
    main()
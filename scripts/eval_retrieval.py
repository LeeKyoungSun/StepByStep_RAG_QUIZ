#!/usr/bin/env python3
import os, sys, json, argparse, csv, math
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

# --- 내부 의존 (네 프로젝트 경로 그대로) ---
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 프로젝트 루트로
from app.core.stores.faiss_store import FaissStore
from app.core.stores.bm25_store import BM25Store
from app.rag.retrievers.rrf import rrf_fuse


def load_qa(fp: Path) -> List[Dict]:
    rows = []
    for ln in fp.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rows.append(json.loads(ln))
    return rows


def keyword_hit(doc_text: str, gold_keywords: List[str]) -> bool:
    t = (doc_text or "").lower()
    return any(kw.lower() in t for kw in gold_keywords)


def compute_metrics(hits: List[List[bool]]) -> Tuple[float, float, float]:
    """
    hits: per query -> [True/False ...] by rank
    Recall@k, MRR@k, nDCG@k
    """
    n = len(hits)
    recall = sum(any(h) for h in hits) / n if n else 0.0
    mrr = 0.0
    ndcg = 0.0
    for hs in hits:
        # MRR
        rank = next((i + 1 for i, v in enumerate(hs) if v), None)
        if rank: mrr += 1.0 / rank
        # nDCG
        dcg = sum((1.0 / math.log2(i + 2)) for i, v in enumerate(hs) if v)
        idcg = 1.0  # 단정답 가정(키워드 매치 1개만 중요)
        ndcg += (dcg / idcg if idcg > 0 else 0.0)
    return recall, mrr / n if n else 0.0, ndcg / n if n else 0.0


def search_faiss(store: FaissStore, q: str, k: int) -> List[Dict]:
    return store.search(q, top_k=k)


def search_bm25(stores: List[BM25Store], q: str, k: int) -> List[Dict]:
    bags = []
    for st in stores:
        rows = st.search(q, top_k=k)
        bags.append(rows)
    # dict -> LangChain Document 로 바꾸기 전에 RRF용 최소키만 보존
    # 간단히 동일 키 함수로 dedup 되도록 맞춤
    from langchain_core.documents import Document
    def to_doc(r):
        meta = dict(r)
        text = meta.pop("text", "") or meta.pop("content", "") or meta.pop("chunk", "")
        return Document(page_content=text, metadata=meta)

    bags_docs = [[to_doc(r) for r in rows] for rows in bags]
    fused_docs = rrf_fuse(bags_docs, k=int(os.getenv("RRF_K", "60")), top_k=k)
    # 다시 dict로
    out = []
    for d in fused_docs:
        m = dict(d.metadata);
        m["text"] = d.page_content
        out.append(m)
    return out


def rrf_weighted(faiss_rows: List[Dict], bm25_rows: List[Dict], k: int, k_rrf: int, w_f: float, w_b: float) -> List[
    Dict]:
    """
    간단한 가중-RRF: 각 리스트 내 rank 기반 점수(1/(k_rrf+rank))에 가중치 곱해 합산 → top-k.
    동일 문서 판단은 (source, chunk_id) 또는 doc_id 기준.
    """

    def key_of(r: Dict):
        md = r
        return md.get("doc_id") or (md.get("source"), md.get("chunk_id"))

    scores = {}
    seen = {}
    order = 0
    for lst, w in ((faiss_rows, w_f), (bm25_rows, w_b)):
        for i, r in enumerate(lst):
            key = key_of(r)
            if key not in scores:
                scores[key] = 0.0
                seen[key] = (order, r)  # 최초 등장 보관
                order += 1
            scores[key] += w * (1.0 / (k_rrf + (i + 1)))
    top = sorted(scores.items(), key=lambda x: (-x[1], seen[x[0]][0]))[:k]
    return [seen[k][1] for k, _ in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="eval/*.jsonl")
    ap.add_argument("--modes", default="faiss,bm25,rrf,rrfw", help="comma sep: faiss,bm25,rrf,rrfw")
    ap.add_argument("--k", default="3,5", help="comma sep top-k")
    ap.add_argument("--csv", default=None, help="write metrics to csv")
    ap.add_argument("--debug", action="store_true", help="print top docs & hit flags")
    # 튜닝 파라미터
    ap.add_argument("--rrf-k", type=int, default=int(os.getenv("RRF_K", "60")))
    ap.add_argument("--w-faiss", type=float, default=0.7)
    ap.add_argument("--w-bm25", type=float, default=0.3)
    args = ap.parse_args()

    qa = load_qa(Path(args.qa))
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # 스토어 로드
    faiss_dir = os.getenv("FAISS_DIR", "./data/indexes/merged/faiss")
    faiss_store = FaissStore.load(faiss_dir)
    # BM25 여러 디렉토리
    bm25_dirs = []
    if os.getenv("BM25_DIRS"):
        bm25_dirs = [d.strip() for d in os.getenv("BM25_DIRS").split(",") if d.strip()]
    elif os.getenv("BM25_DIRS_FILE"):
        bm25_dirs = [ln.strip() for ln in Path(os.getenv("BM25_DIRS_FILE")).read_text(encoding="utf-8").splitlines() if
                     ln.strip()]
    bm25_stores = [BM25Store.load(d) for d in bm25_dirs]

    rows_out = []
    for mode in modes:
        for k in ks:
            per_hits = []
            if args.debug:
                print(f"\n=== DEBUG [{mode}] k={k} ===")

            for i, item in enumerate(qa, 1):
                q = item.get("q") or item.get("question")  # 추가
                gold = item.get("gold_keywords") or item.get("doc_keywords")  # 추가
                if q is None or gold is None:
                    print(f"[WARN] QA #{i} missing keys: {item}")
                    continue

                if mode == "faiss":
                    docs = search_faiss(faiss_store, q, k)
                elif mode == "bm25":
                    docs = search_bm25(bm25_stores, q, k) if bm25_stores else []
                elif mode == "rrf":
                    # FAISS top-k + BM25 top-k 가져와서 vanilla RRF
                    f_docs = search_faiss(faiss_store, q, k)
                    b_docs = search_bm25(bm25_stores, q, k) if bm25_stores else []

                    # (이미 search_bm25에서 RRF가 걸려 있으므로 여기선 단순 합집합 정렬 유지)
                    docs = (f_docs + b_docs)[:k] if b_docs else f_docs
                elif mode == "rrfw":
                    f_docs = search_faiss(faiss_store, q, k)
                    b_docs = search_bm25(bm25_stores, q, k) if bm25_stores else []
                    docs = rrf_weighted(f_docs, b_docs, k=k, k_rrf=args.rrf_k, w_f=args.w_faiss, w_b=args.w_bm25)
                else:
                    docs = []

                flags = [keyword_hit(d.get("text", "") + " " + str(d.get("source", "")), gold) for d in docs]
                per_hits.append(flags)

                if args.debug:
                    print(f"\n[{i}] Q: {q}")
                    for j, d in enumerate(docs, 1):
                        s = d.get("source") or d.get("src") or ""
                        t = (d.get("text") or d.get("content") or "")[:90].replace("\n", " ")
                        print(f"  {j:>2}. {'✔' if flags[j - 1] else '✘'} {s} :: {t}")

            rec, mrr, ndcg = compute_metrics(per_hits)
            print(f"\n{mode:>5}  k={k:>2} | Recall@k={rec:.3f}  MRR@k={mrr:.3f}  nDCG@k={ndcg:.3f}")

            rows_out.append({
                "mode": mode, "k": k,
                "Recall@k": f"{rec:.3f}",
                "MRR@k": f"{mrr:.3f}",
                "nDCG@k": f"{ndcg:.3f}",
                "rrf_k": args.rrf_k,
                "w_faiss": args.w_faiss,
                "w_bm25": args.w_bm25,
            })
    print(f"[INFO] Loaded QA items: {len(qa)} from {args.qa}")
    if len(qa) == 0:
        print("[ERR] No valid QA items! Check format of --qa file (expect keys 'q' and 'gold_keywords').")
        sys.exit(1)

    if args.csv:
        outp = Path(args.csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n[OK] wrote {outp}")

if __name__ == "__main__":
    main()
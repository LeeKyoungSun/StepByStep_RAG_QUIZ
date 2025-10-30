#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    print("faiss import error:", e)
    sys.exit(1)


def load_meta_list(meta_json: Path, meta_jsonl: Path) -> List[dict]:
    """meta.json(list or dict) / meta.jsonl 둘 다 지원 -> list[dict]로 정규화"""
    if meta_json.exists():
        raw = json.loads(meta_json.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            items = []
            for k, v in raw.items():
                try:
                    i = int(k)
                except Exception:
                    continue
                items.append((i, v if isinstance(v, dict) else {"payload": v}))
            if not items:
                return []
            max_id = max(i for i, _ in items)
            out = [{} for _ in range(max_id + 1)]
            for i, v in items:
                if 0 <= i <= max_id:
                    out[i] = v
            return out
        raise ValueError("meta.json must be list or dict")
    if meta_jsonl.exists():
        lines = [
            json.loads(ln)
            for ln in meta_jsonl.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        return lines
    raise FileNotFoundError(f"meta.json(.jsonl) not found: {meta_json} / {meta_jsonl}")


def load_ids(ids_fp: Path, expect_len: Optional[int]) -> Tuple[np.ndarray, Optional[list]]:
    """
    returns: (numeric_ids, original_keys_if_non_numeric_or_None)
    """
    orig_keys = None
    ids = None
    if ids_fp.exists():
        try:
            ids = np.load(str(ids_fp), allow_pickle=True)
        except Exception as e:
            print(f"[WARN] failed to load ids.npy ({ids_fp}): {e}")

    if ids is None:
        n = int(expect_len or 0)
        return np.arange(n, dtype=np.int64), None

    # 1) 평탄화
    ids = np.asarray(ids).ravel()

    # 2) 전부 숫자로 변환 가능한지 확인
    try:
        num = ids.astype(np.int64, copy=False)
        if expect_len is not None and len(num) != expect_len:
            print(f"[WARN] ids length {len(num)} != meta length {expect_len} -> reindex 0..N-1")
            num = np.arange(int(expect_len), dtype=np.int64)
        return num, None
    except Exception:
        # 3) 문자열 키 → 순번 ID + 원본 키 보관
        print(f"[WARN] non-numeric ids detected in {ids_fp}, using sequential ids and keeping original keys")
        orig_keys = [str(x) for x in ids.tolist()]
        n = int(expect_len or len(orig_keys))
        num = np.arange(n, dtype=np.int64)
        if len(orig_keys) != n:
            orig_keys = (orig_keys + [""] * n)[:n]
        return num, orig_keys


def main():
    ap = argparse.ArgumentParser(description="Merge multiple FAISS indexes into one.")
    ap.add_argument("sources", nargs="+",
                    help="source faiss dirs (each contains index.faiss, meta.json/.jsonl, ids.npy?)")
    ap.add_argument("--out", required=True, help="output dir for merged faiss")
    args = ap.parse_args()

    src_dirs = [Path(s).resolve() for s in args.sources]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 준비(루프 전) : 여기서만 초기화 ---
    dim: Optional[int] = None
    metric_type: Optional[int] = None
    shards: List[faiss.Index] = []
    meta_all: List[dict] = []
    ids_all: List[int] = []
    global_id = 0

    print(f"[INFO] Scanning {len(src_dirs)} source indexes...")
    for sdir in src_dirs:
        idx_fp = sdir / "index.faiss"
        meta_json = sdir / "meta.json"
        meta_jsonl = sdir / "meta.jsonl"
        ids_fp = sdir / "ids.npy"

        if not idx_fp.exists():
            print(f"[ERR] missing index.faiss: {idx_fp}")
            return 2

        # faiss 로드 및 규격 확인
        idx = faiss.read_index(str(idx_fp))
        cur_dim = idx.d
        cur_metric = getattr(idx, "metric_type", faiss.METRIC_INNER_PRODUCT)

        if dim is None:
            dim = cur_dim
            metric_type = cur_metric
        else:
            if cur_dim != dim:
                print(f"[ERR] dim mismatch in {sdir}: {cur_dim} vs {dim}")
                return 3
            if cur_metric != metric_type:
                print(f"[ERR] metric mismatch in {sdir}: {cur_metric} vs {metric_type}")
                return 4

        # 메타/아이디 로드
        local_meta = load_meta_list(meta_json, meta_jsonl)
        local_ids, original_keys = load_ids(ids_fp, expect_len=len(local_meta))

        # 길이 안전장치 (faiss ntotal과도 맞춰보기)
        ntotal = getattr(idx, "ntotal", len(local_meta))
        if ntotal != len(local_meta):
            print(f"[WARN] faiss.ntotal({ntotal}) != meta({len(local_meta)}) -> pad/truncate meta & reset ids.")
            if len(local_meta) < ntotal:
                local_meta.extend({} for _ in range(ntotal - len(local_meta)))
            else:
                local_meta = local_meta[:ntotal]
            local_ids = np.arange(ntotal, dtype=np.int64)
            original_keys = None

        shards.append(idx)

        # 메타에 추적용 필드 추가 + global id 부여
        for i, m in enumerate(local_meta):
            m = m or {}
            m["_src_dir"] = str(sdir)
            if original_keys is not None and i < len(original_keys):
                m["_src_local_key"] = original_keys[i]
            lid = int(local_ids[i]) if i < len(local_ids) else i
            m["_src_local_id"] = lid
            m["_global_id"] = int(global_id)
            meta_all.append(m)
            ids_all.append(global_id)
            global_id += 1

    # --- 병합: metric_type 에 맞춰 '실제' 인덱스 생성 ---
    assert dim is not None, "No source index found."
    if metric_type == faiss.METRIC_L2:
        merged_index = faiss.IndexFlatL2(dim)
    else:
        merged_index = faiss.IndexFlatIP(dim)

    print(f"\n[INFO] All sources scanned. Starting merge process for {len(shards)} shards...")
    for i, idx in enumerate(shards):
        ntotal = idx.ntotal
        try:
            vectors = idx.reconstruct_n(0, ntotal)
        except Exception:
            # 일부 인덱스 타입에서 reconstruct_n 미지원 시, 느리지만 안전한 폴백
            vectors = np.vstack([idx.reconstruct(j) for j in range(ntotal)])
        # IndexFlat은 IP/L2 모두 add 시 float32 필요
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        merged_index.add(vectors)
        print(f"  -> Merged shard {i + 1}/{len(shards)}. Total vectors: {merged_index.ntotal}")

    # --- 저장 ---
    print(f"\n[INFO] Saving merged index to {out_dir / 'index.faiss'}...")
    faiss.write_index(merged_index, str(out_dir / "index.faiss"))

    np.save(str(out_dir / "ids.npy"), np.array(ids_all, dtype=np.int64))
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    first_model = {}
    first_model_fp = Path(src_dirs[0]) / "model.json"
    if first_model_fp.exists():
        first_model = json.loads(first_model_fp.read_text(encoding="utf-8"))
    first_model["_merged_sources"] = [str(s) for s in src_dirs]
    first_model["_merged_count"] = len(src_dirs)
    with (out_dir / "model.json").open("w", encoding="utf-8") as f:
        json.dump(first_model, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] merged {len(src_dirs)} indexes → {out_dir}")
    print(f"     total docs: {len(ids_all)}")


if __name__ == "__main__":
    sys.exit(main())
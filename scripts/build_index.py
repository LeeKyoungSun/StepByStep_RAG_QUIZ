#!/usr/bin/env python
# scripts/build_index.py
# 폴더/단일 txt → 정제 → 윈도우 청킹 → OpenAI 임베딩 → FAISS(IP=코사인) + BM25 + 메타 저장

import argparse, re, json, pathlib, pickle, time, os, sys, math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

# 내부 유틸
from utils.cleaning import clean_ocr_text, clean_for_bm25, sha1

# ---- OpenAI 임베딩 with 백오프
from openai import OpenAI
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _embed_openai(texts: List[str], model="text-embedding-3-small", normalize=True) -> np.ndarray:
    """
    안전 임베딩:
      - 배치 8
      - 요청 문자수 예산(≈토큰) 7,000
      - 길이 초과 자동 분할
      - 지수 백오프 재시도
    """
    BATCH = 8
    REQ_CHAR_BUDGET = 7000
    SPLIT_CHARS = 3500

    def _norm(v: np.ndarray) -> np.ndarray:
        if not normalize:
            return v
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def _call(batch: List[str], attempt=1) -> np.ndarray:
        try:
            resp = _client.embeddings.create(model=model, input=batch)
            arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
            return _norm(arr)
        except Exception as e:
            if attempt >= 5:
                raise
            delay = 1.5 * (1.8 ** (attempt - 1))
            time.sleep(delay)
            return _call(batch, attempt + 1)

    out: List[np.ndarray] = []
    i = 0
    while i < len(texts):
        cur: List[str] = []
        total_chars = 0
        # 문자수 예산 + 배치 한도 동시 만족
        while i < len(texts) and len(cur) < BATCH:
            t = texts[i]
            if len(t) > SPLIT_CHARS * 2:
                pieces = [t[k:k+SPLIT_CHARS] for k in range(0, len(t), SPLIT_CHARS)]
                out.append(_embed_openai(pieces, model=model, normalize=normalize))
                i += 1
                continue
            if total_chars + len(t) <= REQ_CHAR_BUDGET or not cur:
                cur.append(t); total_chars += len(t); i += 1
            else:
                break
        if cur:
            out.append(_call(cur))
    return np.vstack(out) if out else np.zeros((0, 1536), dtype=np.float32)


# ---- 파일/텍스트 유틸
def iter_txt_files(path):
    p = pathlib.Path(path)
    if p.is_file() and p.suffix.lower() == ".txt":
        yield p; return
    for f in p.rglob("*.txt"):
        yield f


def normalize_text(s: str) -> str:
    s = re.sub(r"\r\n|\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


@dataclass
class Chunk:
    chunk_id: int
    text: str
    start: int
    end: int


def slide_chunks(txt: str, min_chars=320, max_chars=640, stride_chars=520) -> List[Chunk]:
    ps = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
    chunks: List[Chunk] = []
    cid, start = 0, 0

    def flush(buf, start, cid):
        t = "\n\n".join(buf).strip()
        return Chunk(cid, t, start, start + len(t))

    buf: List[str] = []
    for p in ps:
        if len(p) > max_chars:
            s = 0
            while s < len(p):
                e = min(s + max_chars, len(p))
                t = p[s:e]
                chunks.append(Chunk(cid, t, start + s, start + e))
                cid += 1
                s += stride_chars
            start += len(p) + 2
            buf = []
            continue

        if sum(len(x) for x in buf) + len(buf)*2 + len(p) < max_chars:
            buf.append(p)
        else:
            if buf:
                ch = flush(buf, start, cid); chunks.append(ch); cid += 1
                start = ch.end + 2
            buf = [p]
    if buf:
        ch = flush(buf, start, cid); chunks.append(ch)

    # 짧은 청크 병합
    merged: List[Chunk] = []
    i = 0
    while i < len(chunks):
        if len(chunks[i].text) >= min_chars or i == len(chunks) - 1:
            merged.append(chunks[i]); i += 1; continue
        j = i + 1
        if j < len(chunks):
            t = (chunks[i].text + "\n\n" + chunks[j].text).strip()
            merged.append(Chunk(chunks[i].chunk_id, t, chunks[i].start, chunks[j].end))
            i += 2
        else:
            merged.append(chunks[i]); i += 1

    for k, ch in enumerate(merged):
        ch.chunk_id = k
    return merged


# ---- BM25 (간단 dict 저장 — 기존 하위 호환)
def build_bm25_corpus(docids: List[str], texts: List[str]) -> Dict[str, List[str]]:
    return {"docids": docids, "texts": texts}


def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)


def _lang_stats(txt: str) -> Tuple[int, int]:
    return (len(re.findall(r"[가-힣]", txt)), len(re.findall(r"[A-Za-z]", txt)))


def _make_doc_id(path: pathlib.Path) -> str:
    return f"{path.stem}-{sha1(str(path.resolve()))[:8]}"


# ---- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--name-suffix", default="_window")
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-clean", action="store_true")
    args = ap.parse_args()

    import faiss

    for fp in iter_txt_files(args.input):
        base = fp.stem
        out_dir = pathlib.Path(args.outdir) / f"{base}{args.name_suffix}"
        if (out_dir / "index.faiss").exists():
            print(f"[skip] {base} → already indexed"); continue
        ensure_dir(out_dir)

        raw = fp.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw if args.no_clean else clean_ocr_text(raw))

        if not text.strip():
            print(f"[warn] empty after cleaning: {fp}")
            (out_dir / "meta.json").write_text("[]", encoding="utf-8")
            continue

        chunks = slide_chunks(text, 320, 640, 520)
        print(f"[{base}] chunks={len(chunks)}")

        doc_id = _make_doc_id(fp)
        ids = [f"{base}.txt::chunk_{c.chunk_id}" for c in chunks]
        texts = [c.text for c in chunks]

        # ---- 임베딩 (L2 정규화: 코사인 유사도용)
        embeddings = _embed_openai(texts, model=args.model, normalize=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms

        # ---- FAISS (HNSW + IP = 코사인)
        d = embeddings.shape[1] if embeddings.size else 1536
        index = faiss.IndexHNSWFlat(d, 32)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64

        if embeddings.size:
            index.add(embeddings)
        faiss.write_index(index, str(out_dir / "index.faiss"))

        # ---- ids.npy (청크 ID 배열 저장)
        np.save(out_dir / "ids.npy", np.array(ids, dtype=object))

        # ---- meta.json
        created = int(time.time())
        meta = []
        for c in chunks:
            ko_cnt, en_cnt = _lang_stats(c.text)
            meta.append({
                "doc_id": doc_id,
                "file_name": f"{base}.txt",
                "source_path": str(fp.resolve()),
                "chunk_id": c.chunk_id,
                "start": c.start,
                "end": c.end,
                "text_len": len(c.text),
                "lang_guess": "ko" if ko_cnt > en_cnt else ("en" if en_cnt > ko_cnt else "mixed"),
                "text": c.text,
                "created_at": created,
            })
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---- BM25
        bm25_texts = [clean_for_bm25(t) for t in texts]
        bm25_dict = build_bm25_corpus(ids, bm25_texts)
        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_dict, f)

        # ---- model.json (코사인 설정 명시)
        model_cfg = {
            "model": args.model,
            "normalize": True,
            "faiss": "HNSW",
            "metric": "IP",
            "similarity": "cosine",
            "dim": int(d),
            "hnsw": {"M": 32, "efC": 200, "efS": 64}
        }
        (out_dir / "model.json").write_text(json.dumps(model_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[saved] {out_dir}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
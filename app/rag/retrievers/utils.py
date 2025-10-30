# app/rag/retrievers/utils.py
from __future__ import annotations
from langchain_core.documents import Document

def dict_to_doc(d: dict) -> Document:
    text = d.get("text") or d.get("content") or d.get("chunk") or ""
    meta = dict(d)
    for k in ("text", "content", "chunk"):
        meta.pop(k, None)
    meta.setdefault("source",
                    d.get("src") or d.get("source") or d.get("source_path") or d.get("doc") or d.get("file"))
    meta.setdefault("chunk_id", d.get("chunk_id") or d.get("id"))
    meta.setdefault("doc_id", d.get("id"))
    return Document(page_content=text, metadata=meta)
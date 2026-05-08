from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from .pdf_reader import extract_text_from_pdf


DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "vector_index.faiss"
META_PATH = DATA_DIR / "vector_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


@dataclass(frozen=True)
class RagStats:
    documents: int
    chunks: int


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if len(chunk) < 80:
            continue
        chunks.append(chunk)
    return chunks


def _load_meta() -> list[dict[str, Any]]:
    if not META_PATH.exists():
        return []
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_meta(meta: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _faiss():
    import faiss  # type: ignore

    return faiss


def _embedder():
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner=False)
def _cached_embedder():
    return _embedder()


def build_vector_index() -> None:
    """
    Create a new empty index + metadata store.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss = _faiss()
    index = faiss.IndexFlatL2(EMBED_DIM)
    faiss.write_index(index, str(INDEX_PATH))
    _save_meta([])


def load_vector_index():
    """
    Load index, creating it if needed.
    Returns (index, meta_list).
    """
    if not INDEX_PATH.exists() or not META_PATH.exists():
        build_vector_index()
    faiss = _faiss()
    index = faiss.read_index(str(INDEX_PATH))
    meta = _load_meta()
    return index, meta


@st.cache_resource(show_spinner=False)
def load_vector_index_cached():
    return load_vector_index()


def _reset_index_cache() -> None:
    load_vector_index_cached.clear()


def _chunk_fingerprint(chunk: str) -> str:
    return hashlib.sha1(chunk.encode("utf-8", errors="ignore")).hexdigest()


def add_document_to_index(pdf_path: Path, subject: str | None = None, chapter: str | None = None) -> RagStats:
    """
    Extract text → chunk → embed → add to FAISS → save metadata.
    """
    index, meta = load_vector_index_cached()
    embedder = _cached_embedder()

    text = extract_text_from_pdf(pdf_path, max_chars=300_000)
    chunks = _chunk_text(text, chunk_size=500, overlap=100)
    if not chunks:
        return RagStats(documents=len({m.get("document_name") for m in meta if m.get("document_name")}), chunks=len(meta))

    existing = {
        (m.get("document_name", ""), m.get("chunk_hash", ""))
        for m in meta
        if m.get("document_name") and m.get("chunk_hash")
    }
    new_chunks = [c for c in chunks if (pdf_path.name, _chunk_fingerprint(c)) not in existing]
    if not new_chunks:
        docs = len({m.get("document_name") for m in meta if m.get("document_name")})
        return RagStats(documents=docs, chunks=len(meta))

    embeddings = embedder.encode(new_chunks, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    index.add(embeddings)

    doc_name = pdf_path.name
    for c in new_chunks:
        meta.append(
            {
                "document_name": doc_name,
                "subject": subject or "",
                "chapter": chapter or "",
                "text_chunk": c,
                "chunk_hash": _chunk_fingerprint(c),
            }
        )

    faiss = _faiss()
    faiss.write_index(index, str(INDEX_PATH))
    _save_meta(meta)
    _reset_index_cache()

    docs = len({m.get("document_name") for m in meta if m.get("document_name")})
    return RagStats(documents=docs, chunks=len(meta))


def search_context(query: str, top_k: int = 5) -> tuple[str, RagStats]:
    """
    Similarity search across all indexed chunks.
    Returns (context, stats).
    """
    query = (query or "").strip()
    index, meta = load_vector_index_cached()
    docs = len({m.get("document_name") for m in meta if m.get("document_name")})
    stats = RagStats(documents=docs, chunks=len(meta))
    if not query or not meta or index.ntotal == 0:
        return "", stats

    embedder = _cached_embedder()
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    D, I = index.search(q, min(top_k, len(meta)))
    idxs = [int(i) for i in I[0] if int(i) >= 0]

    chosen = []
    for i in idxs:
        if i >= len(meta):
            continue
        m = meta[i]
        chosen.append(f"[{m.get('document_name','')}] {m.get('text_chunk','')}")

    return "\n\n".join(chosen).strip(), stats


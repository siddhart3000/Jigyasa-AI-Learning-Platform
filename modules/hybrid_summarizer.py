"""Hybrid extractive summarizer for real-time PDF text processing.

This module implements sentence-level extractive summarization using:
- NLTK sentence tokenization
- TF-IDF vectorization
- Cosine similarity graph
- TextRank (PageRank on sentence graph)
- Position and length heuristic scoring
"""

from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_into_chunks(text: str, max_chars: int = 2000) -> List[str]:
    """Split text into sentence-preserving chunks near max_chars size."""
    if not text or not text.strip():
        return []

    stripped_text = text.strip()
    if len(stripped_text) <= max_chars:
        return [stripped_text]

    sentences = [s.strip() for s in sent_tokenize(stripped_text) if s.strip()]
    if not sentences:
        return [stripped_text[i : i + max_chars] for i in range(0, len(stripped_text), max_chars)]

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if sentence_len > max_chars:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_len = 0
            chunks.extend(sentence[i : i + max_chars] for i in range(0, sentence_len, max_chars))
            continue

        additional = sentence_len + (1 if current_sentences else 0)
        if current_len + additional <= max_chars:
            current_sentences.append(sentence)
            current_len += additional
        else:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_len = sentence_len

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def _position_scores(n_sentences: int) -> np.ndarray:
    """Compute linear decay position scores from 1.0 to 0.0."""
    if n_sentences <= 0:
        return np.array([], dtype=np.float64)
    if n_sentences == 1:
        return np.array([1.0], dtype=np.float64)
    return np.linspace(1.0, 0.0, n_sentences, dtype=np.float64)


def _length_scores(sentences: List[str]) -> np.ndarray:
    """Assign sentence length scores: 1.0 if 5-40 words else 0.5."""
    if not sentences:
        return np.array([], dtype=np.float64)

    word_counts = np.fromiter((len(sentence.split()) for sentence in sentences), dtype=np.int32)
    in_range = (word_counts >= 5) & (word_counts <= 40)
    return np.where(in_range, 1.0, 0.5).astype(np.float64)


def _textrank_scores(sentences: List[str]) -> np.ndarray:
    """Compute TextRank scores from TF-IDF cosine similarity graph."""
    n_sentences = len(sentences)
    if n_sentences == 0:
        return np.array([], dtype=np.float64)
    if n_sentences == 1:
        return np.array([1.0], dtype=np.float64)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=True)
    np.fill_diagonal(similarity_matrix, 0.0)

    graph = nx.from_numpy_array(similarity_matrix)
    pagerank_scores = nx.pagerank(graph, max_iter=100, tol=1e-6)

    return np.array([pagerank_scores.get(i, 0.0) for i in range(n_sentences)], dtype=np.float64)


def summarize(text: str, top_ratio: float = 0.2) -> str:
    """Summarize text using hybrid TextRank + heuristics scoring."""
    if not text or not text.strip():
        return ""

    sentences = [s.strip() for s in sent_tokenize(text.strip()) if s.strip()]
    n_sentences = len(sentences)

    if n_sentences < 3:
        return text.strip()

    textrank = _textrank_scores(sentences)
    position = _position_scores(n_sentences)
    length = _length_scores(sentences)

    final_scores = (0.4 * textrank) + (0.3 * position) + (0.3 * length)

    k = max(1, int(np.ceil(n_sentences * top_ratio)))
    top_indices = np.argpartition(-final_scores, k - 1)[:k]
    ordered_indices = np.sort(top_indices)

    return " ".join(sentences[idx] for idx in ordered_indices)


def summarize_pdf(text: str) -> str:
    """Summarize long PDF text by chunking and summarizing each chunk."""
    if not text or not text.strip():
        return ""

    chunks = split_into_chunks(text, max_chars=2000)
    if not chunks:
        return ""

    chunk_summaries = [summarize(chunk, top_ratio=0.2) for chunk in chunks]
    cleaned = [summary.strip() for summary in chunk_summaries if summary and summary.strip()]

    return " ".join(cleaned).strip()


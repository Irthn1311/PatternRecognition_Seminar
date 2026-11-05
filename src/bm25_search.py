"""
Vietnamese BM25 Lexical Retriever (Task G2).
Tokenizes Vietnamese text using PyVi word segmentation (compound words)
and calculates lexical relevance scores using BM25Okapi on corpus passages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi

from src.indexer import DEFAULT_DATA_PATH


def tokenize_vietnamese(text: str) -> List[str]:
    """
    Tokenizes Vietnamese text into compound words and terms using PyVi.
    Lowercases tokens and strips edge punctuation while preserving compound underscores.
    """
    if not text or not text.strip():
        return []
    segmented = ViTokenizer.tokenize(text.strip())
    tokens = []
    for token in segmented.split():
        cleaned = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", token.lower())
        if cleaned:
            tokens.append(cleaned)
    return tokens


class BM25Retriever:
    """BM25 retriever implementation for Vietnamese IT corpus."""

    def __init__(self, data_path: str = DEFAULT_DATA_PATH) -> None:
        self.data_path = Path(data_path)
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def _ensure_indexed(self) -> None:
        """Loads corpus documents and constructs BM25 index on first demand."""
        if self._bm25 is not None:
            return

        if not self.data_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.data_path}")

        print(f"[BM25] Loading corpus from '{self.data_path}' and building BM25 index...")
        self.documents = []
        self.tokenized_corpus = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    doc = json.loads(line_str)
                    self.documents.append(doc)
                    tokens = tokenize_vietnamese(doc["text"])
                    self.tokenized_corpus.append(tokens)

        self._bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[BM25] Index constructed for {len(self.documents)} passages.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes BM25 search.
        If topic is specified, filters valid candidate documents before ranking Top-K.
        """
        if top_k <= 0:
            return []

        self._ensure_indexed()
        assert self._bm25 is not None

        query_str = query.strip()
        if not query_str or not self.documents:
            return []

        query_tokens = tokenize_vietnamese(query_str)
        if not query_tokens:
            return []

        # Get raw BM25 scores for all documents in the corpus
        all_scores = self._bm25.get_scores(query_tokens)

        # Filter candidates by topic if specified
        candidates: List[tuple[int, float]] = []
        for idx, doc in enumerate(self.documents):
            if topic is None or doc.get("topic") == topic:
                candidates.append((idx, float(all_scores[idx])))

        if not candidates:
            return []

        # Sort candidate documents by BM25 score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        effective_k = min(top_k, len(candidates))
        top_candidates = candidates[:effective_k]

        formatted: List[Dict[str, Any]] = []
        for rank, (doc_idx, score) in enumerate(top_candidates, start=1):
            doc = self.documents[doc_idx]
            formatted.append(
                {
                    "rank": rank,
                    "id": doc["id"],
                    "title": doc.get("title", ""),
                    "document": doc["text"],
                    "topic": doc.get("topic", ""),
                    "source_id": doc.get("source_id", ""),
                    "source_url": doc.get("source_url") or "",
                    "score": round(score, 4),
                    "retriever": "bm25",
                }
            )

        return formatted

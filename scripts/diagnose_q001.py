"""
Comprehensive diagnostic and hypothesis test for dev queries (q_001, q_005, q_019).
Tests Variant A (Title-enriched BM25) and Variant B (Title-enriched E5 embedding)
without modifying any production collections or official files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.bm25_search import BM25Retriever, tokenize_vietnamese
from src.chroma_search import ChromaRetriever
from src.indexer import DEFAULT_MODEL_NAME, load_corpus_documents


def test_hypotheses() -> None:
    docs = load_corpus_documents("data/corpus/documents.jsonl")
    dev_queries = [
        ("q_001", "làm thế nào để máy tính tự động tìm ra các vật thể và vị trí của chúng trong một bức ảnh?"),
        ("q_005", "làm sao để ngăn chặn người dùng nhập mã độc vào ô tìm kiếm nhằm phá hoại cơ sở dữ liệu web"),
        ("q_019", "sự khác biệt giữa điều chuẩn L1 Lasso và điều chuẩn L2 Ridge"),
    ]

    print("=" * 80)
    print("HYPOTHESIS TESTING: BASELINE vs TITLE-ENRICHED REPRESENTATION")
    print("=" * 80)

    # 1. Setup Baseline & Title-Enriched BM25
    corpus_text_only = [tokenize_vietnamese(d["text"]) for d in docs]
    bm25_baseline = BM25Okapi(corpus_text_only)

    corpus_title_text = [tokenize_vietnamese(f"{d['title']}. {d['text']}") for d in docs]
    bm25_title_enriched = BM25Okapi(corpus_title_text)

    # 2. Setup Baseline & Title-Enriched E5 Embeddings in memory
    print("\n[E5] Encoding baseline passages (text only)...")
    model = SentenceTransformer(DEFAULT_MODEL_NAME, device="cpu")
    passages_baseline = [f"passage: {d['text'].strip()}" for d in docs]
    emb_baseline = model.encode(passages_baseline, normalize_embeddings=True, show_progress_bar=False)

    print("[E5] Encoding title-enriched passages (title + text)...")
    passages_title = [f"passage: {d['title']}. {d['text'].strip()}" for d in docs]
    emb_title = model.encode(passages_title, normalize_embeddings=True, show_progress_bar=False)

    for qid, qtext in dev_queries:
        print("\n" + "#" * 80)
        print(f"QUERY {qid}: \"{qtext}\"")
        print("#" * 80)

        # A. BM25 Comparison
        q_tokens = tokenize_vietnamese(qtext)
        scores_base = bm25_baseline.get_scores(q_tokens)
        top5_b_base = sorted(range(len(docs)), key=lambda i: scores_base[i], reverse=True)[:5]

        scores_title = bm25_title_enriched.get_scores(q_tokens)
        top5_b_title = sorted(range(len(docs)), key=lambda i: scores_title[i], reverse=True)[:5]

        print("\n[BM25 BASELINE (Text-only)] Top-5:")
        for rank, idx in enumerate(top5_b_base, 1):
            d = docs[idx]
            print(f"  Rank {rank}: [{d['id']}] {d['title'][:45]:<45} (Score: {scores_base[idx]:.4f})")

        print("\n[BM25 VARIANT (Title + Text)] Top-5:")
        for rank, idx in enumerate(top5_b_title, 1):
            d = docs[idx]
            print(f"  Rank {rank}: [{d['id']}] {d['title'][:45]:<45} (Score: {scores_title[idx]:.4f})")

        # B. E5 Comparison
        q_emb = model.encode(f"query: {qtext.strip()}", normalize_embeddings=True, show_progress_bar=False)
        # Cosine distance = 1.0 - dot_product for normalized vectors
        dist_base = 1.0 - np.dot(emb_baseline, q_emb)
        top5_e_base = np.argsort(dist_base)[:5]

        dist_title = 1.0 - np.dot(emb_title, q_emb)
        top5_e_title = np.argsort(dist_title)[:5]

        print("\n[CHROMA E5 BASELINE (Text-only)] Top-5:")
        for rank, idx in enumerate(top5_e_base, 1):
            d = docs[idx]
            print(f"  Rank {rank}: [{d['id']}] {d['title'][:45]:<45} (Dist: {dist_base[idx]:.4f})")

        print("\n[CHROMA E5 VARIANT (Title + Text)] Top-5:")
        for rank, idx in enumerate(top5_e_title, 1):
            d = docs[idx]
            print(f"  Rank {rank}: [{d['id']}] {d['title'][:45]:<45} (Dist: {dist_title[idx]:.4f})")


if __name__ == "__main__":
    test_hypotheses()

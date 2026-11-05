"""
Comprehensive diagnostic and hypothesis test for dev queries (q_001, q_005, q_019).
Specifically diagnoses q_001 by comparing actual Chroma HNSW search, actual BM25 search,
and in-memory exact cosine search variants.
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


def diagnose_actual_q001_retrieval() -> None:
    """Runs actual ChromaRetriever (HNSW vector search) and BM25Retriever on q_001."""
    q_001 = "làm thế nào để máy tính tự động tìm ra các vật thể và vị trí của chúng trong một bức ảnh?"
    target_ids = {
        "cv_009": "Image Classification",
        "cv_010": "Faster R-CNN (Two-stage Detection)",
        "cv_011": "YOLO (One-stage Detection)",
        "cv_014": "Instance Segmentation & Mask R-CNN",
    }

    print("=" * 85)
    print("DIAGNOSIS 1: ACTUAL CHROMA & BM25 RETRIEVAL ON q_001 (Top-30 Inspection)")
    print("=" * 85)
    print(f"Query: \"{q_001}\"\n")

    # 1. Actual Chroma HNSW Retrieval
    chroma_retriever = ChromaRetriever()
    chroma_top30 = chroma_retriever.search(q_001, top_k=30)

    print("[ACTUAL CHROMA HNSW SEARCH] Target Document Ranks:")
    chroma_ranks = {r["id"]: (r["rank"], r["distance"]) for r in chroma_top30}
    for did, desc in target_ids.items():
        if did in chroma_ranks:
            rank, dist = chroma_ranks[did]
            print(f"  -> [{did}] {desc:<38} | Rank: {rank:2d} | Cosine Distance: {dist:.4f}")
        else:
            print(f"  -> [{did}] {desc:<38} | Rank: > 30")

    print("\n[ACTUAL CHROMA HNSW SEARCH] Top-10 Retrieved Documents:")
    for r in chroma_top30[:10]:
        mark = " *** [TARGET]" if r["id"] in target_ids else ""
        print(f"  Rank {r['rank']:2d}: [{r['id']}] {r['title'][:45]:<45} (Dist: {r['distance']:.4f}){mark}")

    # 2. Actual BM25 Retrieval
    bm25_retriever = BM25Retriever()
    bm25_top30 = bm25_retriever.search(q_001, top_k=30)

    print("\n[ACTUAL BM25 RETRIEVAL] Target Document Ranks:")
    bm25_ranks = {r["id"]: (r["rank"], r["score"]) for r in bm25_top30}
    for did, desc in target_ids.items():
        if did in bm25_ranks:
            rank, score = bm25_ranks[did]
            print(f"  -> [{did}] {desc:<38} | Rank: {rank:2d} | BM25 Score: {score:.4f}")
        else:
            print(f"  -> [{did}] {desc:<38} | Rank: > 30")

    print("\n[ACTUAL BM25 RETRIEVAL] Top-10 Retrieved Documents:")
    for r in bm25_top30[:10]:
        mark = " *** [TARGET]" if r["id"] in target_ids else ""
        print(f"  Rank {r['rank']:2d}: [{r['id']}] {r['title'][:45]:<45} (Score: {r['score']:.4f}){mark}")


def test_hypotheses_and_variants() -> None:
    """
    Compares baseline representations against title-enriched representations.
    Labels in-memory exact cosine calculations clearly as 'E5 EXACT COSINE (RAM)',
    distinguishing them from Chroma's HNSW vector index.
    """
    docs = load_corpus_documents("data/corpus/documents.jsonl")
    dev_queries = [
        ("q_001", "làm thế nào để máy tính tự động tìm ra các vật thể và vị trí của chúng trong một bức ảnh?"),
        ("q_005", "làm sao để ngăn chặn người dùng nhập mã độc vào ô tìm kiếm nhằm phá hoại cơ sở dữ liệu web"),
        ("q_019", "sự khác biệt giữa điều chuẩn L1 Lasso và điều chuẩn L2 Ridge"),
    ]

    print("\n" + "=" * 85)
    print("DIAGNOSIS 2: HYPOTHESIS TESTING (EXACT IN-MEMORY COSINE vs TITLE-ENRICHED)")
    print("=" * 85)
    print("Note: The vector distances below are computed as exhaustive exact cosine in RAM (1 - dot),")
    print("serving as a theoretical baseline distinct from Chroma's HNSW graph index.\n")

    # BM25 Baseline & Variant
    corpus_text_only = [tokenize_vietnamese(d["text"]) for d in docs]
    bm25_baseline = BM25Okapi(corpus_text_only)

    corpus_title_text = [tokenize_vietnamese(f"{d['title']}. {d['text']}") for d in docs]
    bm25_title_enriched = BM25Okapi(corpus_title_text)

    # In-Memory E5 Baseline & Variant
    print("[E5] Encoding baseline passages in RAM (passage: text)...")
    model = SentenceTransformer(DEFAULT_MODEL_NAME, device="cpu")
    passages_baseline = [f"passage: {d['text'].strip()}" for d in docs]
    emb_baseline = model.encode(passages_baseline, normalize_embeddings=True, show_progress_bar=False)

    print("[E5] Encoding title-enriched passages in RAM (passage: title. text)...")
    passages_title = [f"passage: {d['title']}. {d['text'].strip()}" for d in docs]
    emb_title = model.encode(passages_title, normalize_embeddings=True, show_progress_bar=False)

    for qid, qtext in dev_queries:
        print("\n" + "#" * 85)
        print(f"QUERY {qid}: \"{qtext}\"")
        print("#" * 85)

        # BM25
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

        # E5 In-Memory Exact Cosine
        q_emb = model.encode(f"query: {qtext.strip()}", normalize_embeddings=True, show_progress_bar=False)
        dist_base = 1.0 - np.dot(emb_baseline, q_emb)
        top5_e_base = np.argsort(dist_base)[:5]

        dist_title = 1.0 - np.dot(emb_title, q_emb)
        top5_e_title = np.argsort(dist_title)[:5]

        print("\n[E5 EXACT COSINE IN RAM (Text-only)] Top-5:")
        for rank, idx in enumerate(top5_e_base, 1):
            d = docs[idx]
            print(f"  Rank {rank}: [{d['id']}] {d['title'][:45]:<45} (Dist: {dist_base[idx]:.4f})")

        print("\n[E5 TITLE-ENRICHED EXACT COSINE IN RAM (Title + Text)] Top-5:")
        for rank, idx in enumerate(top5_e_title, 1):
            d = docs[idx]
            print(f"  Rank {rank}: [{d['id']}] {d['title'][:45]:<45} (Dist: {dist_title[idx]:.4f})")


def main() -> None:
    diagnose_actual_q001_retrieval()
    test_hypotheses_and_variants()


if __name__ == "__main__":
    main()

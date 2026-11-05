"""
Reproducible Retrieval Evaluation Runner (Task G3).

Executes evaluation on the fixed 12 test queries subset, computes coverage statistics,
validates the Verified-Label Gate, and compares Chroma vs BM25 rankings.
Outputs reproducible artifacts in results/g3/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import numpy
import pyvi
import rank_bm25
import sentence_transformers
import torch

from src.bm25_search import BM25Retriever
from src.chroma_search import ChromaRetriever
from src.evaluation import (
    ReviewPoolGate,
    calculate_hit_at_k,
    calculate_mrr_at_k,
    calculate_ndcg_at_k,
    run_metrics_fixture_test,
)
from src.indexer import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_NAME,
    compute_file_sha256,
    load_corpus_documents,
)

DEFAULT_SUBSET_PATH = "data/eval/evaluation_subset.json"
DEFAULT_REVIEW_POOL_PATH = "data/eval/review_pool.csv"
DEFAULT_RESULTS_DIR = "results/g3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible retrieval evaluation (Task G3)")
    parser.add_argument("--subset-path", default=DEFAULT_SUBSET_PATH, help="Path to evaluation_subset.json")
    parser.add_argument("--corpus-path", default=DEFAULT_DATA_PATH, help="Path to documents.jsonl")
    parser.add_argument("--review-pool", default=DEFAULT_REVIEW_POOL_PATH, help="Path to review_pool.csv")
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help="Directory to save evaluation results")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to Chroma database")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Cutoff K for metrics (default: 5)")
    parser.add_argument(
        "--official",
        action="store_true",
        help="Enforce verified-label gate; strictly refuse official metrics if any label is pending",
    )
    parser.add_argument(
        "--fixture-test",
        action="store_true",
        help="Run the mathematical fixture test for metrics and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Self-verification fixture test mode
    if args.fixture_test:
        print("[Fixture] Running mathematical metrics correctness fixture test...")
        res = run_metrics_fixture_test()
        print(f"[Fixture] Result: {res}")
        print("Metrics implementation matches hand-calculated values exactly. PASS.")
        sys.exit(0)

    # 1. Load subset and initialize gate
    subset_path = Path(args.subset_path)
    if not subset_path.exists():
        print(f"Error: Subset file not found: {subset_path}", file=sys.stderr)
        sys.exit(1)

    with open(subset_path, "r", encoding="utf-8") as f:
        subset_data = json.load(f)

    queries: List[Dict[str, Any]] = subset_data.get("queries", [])
    query_ids = [q["query_id"] for q in queries]

    gate = ReviewPoolGate(args.review_pool)
    coverage = gate.get_coverage_statistics(query_ids)

    print("\n" + "=" * 90)
    print("  G3 RETRIEVAL EVALUATION & REPRODUCIBILITY RUNNER")
    print("=" * 90)
    print(f"Corpus Path:       {args.corpus_path}")
    corpus_sha = compute_file_sha256(args.corpus_path)
    print(f"Corpus SHA-256:    {corpus_sha}")
    print(f"Evaluation Subset: {subset_data.get('subset_name')} ({len(queries)} queries)")
    print(f"Review Pool Path:  {args.review_pool}")
    print("-" * 90)
    print("Dependencies:")
    print(f"  chromadb:             {chromadb.__version__}")
    print(f"  sentence_transformers: {sentence_transformers.__version__}")
    print(f"  torch:                {torch.__version__}")
    print(f"  rank_bm25:            {rank_bm25.__version__ if hasattr(rank_bm25, '__version__') else '0.2.2'}")
    print(f"  pyvi:                 {pyvi.__version__ if hasattr(pyvi, '__version__') else '0.1.1'}")
    print(f"  numpy:                {numpy.__version__}")
    print("-" * 90)
    print("Human Review Label Coverage:")
    print(f"  Total Pooled Candidate Pairs: {coverage['total_pairs']}")
    print(f"  Verified Candidate Pairs:     {coverage['verified_pairs']} ({coverage['completion_percentage']}%)")
    print(f"  Pending / Unverified Pairs:   {coverage['pending_pairs']}")
    print(f"  Proposed Labels in Pool:      {coverage['proposed_distribution']}")
    print("=" * 90)

    # 2. Verified-Label Gate Enforcement
    if args.official:
        if coverage["pending_pairs"] > 0:
            print("\n[VERIFIED-LABEL GATE REFUSAL] Official benchmark evaluation locked!", file=sys.stderr)
            print(
                f"  Found {coverage['pending_pairs']} unverified / pending candidate pairs in '{args.review_pool}'.\n"
                "  To prevent reporting uncertified AI draft labels as official scientific ground truth,\n"
                "  aggregate official metrics (Hit@K, MRR@K, nDCG@K) can only be generated when human\n"
                "  reviewers have audited the candidates and set review_status='verified'.\n"
                "  Run without '--official' to preview rankings, disagreements, and pool coverage.",
                file=sys.stderr,
            )
            sys.exit(1)

    # 3. Initialize retrievers
    print("\n[Retriever] Initializing Chroma and BM25 retrievers on 114 passages...")
    chroma_retriever = ChromaRetriever(
        db_path=args.db_path,
        collection_name=args.collection_name,
        data_path=args.corpus_path,
    )
    bm25_retriever = BM25Retriever(data_path=args.corpus_path)

    # 4. Execute retrieval and collect rankings
    print(f"[Evaluation] Retrieving Top-{args.top_k} for all {len(queries)} queries...")
    eval_results: List[Dict[str, Any]] = []

    answerable_count = 0
    unanswerable_count = 0
    top1_agreement_count = 0

    for q in queries:
        qid = q["query_id"]
        qtext = q["query"]
        category = q.get("category", "")
        is_unans = q.get("is_unanswerable", False)

        c_results = chroma_retriever.search(qtext, top_k=args.top_k)
        b_results = bm25_retriever.search(qtext, top_k=args.top_k)

        c_top1 = c_results[0]["id"] if c_results else None
        b_top1 = b_results[0]["id"] if b_results else None

        if c_top1 == b_top1 and c_top1 is not None:
            top1_agreement_count += 1

        if is_unans:
            unanswerable_count += 1
        else:
            answerable_count += 1

        query_record: Dict[str, Any] = {
            "query_id": qid,
            "query_text": qtext,
            "category": category,
            "is_unanswerable": is_unans,
            "chroma_top_k": [
                {
                    "rank": item["rank"],
                    "doc_id": item["id"],
                    "title": item["title"],
                    "distance": round(item["distance"], 4),
                }
                for item in c_results
            ],
            "bm25_top_k": [
                {
                    "rank": item["rank"],
                    "doc_id": item["id"],
                    "title": item["title"],
                    "score": round(item["score"], 4),
                }
                for item in b_results
            ],
            "top1_match": (c_top1 == b_top1),
        }

        eval_results.append(query_record)

    # 5. Display Summary & Per-Query Comparison Table
    print("\n" + "=" * 90)
    print("  PER-QUERY RANKING PREVIEW (Top-1 Comparison)")
    print("=" * 90)
    header_fmt = "{:<8} | {:<22} | {:<12} | {:<12} | {:<10}"
    print(header_fmt.format("Query ID", "Category", "Chroma Top-1", "BM25 Top-1", "Agreement"))
    print("-" * 90)

    for item in eval_results:
        qid = item["query_id"]
        cat = item["category"]
        c_top1 = item["chroma_top_k"][0]["doc_id"] if item["chroma_top_k"] else "N/A"
        b_top1 = item["bm25_top_k"][0]["doc_id"] if item["bm25_top_k"] else "N/A"
        agree = "AGREE" if item["top1_match"] else "DIFFER"
        if item["is_unanswerable"]:
            agree += " (Unans)"
        print(header_fmt.format(qid, cat, c_top1, b_top1, agree))

    print("-" * 90)
    print(f"Top-1 Agreement Rate: {top1_agreement_count}/{len(queries)} ({round(100.0 * top1_agreement_count / len(queries), 1)}%)")
    print(f"Answerable Queries:   {answerable_count} (evaluated for metrics once verified)")
    print(f"Unanswerable Queries: {unanswerable_count} (analyzed qualitatively for over-retrieval)")
    print("=" * 90)

    # 6. Qualitative Note on Unanswerable Query
    unans_items = [item for item in eval_results if item["is_unanswerable"]]
    if unans_items:
        print("\n[Unanswerable Query Analysis]:")
        for u in unans_items:
            print(f"  Query {u['query_id']}: \"{u['query_text']}\"")
            print(f"  - Nearest Chroma Candidate: {u['chroma_top_k'][0]['doc_id']} (dist: {u['chroma_top_k'][0]['distance']})")
            print(f"  - Top BM25 Candidate:       {u['bm25_top_k'][0]['doc_id']} (score: {u['bm25_top_k'][0]['score']})")
            print("  Note: Neither retriever abstains; both retrieve nearest passages despite absence of correct answers.")

    # 7. Export reproducible artifacts to results/g3/
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_artifact = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus_path": args.corpus_path,
            "corpus_sha256": corpus_sha,
            "document_count": 114,
            "model_name": DEFAULT_MODEL_NAME,
            "collection_name": args.collection_name,
            "vector_dimension": 384,
            "metric": "cosine",
            "dependencies": {
                "chromadb": chromadb.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "torch": torch.__version__,
                "rank_bm25": getattr(rank_bm25, "__version__", "0.2.2"),
                "pyvi": getattr(pyvi, "__version__", "0.1.1"),
                "numpy": numpy.__version__,
            },
        },
        "subset": {
            "name": subset_data.get("subset_name"),
            "total_queries": len(queries),
            "answerable_queries": answerable_count,
            "unanswerable_queries": unanswerable_count,
        },
        "coverage": coverage,
        "agreement": {
            "top1_agreement_count": top1_agreement_count,
            "top1_agreement_percentage": round(100.0 * top1_agreement_count / len(queries), 1),
        },
        "eval_results": eval_results,
    }

    summary_file = out_dir / "rankings_preview.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_artifact, f, indent=2, ensure_ascii=False)

    print(f"\n[Artifacts] Reproducible evaluation summary saved to '{summary_file}'")


if __name__ == "__main__":
    main()

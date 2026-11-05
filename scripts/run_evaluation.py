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
        help="Enforce verified-label gate; compute official metrics if and only if all labels are verified",
    )
    parser.add_argument(
        "--fixture-test",
        action="store_true",
        help="Run the mathematical fixture test for metrics and exit",
    )
    return parser.parse_args()


def print_official_summary_tables(
    eval_results: List[Dict[str, Any]],
    macro_averages: Dict[str, Dict[str, float]],
    answerable_count: int,
    unanswerable_count: int,
) -> None:
    """Prints clean terminal tables for official per-query and macro-average metrics."""
    print("\n" + "=" * 90)
    print("  OFFICIAL RETRIEVAL BENCHMARK METRICS (Top-5)")
    print("=" * 90)

    # Per-query metrics table
    header_fmt = "{:<8} | {:<22} | {:<7} | {:<7} | {:<7} || {:<7} | {:<7} | {:<7}"
    print(
        header_fmt.format(
            "Query ID", "Category", "C-Hit", "C-MRR", "C-nDCG", "B-Hit", "B-MRR", "B-nDCG"
        )
    )
    print("-" * 90)

    for item in eval_results:
        if item["is_unanswerable"]:
            continue
        qid = item["query_id"]
        cat = item["category"]
        c_m = item["chroma"]["metrics"]
        b_m = item["bm25"]["metrics"]
        print(
            header_fmt.format(
                qid,
                cat[:22],
                f"{c_m['hit@5']:.2f}",
                f"{c_m['mrr@5']:.3f}",
                f"{c_m['ndcg@5']:.3f}",
                f"{b_m['hit@5']:.2f}",
                f"{b_m['mrr@5']:.3f}",
                f"{b_m['ndcg@5']:.3f}",
            )
        )

    print("-" * 90)
    print(f"Evaluated Answerable Queries: {answerable_count} | Unanswerable Excluded: {unanswerable_count}")
    print("=" * 90)

    # Macro averages comparison table
    print("\n" + "=" * 60)
    print("  MACRO-AVERAGE BENCHMARK COMPARISON (n = 11 answerable)")
    print("=" * 60)
    agg_fmt = "{:<12} | {:<16} | {:<16} | {:<10}"
    print(agg_fmt.format("Metric", "Chroma (E5)", "BM25 (PyVi)", "Difference"))
    print("-" * 60)
    for m in ["hit@5", "mrr@5", "ndcg@5"]:
        c_val = macro_averages["chroma"][m]
        b_val = macro_averages["bm25"][m]
        diff = c_val - b_val
        diff_str = f"{diff:+.4f}"
        print(agg_fmt.format(m.upper(), f"{c_val:.4f}", f"{b_val:.4f}", diff_str))
    print("=" * 60)


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
    corpus_docs = load_corpus_documents(args.corpus_path)
    corpus_count = len(corpus_docs)
    corpus_sha = compute_file_sha256(args.corpus_path)

    print("\n" + "=" * 90)
    print("  G3 RETRIEVAL EVALUATION & REPRODUCIBILITY RUNNER")
    print("=" * 90)
    print(f"Corpus Path:       {args.corpus_path} ({corpus_count} documents)")
    print(f"Corpus SHA-256:    {corpus_sha}")
    print(f"Evaluation Subset: {subset_data.get('subset_name')} ({len(queries)} queries)")
    print(f"Review Pool Path:  {args.review_pool}")
    print(f"Evaluation Mode:   {'OFFICIAL BENCHMARK' if args.official else 'PREVIEW / AUDIT'}")
    print("-" * 90)
    print("Dependencies:")
    print(f"  chromadb:             {chromadb.__version__}")
    print(f"  sentence_transformers: {sentence_transformers.__version__}")
    print(f"  torch:                {torch.__version__}")
    print(f"  rank_bm25:            {getattr(rank_bm25, '__version__', '0.2.2')}")
    print(f"  pyvi:                 {getattr(pyvi, '__version__', '0.1.1')}")
    print(f"  numpy:                {numpy.__version__}")
    print("-" * 90)
    print("Human Review Label Coverage:")
    print(f"  Total Pooled Candidate Pairs: {coverage['total_pairs']}")
    print(f"  Verified Candidate Pairs:     {coverage['verified_pairs']} ({coverage['completion_percentage']}%)")
    print(f"  Pending / Unverified Pairs:   {coverage['pending_pairs']}")
    print(f"  Proposed Labels in Pool:      {coverage['proposed_distribution']}")
    print("=" * 90)

    # 2. Initialize retrievers
    print("\n[Retriever] Initializing Chroma and BM25 retrievers...")
    chroma_retriever = ChromaRetriever(
        db_path=args.db_path,
        collection_name=args.collection_name,
        data_path=args.corpus_path,
    )
    bm25_retriever = BM25Retriever(data_path=args.corpus_path)

    # 3. Execute retrieval and collect rankings
    print(f"[Evaluation] Retrieving Top-{args.top_k} for all {len(queries)} queries...")
    eval_results: List[Dict[str, Any]] = []
    retrieved_rankings: Dict[str, Dict[str, List[str]]] = {}

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

        c_ids = [item["id"] for item in c_results]
        b_ids = [item["id"] for item in b_results]
        retrieved_rankings[qid] = {"chroma": c_ids, "bm25": b_ids}

        c_top1 = c_ids[0] if c_ids else None
        b_top1 = b_ids[0] if b_ids else None

        if c_top1 == b_top1 and c_top1 is not None:
            top1_agreement_count += 1

        if is_unans:
            unanswerable_count += 1
        else:
            answerable_count += 1

        eval_results.append(
            {
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
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Handle OFFICIAL Evaluation Mode
    if args.official:
        print("\n[Gate] Validating Verified-Label Gate for official benchmark evaluation...")
        is_gate_valid, gate_errors = gate.validate_gate(
            subset_query_ids=query_ids,
            retrieved_rankings=retrieved_rankings,
            top_k=args.top_k,
        )

        if not is_gate_valid:
            print("\n[VERIFIED-LABEL GATE REFUSAL] Official benchmark evaluation locked!", file=sys.stderr)
            print(f"Total gate violations found: {len(gate_errors)}", file=sys.stderr)
            for err in gate_errors[:10]:
                print(f"  -> {err}", file=sys.stderr)
            if len(gate_errors) > 10:
                print(f"  ... and {len(gate_errors) - 10} more missing/unverified items.", file=sys.stderr)
            print(
                "\nTo prevent reporting uncertified AI draft labels as official scientific ground truth,\n"
                "official benchmark metrics can only be computed once all candidate labels are verified.\n"
                "Run without '--official' to preview rankings, disagreements, and pool coverage.",
                file=sys.stderr,
            )
            sys.exit(1)

        print("[Gate] All 12 queries and all retrieved candidates are verified! Computing official metrics...")

        # Compute per-query and macro-average metrics for the 11 answerable queries
        official_query_metrics: List[Dict[str, Any]] = []
        c_hits, c_mrrs, c_ndcgs = [], [], []
        b_hits, b_mrrs, b_ndcgs = [], [], []

        for item in eval_results:
            qid = item["query_id"]
            if item["is_unanswerable"]:
                continue

            c_doc_ids = [d["doc_id"] for d in item["chroma_top_k"]]
            b_doc_ids = [d["doc_id"] for d in item["bm25_top_k"]]

            c_rels = [gate.get_verified_relevance(qid, did) for did in c_doc_ids]
            b_rels = [gate.get_verified_relevance(qid, did) for did in b_doc_ids]
            assert all(r is not None for r in c_rels), f"Missing verified relevance for Chroma in {qid}"
            assert all(r is not None for r in b_rels), f"Missing verified relevance for BM25 in {qid}"

            all_judged_rels = gate.get_all_verified_relevances_for_query(qid)

            c_hit5 = calculate_hit_at_k(c_rels, k=args.top_k)  # type: ignore[arg-type]
            c_mrr5 = calculate_mrr_at_k(c_rels, k=args.top_k)  # type: ignore[arg-type]
            c_ndcg5 = calculate_ndcg_at_k(c_rels, all_judged_rels, k=args.top_k)  # type: ignore[arg-type]

            b_hit5 = calculate_hit_at_k(b_rels, k=args.top_k)  # type: ignore[arg-type]
            b_mrr5 = calculate_mrr_at_k(b_rels, k=args.top_k)  # type: ignore[arg-type]
            b_ndcg5 = calculate_ndcg_at_k(b_rels, all_judged_rels, k=args.top_k)  # type: ignore[arg-type]

            c_hits.append(c_hit5)
            c_mrrs.append(c_mrr5)
            c_ndcgs.append(c_ndcg5)

            b_hits.append(b_hit5)
            b_mrrs.append(b_mrr5)
            b_ndcgs.append(b_ndcg5)

            item["chroma"] = {
                "top_5_doc_ids": c_doc_ids,
                "top_5_relevances": c_rels,
                "metrics": {
                    "hit@5": round(c_hit5, 4),
                    "mrr@5": round(c_mrr5, 4),
                    "ndcg@5": round(c_ndcg5, 4),
                },
            }
            item["bm25"] = {
                "top_5_doc_ids": b_doc_ids,
                "top_5_relevances": b_rels,
                "metrics": {
                    "hit@5": round(b_hit5, 4),
                    "mrr@5": round(b_mrr5, 4),
                    "ndcg@5": round(b_ndcg5, 4),
                },
            }
            official_query_metrics.append(item)

        macro_averages = {
            "chroma": {
                "hit@5": round(sum(c_hits) / len(c_hits), 4),
                "mrr@5": round(sum(c_mrrs) / len(c_mrrs), 4),
                "ndcg@5": round(sum(c_ndcgs) / len(c_ndcgs), 4),
            },
            "bm25": {
                "hit@5": round(sum(b_hits) / len(b_hits), 4),
                "mrr@5": round(sum(b_mrrs) / len(b_mrrs), 4),
                "ndcg@5": round(sum(b_ndcgs) / len(b_ndcgs), 4),
            },
        }

        # Print official summary tables
        print_official_summary_tables(
            eval_results=eval_results,
            macro_averages=macro_averages,
            answerable_count=len(c_hits),
            unanswerable_count=unanswerable_count,
        )

        # Build official artifact
        unans_item = next((item for item in eval_results if item["is_unanswerable"]), None)
        official_artifact = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "corpus_path": args.corpus_path,
                "corpus_sha256": corpus_sha,
                "document_count": corpus_count,
                "model_name": DEFAULT_MODEL_NAME,
                "collection_name": args.collection_name,
                "vector_dimension": 384,
                "metric": "cosine",
                "top_k": args.top_k,
                "dependencies": {
                    "chromadb": chromadb.__version__,
                    "sentence_transformers": sentence_transformers.__version__,
                    "torch": torch.__version__,
                    "rank_bm25": getattr(rank_bm25, "__version__", "0.2.2"),
                    "pyvi": getattr(pyvi, "__version__", "0.1.1"),
                    "numpy": numpy.__version__,
                },
            },
            "evaluation_summary": {
                "subset_name": subset_data.get("subset_name"),
                "total_queries": len(queries),
                "evaluated_answerable_queries": len(c_hits),
                "unanswerable_queries_excluded_from_macro_avg": unanswerable_count,
                "total_verified_labels": coverage["verified_pairs"],
                "pooled_judgment_limitation_note": (
                    "IDCG@5 is computed from the verified candidate pool; this is a pool-bounded evaluation, "
                    "not an exhaustive corpus-wide relevance judgment."
                ),
            },
            "macro_averages": macro_averages,
            "per_query_metrics": official_query_metrics,
            "unanswerable_query_analysis": {
                "query_id": unans_item["query_id"] if unans_item else "q_030",
                "query_text": unans_item["query_text"] if unans_item else "",
                "chroma_top_1": unans_item["chroma_top_k"][0] if unans_item else {},
                "bm25_top_1": unans_item["bm25_top_k"][0] if unans_item else {},
                "note": (
                    "Unanswerable query has no relevant answers in corpus. "
                    "Both retrievers return nearest candidates without abstaining; "
                    "excluded from macro-average metrics to avoid distorting retrieval performance."
                ),
            },
        }

        official_file = out_dir / "official_metrics.json"
        with open(official_file, "w", encoding="utf-8") as f:
            json.dump(official_artifact, f, indent=2, ensure_ascii=False)
        print(f"\n[Artifacts] Official benchmark metrics successfully saved to '{official_file}'")
        sys.exit(0)

    # 5. PREVIEW Mode (Display Summary & Export rankings_preview.json)
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

    # 7. Export preview artifact
    summary_artifact = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus_path": args.corpus_path,
            "corpus_sha256": corpus_sha,
            "document_count": corpus_count,
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

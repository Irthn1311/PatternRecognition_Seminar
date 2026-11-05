"""
Candidate Pooling Script for Retrieval Evaluation & Relevance Review (Task G3).

Constructs the evaluation candidate pool for the 12 fixed test queries by pooling:
  1. Chroma Top-5 retrieved candidates
  2. BM25 Top-5 retrieved candidates
  3. Pre-existing draft qrels candidates
Exports data/eval/review_pool.csv (UTF-8-SIG) with review_status='pending' and
empty verified_relevance for human inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_search import BM25Retriever
from src.chroma_search import ChromaRetriever
from src.indexer import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_PATH,
    DEFAULT_DB_PATH,
    load_corpus_documents,
)

DEFAULT_SUBSET_PATH = "data/eval/evaluation_subset.json"
DEFAULT_QRELS_PATH = "data/eval/qrels_draft.jsonl"
DEFAULT_OUTPUT_PATH = "data/eval/review_pool.csv"

CSV_FIELDNAMES = [
    "query_id",
    "query_text",
    "category",
    "split",
    "is_unanswerable",
    "doc_id",
    "doc_title",
    "doc_text",
    "chroma_rank",
    "bm25_rank",
    "proposed_relevance",
    "proposed_note",
    "verified_relevance",
    "review_status",
]


def load_subset(subset_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(subset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation subset file not found: {subset_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", [])


def load_draft_qrels(qrels_path: str | Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Loads draft qrels mapped by query_id -> doc_id -> record."""
    path = Path(qrels_path)
    qrels: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not path.exists():
        return qrels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                r = json.loads(line_str)
                qid = r["query_id"]
                did = r["doc_id"]
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = r
    return qrels


def check_existing_verified_labels(output_path: Path) -> int:
    """Checks if existing CSV contains already verified labels."""
    if not output_path.exists():
        return 0
    verified_count = 0
    try:
        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("review_status", "").strip().lower()
                rel = row.get("verified_relevance", "").strip()
                if status == "verified" or rel != "":
                    verified_count += 1
    except Exception:
        pass
    return verified_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate review pool for G3 evaluation")
    parser.add_argument("--subset-path", default=DEFAULT_SUBSET_PATH, help="Path to evaluation_subset.json")
    parser.add_argument("--corpus-path", default=DEFAULT_DATA_PATH, help="Path to documents.jsonl")
    parser.add_argument("--qrels-path", default=DEFAULT_QRELS_PATH, help="Path to qrels_draft.jsonl")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Output path for review_pool.csv")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to Chroma database")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K candidates per retriever to pool (default: 5)")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite even if verified labels exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)

    # Overwrite protection check
    if output_path.exists() and not args.overwrite:
        verified_count = check_existing_verified_labels(output_path)
        if verified_count > 0:
            print(
                f"[Refusal] Target '{args.output_path}' contains {verified_count} verified/annotated label(s).\n"
                "Refusing to overwrite without explicit '--overwrite' flag to protect review work.",
                file=sys.stderr,
            )
            sys.exit(1)

    # 1. Load subset queries and corpus documents
    queries = load_subset(args.subset_path)
    corpus_docs = {d["id"]: d for d in load_corpus_documents(args.corpus_path)}
    draft_qrels = load_draft_qrels(args.qrels_path)

    print(f"[Pool] Loaded {len(queries)} evaluation queries from '{args.subset_path}'")
    print(f"[Pool] Loaded {len(corpus_docs)} corpus documents from '{args.corpus_path}'")

    # 2. Initialize retrievers
    chroma_retriever = ChromaRetriever(
        db_path=args.db_path,
        collection_name=args.collection_name,
        data_path=args.corpus_path,
    )
    bm25_retriever = BM25Retriever(data_path=args.corpus_path)

    # 3. Build candidate pool for each query
    pool_rows: List[Dict[str, Any]] = []
    total_unique_pairs = 0

    for q in queries:
        qid = q["query_id"]
        qtext = q["query"]
        category = q.get("category", "")
        split = q.get("split", "")
        is_unanswerable = q.get("is_unanswerable", False)

        # Retrieve Top-K from Chroma
        chroma_res = chroma_retriever.search(qtext, top_k=args.top_k)
        chroma_ranks = {item["id"]: item["rank"] for item in chroma_res}

        # Retrieve Top-K from BM25
        bm25_res = bm25_retriever.search(qtext, top_k=args.top_k)
        bm25_ranks = {item["id"]: item["rank"] for item in bm25_res}

        # Draft qrels for this query
        query_drafts = draft_qrels.get(qid, {})

        # Union of all candidate doc IDs
        candidate_ids: Set[str] = set(chroma_ranks.keys()) | set(bm25_ranks.keys()) | set(query_drafts.keys())

        # Sort candidate IDs deterministically:
        # Priority: lowest Chroma rank, then lowest BM25 rank, then doc_id alphabetically
        def sort_key(doc_id: str) -> tuple[int, int, str]:
            c_rank = chroma_ranks.get(doc_id, 999)
            b_rank = bm25_ranks.get(doc_id, 999)
            return (c_rank, b_rank, doc_id)

        sorted_candidate_ids = sorted(candidate_ids, key=sort_key)

        for did in sorted_candidate_ids:
            doc_info = corpus_docs.get(did, {})
            draft_info = query_drafts.get(did, {})

            c_rank_str = str(chroma_ranks[did]) if did in chroma_ranks else ""
            b_rank_str = str(bm25_ranks[did]) if did in bm25_ranks else ""

            prop_rel = str(draft_info["relevance"]) if "relevance" in draft_info else ""
            prop_note = draft_info.get("note", "")

            pool_rows.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "category": category,
                    "split": split,
                    "is_unanswerable": "true" if is_unanswerable else "false",
                    "doc_id": did,
                    "doc_title": doc_info.get("title", ""),
                    "doc_text": doc_info.get("text", ""),
                    "chroma_rank": c_rank_str,
                    "bm25_rank": b_rank_str,
                    "proposed_relevance": prop_rel,
                    "proposed_note": prop_note,
                    "verified_relevance": "",  # Left blank for human review
                    "review_status": "pending",  # Strictly pending, never auto-verified
                }
            )

        print(f"  - Query {qid} ({category}): {len(sorted_candidate_ids)} candidates pooled "
              f"(Chroma Top-{len(chroma_ranks)}, BM25 Top-{len(bm25_ranks)}, Draft {len(query_drafts)})")
        total_unique_pairs += len(sorted_candidate_ids)

    # 4. Write pool to review_pool.csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(pool_rows)

    print(f"\n[Pool] Successfully exported {len(pool_rows)} candidate pairs to '{args.output_path}'")
    print(f"[Pool] All rows initialized with review_status='pending' and empty verified_relevance.")


if __name__ == "__main__":
    main()

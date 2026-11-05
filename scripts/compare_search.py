"""
CLI tool to compare Chroma Semantic Search vs Vietnamese BM25 Retrieval (Task G2).

Usage examples:
    python scripts/compare_search.py --query "chỉ mục B-Tree và tối ưu truy vấn"
    python scripts/compare_search.py --query-id q_001 --mode both
    python scripts/compare_search.py --query-id q_003 --top-k 3 --topic "Database"
    python scripts/compare_search.py --query "mạng nơ-ron CNN" --mode chroma
    python scripts/compare_search.py --query "thuật toán Dijkstra" --mode bm25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    DEFAULT_MODEL_NAME,
)

DEFAULT_QUERIES_PATH = "data/eval/queries.jsonl"


def load_query_by_id(query_id: str, queries_path: str = DEFAULT_QUERIES_PATH) -> Dict[str, Any]:
    """Loads a specific query from queries.jsonl by query_id."""
    path = Path(queries_path)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                q = json.loads(line_str)
                if q.get("query_id") == query_id:
                    return q

    raise ValueError(f"query_id '{query_id}' not found in {queries_path}")


def print_results_block(title: str, results: List[Dict[str, Any]], metric_name: str) -> None:
    """Prints retrieved results with rank, ID, title, score/distance, and document preview."""
    print("\n" + "=" * 90)
    print(f"  {title.upper()} (Top-{len(results)})")
    print("=" * 90)

    if not results:
        print("  [No matching documents found]")
        return

    for item in results:
        rank = item["rank"]
        doc_id = item["id"]
        doc_title = item.get("title", "")
        topic = item.get("topic", "")
        source_url = item.get("source_url") or "N/A"
        doc_text = item.get("document", "")

        if "distance" in item:
            score_str = f"Cosine Distance: {item['distance']:.4f} (smaller = closer)"
        else:
            score_str = f"BM25 Score: {item.get('score', 0.0):.4f} (higher = more relevant)"

        print(f"\n[Rank {rank}] {doc_id} - \"{doc_title}\"")
        print(f"  Topic: {topic} | {score_str}")
        print(f"  Source: {source_url}")
        # Show first 220 chars of document as preview
        preview = doc_text if len(doc_text) <= 240 else doc_text[:237] + "..."
        print(f"  Passage: {preview}")
        print("-" * 90)


def print_comparison_disclaimer() -> None:
    print("\n" + "~" * 90)
    print("  METRIC NOTE: Chroma uses Cosine Distance [0..2] where 0 is closest/most similar.")
    print("  BM25 uses an unbounded term-frequency/IDF score where higher values mean greater match.")
    print("  These scores are on fundamentally different scales and CANNOT be directly compared.")
    print("~" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Chroma Semantic Search vs BM25 Lexical Retrieval")
    parser.add_argument("--query", type=str, help="Free-form query text")
    parser.add_argument("--query-id", type=str, help="Query ID from data/eval/queries.jsonl (e.g. q_001)")
    parser.add_argument(
        "--mode",
        choices=["chroma", "bm25", "both"],
        default="both",
        help="Retrieval engine to run (default: both)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest results to retrieve (default: 5)")
    parser.add_argument("--topic", type=str, default=None, help="Optional topic filter (e.g. 'Database')")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to local Chroma DB")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to corpus JSONL")
    parser.add_argument("--queries-path", default=DEFAULT_QUERIES_PATH, help="Path to queries JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine query text and metadata
    query_text = ""
    query_meta: Dict[str, Any] = {}

    if args.query_id:
        try:
            query_meta = load_query_by_id(args.query_id, args.queries_path)
            query_text = query_meta.get("query", "")
            print(f"[Query ID] {args.query_id} (Category: {query_meta.get('category')}, Split: {query_meta.get('split')})")
        except Exception as e:
            print(f"Error loading query_id '{args.query_id}': {e}", file=sys.stderr)
            sys.exit(1)
    elif args.query:
        query_text = args.query.strip()
    else:
        print("Error: You must provide either --query \"<text>\" or --query-id <id>.", file=sys.stderr)
        sys.exit(1)

    if not query_text:
        print("Error: Query text is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"\n==========================================================================================")
    print(f"QUERY: \"{query_text}\"")
    if args.topic:
        print(f"TOPIC FILTER: \"{args.topic}\"")
    print(f"TOP-K: {args.top_k} | MODE: {args.mode.upper()}")
    print(f"==========================================================================================")

    # Chroma retrieval
    if args.mode in ("chroma", "both"):
        try:
            chroma_retriever = ChromaRetriever(
                db_path=args.db_path,
                collection_name=args.collection_name,
            )
            chroma_results = chroma_retriever.search(
                query=query_text,
                top_k=args.top_k,
                topic=args.topic,
            )
            print_results_block("Chroma Semantic Search (Multilingual-E5-Small)", chroma_results, "Cosine Distance")
        except Exception as e:
            print(f"\n[Chroma Error]: {e}", file=sys.stderr)
            print("Hint: If the index has not been built yet, run 'python scripts/build_index.py' first.", file=sys.stderr)

    # BM25 retrieval
    if args.mode in ("bm25", "both"):
        try:
            bm25_retriever = BM25Retriever(data_path=args.data_path)
            bm25_results = bm25_retriever.search(
                query=query_text,
                top_k=args.top_k,
                topic=args.topic,
            )
            print_results_block("Vietnamese BM25 Baseline (PyVi Tokenizer + BM25Okapi)", bm25_results, "BM25 Score")
        except Exception as e:
            print(f"\n[BM25 Error]: {e}", file=sys.stderr)

    if args.mode == "both":
        print_comparison_disclaimer()


if __name__ == "__main__":
    main()

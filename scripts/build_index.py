"""
CLI script to build and verify Chroma index on Vietnamese IT corpus (Task G2).

Usage:
    python scripts/build_index.py                # Build or verify index
    python scripts/build_index.py --check        # Check index consistency only
    python scripts/build_index.py --rebuild      # Force rebuild collection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexer import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_NAME,
    CorpusIndexer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and verify Chroma index for Vietnamese IT Corpus")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to local Chroma DB directory")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="Collection name")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to corpus JSONL")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="SentenceTransformers embedding model")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding encoding batch size")
    parser.add_argument("--check", action="store_true", help="Check existing index consistency and exit")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild collection from scratch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    indexer = CorpusIndexer(
        db_path=args.db_path,
        collection_name=args.collection_name,
        model_name=args.model_name,
        data_path=args.data_path,
    )

    if args.check:
        print(f"[Check] Verifying index at '{args.db_path}' (Collection: '{args.collection_name}')...")
        is_consistent, msg, manifest = indexer.check_index_consistency()
        if is_consistent:
            print(f"[Check] SUCCESS: {msg}")
            if manifest:
                print(f"  - Model: {manifest.get('model_name')}")
                print(f"  - Document Count: {manifest.get('document_count')}")
                print(f"  - Vector Dim: {manifest.get('vector_dim')}")
                print(f"  - Dataset SHA256: {manifest.get('dataset_sha256')}")
                print(f"  - Built At: {manifest.get('built_at')}")
            sys.exit(0)
        else:
            print(f"[Check] INCONSISTENT / MISSING: {msg}")
            sys.exit(1)

    # Build or verify
    print(f"[Main] Managing Chroma index for '{args.collection_name}' at '{args.db_path}'...")
    try:
        count = indexer.build_index(force_rebuild=args.rebuild, batch_size=args.batch_size)
        print(f"\n[Main] Index is ready. Total active records: {count}")
    except RuntimeError as e:
        print(f"\n[Build Error]: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

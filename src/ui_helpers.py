"""
UI Helpers and Mutable Collection Manager for Streamlit Demo (Tasks G4 & G5).

Provides cached resources (Embedding model, BM25 retriever, Chroma client),
manages isolated CRUD operations on 'seminar_demo_mutable', and checks
eligibility for displaying official benchmark metrics vs preview rankings.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

from src.bm25_search import BM25Retriever
from src.chroma_search import ChromaRetriever
from src.indexer import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_NAME,
    compute_file_sha256,
)

DEFAULT_MUTABLE_COLLECTION = "seminar_demo_mutable"
DEFAULT_PREVIEW_JSON_PATH = "results/g3/rankings_preview.json"
DEFAULT_OFFICIAL_JSON_PATH = "results/g3/official_metrics.json"
DEFAULT_REVIEW_POOL_PATH = "data/eval/review_pool.csv"

STANDARD_TOPICS = [
    "All Topics",
    "Machine Learning",
    "Deep Learning",
    "Computer Vision",
    "Database",
    "Algorithms",
    "Computer Networks / Security",
]


@st.cache_resource(show_spinner=False)
def get_cached_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Loads and caches SentenceTransformer model on CPU once across app reruns."""
    return SentenceTransformer(model_name, device="cpu")


@st.cache_resource(show_spinner=False)
def get_cached_bm25_retriever(data_path: str = DEFAULT_DATA_PATH) -> BM25Retriever:
    """Instantiates and caches BM25Retriever on corpus passages once across app reruns."""
    retriever = BM25Retriever(data_path=data_path)
    retriever._ensure_indexed()
    return retriever


@st.cache_resource(show_spinner=False)
def get_cached_chroma_client(db_path: str = DEFAULT_DB_PATH) -> chromadb.PersistentClient:
    """Creates and caches Chroma PersistentClient instance."""
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def get_cached_chroma_retriever(
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    data_path: str = DEFAULT_DATA_PATH,
) -> ChromaRetriever:
    """Creates ChromaRetriever reusing the cached model for instant query embeddings."""
    cached_model = get_cached_embedding_model(DEFAULT_MODEL_NAME)
    retriever = ChromaRetriever(
        db_path=db_path,
        collection_name=collection_name,
        data_path=data_path,
    )
    retriever._model = cached_model
    return retriever


# -----------------------------------------------------------------------------
# Mutable Demo Collection (seminar_demo_mutable) Management
# -----------------------------------------------------------------------------

def get_or_create_mutable_collection(
    client: chromadb.PersistentClient,
    collection_name: str = DEFAULT_MUTABLE_COLLECTION,
) -> Collection:
    """Gets or creates the dedicated mutable demo collection with Cosine distance."""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def list_mutable_documents(collection: Collection) -> List[Dict[str, Any]]:
    """Lists all documents stored in the mutable demo collection."""
    data = collection.get(include=["documents", "metadatas"])
    items: List[Dict[str, Any]] = []
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    for doc_id, doc_text, meta in zip(ids, docs, metas):
        meta_dict = meta if meta is not None else {}
        items.append(
            {
                "id": doc_id,
                "title": meta_dict.get("title", ""),
                "topic": meta_dict.get("topic", "General"),
                "text": doc_text,
            }
        )
    items.sort(key=lambda x: x["id"])
    return items


def add_mutable_document(
    collection: Collection,
    model: SentenceTransformer,
    doc_id: str,
    title: str,
    topic: str,
    text: str,
) -> None:
    """Encodes with 'passage: ' and adds into mutable demo collection. Refuses existing IDs."""
    doc_id_clean = doc_id.strip()
    text_clean = text.strip()
    title_clean = title.strip() or "Untitled"
    topic_clean = topic.strip() or "General"

    if not doc_id_clean:
        raise ValueError("Document ID cannot be empty.")
    if not text_clean:
        raise ValueError("Document text cannot be empty.")

    # Refuse if ID already exists to prevent silent overwrite
    existing = collection.get(ids=[doc_id_clean])
    if existing and existing.get("ids"):
        raise ValueError(
            f"Document ID '{doc_id_clean}' already exists in demo collection. Use Update operation to edit it."
        )

    prefixed = f"passage: {text_clean}"
    emb = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)

    collection.upsert(
        ids=[doc_id_clean],
        embeddings=[emb.tolist()],
        documents=[text_clean],
        metadatas=[{"title": title_clean, "topic": topic_clean}],
    )


def update_mutable_document(
    collection: Collection,
    model: SentenceTransformer,
    doc_id: str,
    title: str,
    topic: str,
    text: str,
) -> None:
    """Re-encodes with 'passage: ' and updates an existing document in mutable demo collection."""
    doc_id_clean = doc_id.strip()
    text_clean = text.strip()
    title_clean = title.strip() or "Untitled"
    topic_clean = topic.strip() or "General"

    if not doc_id_clean:
        raise ValueError("Document ID cannot be empty.")
    if not text_clean:
        raise ValueError("Document text cannot be empty.")

    # Verify ID exists before updating
    existing = collection.get(ids=[doc_id_clean])
    if not existing or not existing.get("ids"):
        raise ValueError(
            f"Document ID '{doc_id_clean}' not found in demo collection. Use Add operation to create it."
        )

    prefixed = f"passage: {text_clean}"
    emb = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)

    collection.upsert(
        ids=[doc_id_clean],
        embeddings=[emb.tolist()],
        documents=[text_clean],
        metadatas=[{"title": title_clean, "topic": topic_clean}],
    )


def delete_mutable_document(collection: Collection, doc_id: str) -> None:
    """Deletes a document by ID from the mutable demo collection."""
    doc_id_clean = doc_id.strip()
    if not doc_id_clean:
        raise ValueError("Document ID cannot be empty.")
    collection.delete(ids=[doc_id_clean])


def search_mutable_collection(
    collection: Collection,
    model: SentenceTransformer,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Encodes query with 'query: ' and searches the mutable demo collection."""
    query_clean = query.strip()
    if not query_clean or collection.count() == 0 or top_k <= 0:
        return []

    effective_k = min(top_k, collection.count())
    prefixed = f"query: {query_clean}"
    query_vec = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    formatted: List[Dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for rank, (d_id, text, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
        meta_dict = meta if meta is not None else {}
        formatted.append(
            {
                "rank": rank,
                "id": d_id,
                "title": meta_dict.get("title", ""),
                "topic": meta_dict.get("topic", ""),
                "document": text,
                "distance": float(dist),
            }
        )
    return formatted


# -----------------------------------------------------------------------------
# Official Benchmark Eligibility & Results Checker
# -----------------------------------------------------------------------------

def check_official_benchmark_eligibility(
    review_pool_path: str | Path = DEFAULT_REVIEW_POOL_PATH,
    corpus_path: str | Path = DEFAULT_DATA_PATH,
    official_json_path: str | Path = DEFAULT_OFFICIAL_JSON_PATH,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates if official metrics are eligible to be displayed.
    Requirements:
      1. Corpus file exists.
      2. Review pool exists, is non-empty, and has 100% verified labels (0 pending).
      3. official_metrics.json exists.
      4. Corpus SHA-256 matches the recorded corpus_sha256 in official JSON.
      5. Review pool SHA-256 matches the recorded review_pool_sha256 in official JSON.
    Returns (is_eligible, reason, official_data).
    """
    # 1. Guard against missing corpus file
    c_path = Path(corpus_path)
    if not c_path.exists():
        return False, f"Corpus data file '{corpus_path}' does not exist.", None

    # 2. Guard against missing or empty review pool
    pool_file = Path(review_pool_path)
    if not pool_file.exists():
        return False, f"Review pool file '{review_pool_path}' does not exist.", None

    pending_count = 0
    total_count = 0
    with open(pool_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_count += 1
            status = row.get("review_status", "").strip().lower()
            rel = row.get("verified_relevance", "").strip()
            if status != "verified" or rel not in {"0", "1", "2"}:
                pending_count += 1

    if total_count == 0:
        return False, f"Review pool file '{review_pool_path}' is empty.", None

    if pending_count > 0:
        return (
            False,
            f"Official benchmark is pending human verification ({pending_count}/{total_count} candidate labels pending).",
            None,
        )

    # 3. Guard against missing official metrics file
    json_file = Path(official_json_path)
    if not json_file.exists():
        return (
            False,
            f"Official metrics file '{official_json_path}' not found. Run 'python scripts/run_evaluation.py --official' after review.",
            None,
        )

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            official_data = json.load(f)
    except Exception as e:
        return False, f"Failed to parse '{official_json_path}': {e}", None

    # 4. Check corpus SHA-256 hash consistency
    current_corpus_sha = compute_file_sha256(corpus_path)
    recorded_corpus_sha = official_data.get("metadata", {}).get("corpus_sha256", "")
    if current_corpus_sha != recorded_corpus_sha:
        return (
            False,
            f"Corpus SHA-256 mismatch: recorded {recorded_corpus_sha[:8]}, current file is {current_corpus_sha[:8]}.",
            None,
        )

    # 5. Check review pool SHA-256 hash consistency to prevent stale metrics when labels change
    current_pool_sha = compute_file_sha256(review_pool_path)
    recorded_pool_sha = official_data.get("metadata", {}).get("review_pool_sha256", "")
    if not recorded_pool_sha or current_pool_sha != recorded_pool_sha:
        return (
            False,
            "Review pool has been modified since official metrics were computed (SHA-256 mismatch). Please re-run 'python scripts/run_evaluation.py --official'.",
            None,
        )

    return True, "Official benchmark is verified and valid.", official_data


def load_preview_artifact(preview_path: str | Path = DEFAULT_PREVIEW_JSON_PATH) -> Optional[Dict[str, Any]]:
    """Loads the G3 reproducible rankings preview JSON artifact if present."""
    path = Path(preview_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

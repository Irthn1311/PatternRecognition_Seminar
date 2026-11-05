"""
Smoke test script for Chroma PersistentClient and Multilingual-E5-Small embedding.
Part of Seminar: Pattern Recognition (PatternRecognition_Seminar) - Task G0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
import numpy as np
from sentence_transformers import SentenceTransformer

# Constant definitions
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "smoke_collection"
DEFAULT_DATA_PATH = "./data/smoke_documents.json"
EXPECTED_DIMENSION = 384

DEFAULT_TEST_QUERIES = [
    # Query 1: Lexically varied query vs relevant document (Overfitting)
    "làm sao để tránh việc mô hình học vẹt và ghi nhớ quá mức dữ liệu huấn luyện",
    # Query 2: Query with direct technical terms (CNN / Convolutional Neural Network)
    "mạng nơ-ron tích chập CNN trích xuất đặc trưng không gian từ hình ảnh",
    # Query 3: Query on another topic (Computer Networks / DNS / IP)
    "hệ thống phân giải tên miền và định tuyến địa chỉ IP trong mạng máy tính",
]


class ChromaEmbeddingManager:
    """Manages Multilingual-E5-Small model embeddings and ChromaDB operations."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.db_path = db_path
        self.collection_name = collection_name
        self._model: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Collection] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[Model] Loading '{self.model_name}' on device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)
        return self._client

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            # Using cosine distance metric for normalized embeddings
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        """Embed document passages using 'passage: ' prefix and L2 normalization."""
        prefixed = [f"passage: {text.strip()}" for text in texts]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query using 'query: ' prefix and L2 normalization."""
        prefixed = f"query: {query.strip()}"
        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32)

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Idempotent document ingestion using Chroma collection.upsert().
        Embeddings are precomputed externally and passed directly.
        """
        if not documents:
            print("[Ingest] No documents to ingest.")
            return 0

        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        print(f"[Ingest] Precomputing embeddings for {len(texts)} passages...")
        embeddings = self.embed_passages(texts)

        actual_dim = embeddings.shape[1]
        print(f"[Ingest] Actual embedding dimension: {actual_dim} (Expected: {EXPECTED_DIMENSION})")
        if actual_dim != EXPECTED_DIMENSION:
            raise ValueError(f"Embedding dimension mismatch: got {actual_dim}, expected {EXPECTED_DIMENSION}")

        print(f"[Ingest] Upserting {len(ids)} records into collection '{self.collection_name}'...")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        total_count = self.collection.count()
        print(f"[Ingest] Upsert completed. Total records in collection: {total_count}")
        return total_count

    def search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the collection with normalized query embedding.
        Returns top_k nearest results.
        """
        query_vector = self.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank, (doc_id, text, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            formatted_results.append(
                {
                    "rank": rank,
                    "id": doc_id,
                    "document": text,
                    "metadata": meta,
                    "distance": dist,
                }
            )
        return formatted_results


def print_search_results(query: str, results: List[Dict[str, Any]]) -> None:
    """Pretty prints query results matching Phase D requirements."""
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    for res in results:
        print(f"Rank {res['rank']}: [ID: {res['id']}] (Cosine Distance: {res['distance']:.4f})")
        print(f"  Topic: {res['metadata'].get('topic', 'N/A')}")
        print(f"  Document: {res['document']}")
        print("-" * 80)
    print()


def load_documents_file(filepath: str) -> List[Dict[str, Any]]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Documents file not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chroma Multilingual-E5 Smoke Test (Task G0)")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to local Chroma DB")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME, help="Collection name")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to documents JSON")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into Chroma collection")
    parser.add_argument("--query", type=str, help="Execute a single semantic search query")
    parser.add_argument("--queries", action="store_true", help="Execute all predefined test queries")
    parser.add_argument("--count", action="store_true", help="Print current collection item count")
    parser.add_argument("--top-k", type=int, default=3, help="Number of nearest neighbors to retrieve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager = ChromaEmbeddingManager(
        db_path=args.db_path,
        collection_name=args.collection_name,
    )

    # If no flags specified, run standard full smoke test
    has_action = args.ingest or args.query or args.queries or args.count

    if not has_action or args.ingest:
        docs = load_documents_file(args.data_path)
        print(f"[Main] Loaded {len(docs)} documents from '{args.data_path}'")
        manager.ingest_documents(docs)

    if args.count:
        count = manager.collection.count()
        print(f"[Main] Collection '{args.collection_name}' record count: {count}")

    if args.query:
        results = manager.search(args.query, top_k=args.top_k)
        print_search_results(args.query, results)

    if not has_action or args.queries:
        print("\n[Main] Running predefined Vietnamese test queries:")
        for q in DEFAULT_TEST_QUERIES:
            results = manager.search(q, top_k=args.top_k)
            print_search_results(q, results)


if __name__ == "__main__":
    main()

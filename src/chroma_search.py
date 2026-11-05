"""
Chroma Semantic Search Retriever (Task G2).
Encodes queries with 'query: ' prefix using Multilingual-E5-Small,
executes cosine distance queries on ChromaDB, and supports topic filtering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
import numpy as np
from sentence_transformers import SentenceTransformer

from src.indexer import DEFAULT_COLLECTION_NAME, DEFAULT_DB_PATH, DEFAULT_MODEL_NAME


class ChromaRetriever:
    """Semantic search retriever using ChromaDB and Multilingual-E5-Small embeddings."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device

        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Collection] = None
        self._model: Optional[SentenceTransformer] = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.db_path)
        return self._client

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self._collection = self.client.get_collection(self.collection_name)
        return self._collection

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[Model] Loading '{self.model_name}' for query embedding on '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds query text with 'query: ' prefix and L2 normalization."""
        prefixed = f"query: {query.strip()}"
        emb = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
        topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search with cosine distance.
        Supports optional metadata topic filtering.
        """
        query_str = query.strip()
        if not query_str:
            return []

        total_records = self.collection.count()
        if total_records == 0:
            return []

        # Clamp top_k to total records
        effective_k = min(top_k, total_records)
        if effective_k <= 0:
            return []

        # External query embedding
        query_vec = self.embed_query(query_str)

        # Topic filter via Chroma where
        where_filter = {"topic": topic} if topic else None

        results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=effective_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        formatted: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank, (doc_id, text, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            formatted.append(
                {
                    "rank": rank,
                    "id": doc_id,
                    "title": meta.get("title", ""),
                    "document": text,
                    "topic": meta.get("topic", ""),
                    "source_id": meta.get("source_id", ""),
                    "source_url": meta.get("source_url", ""),
                    "distance": float(dist),
                    "retriever": "chroma",
                }
            )

        return formatted

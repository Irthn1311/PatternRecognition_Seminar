"""
Chroma Indexer for Vietnamese IT Corpus (Task G2).
Handles document ingestion, E5 embedding with 'passage: ' prefix, batching,
manifest-based index consistency checks, and controlled rebuilding.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "seminar_corpus_e5"
DEFAULT_DATA_PATH = "data/corpus/documents.jsonl"
EXPECTED_DIMENSION = 384
MAX_MODEL_TOKENS = 512


def compute_file_sha256(file_path: str | Path) -> str:
    """Computes SHA-256 hash of a file for consistency verification."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def load_corpus_documents(file_path: str | Path) -> List[Dict[str, Any]]:
    """Loads JSONL corpus documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                docs.append(json.loads(line_str))
    return docs


class CorpusIndexer:
    """Manages ChromaDB collection, E5 embeddings, and manifest consistency."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        data_path: str = DEFAULT_DATA_PATH,
        device: str = "cpu",
    ) -> None:
        self.db_path = str(Path(db_path))
        self.collection_name = collection_name
        self.model_name = model_name
        self.data_path = str(Path(data_path))
        self.device = device

        self.manifest_path = Path(self.db_path) / "index_manifest.json"

        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Collection] = None
        self._model: Optional[SentenceTransformer] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)
        return self._client

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[Model] Loading embedding model '{self.model_name}' on device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def inspect_corpus_tokens(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Checks token lengths of all passages against model limit (512)."""
        max_len = 0
        max_doc_id = ""
        lengths = []
        for doc in docs:
            text = f"passage: {doc['text'].strip()}"
            tokens = self.tokenizer.encode(text)
            t_len = len(tokens)
            lengths.append(t_len)
            if t_len > max_len:
                max_len = t_len
                max_doc_id = doc["id"]

        avg_len = sum(lengths) / len(lengths) if lengths else 0
        truncated_count = sum(1 for l in lengths if l > MAX_MODEL_TOKENS)

        return {
            "total_documents": len(docs),
            "max_token_length": max_len,
            "max_doc_id": max_doc_id,
            "avg_token_length": round(avg_len, 2),
            "model_max_tokens": MAX_MODEL_TOKENS,
            "truncated_count": truncated_count,
        }

    def check_index_consistency(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Verifies whether existing Chroma index matches current dataset and config.
        Returns (is_consistent, message, manifest_dict).
        """
        if not self.manifest_path.exists():
            return False, "Manifest file not found. Index has not been built yet.", None

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return False, f"Failed to read index manifest: {e}", None

        # Check collection existence
        existing_cols = [c.name for c in self.client.list_collections()]
        if self.collection_name not in existing_cols:
            return False, f"Collection '{self.collection_name}' not found in database.", manifest

        col = self.client.get_collection(self.collection_name)
        current_count = col.count()

        # Check dataset SHA-256
        if not Path(self.data_path).exists():
            return False, f"Corpus data file '{self.data_path}' does not exist.", manifest

        current_sha = compute_file_sha256(self.data_path)
        if manifest.get("dataset_sha256") != current_sha:
            return (
                False,
                f"Dataset hash mismatch: index built with {manifest.get('dataset_sha256')[:8]}, current file is {current_sha[:8]}.",
                manifest,
            )

        if manifest.get("model_name") != self.model_name:
            return (
                False,
                f"Model mismatch: index built with {manifest.get('model_name')}, current is {self.model_name}.",
                manifest,
            )

        if manifest.get("collection_name") != self.collection_name:
            return False, f"Collection mismatch: manifest={manifest.get('collection_name')}.", manifest

        if manifest.get("document_count") != current_count:
            return (
                False,
                f"Record count mismatch: manifest expected {manifest.get('document_count')}, collection has {current_count}.",
                manifest,
            )

        return (
            True,
            f"Index '{self.collection_name}' is consistent ({current_count} records, SHA256: {current_sha[:8]}).",
            manifest,
        )

    def write_manifest(self, doc_count: int, dataset_sha: str) -> None:
        """Writes metadata manifest to persist index configuration."""
        manifest_data = {
            "dataset_path": self.data_path,
            "dataset_sha256": dataset_sha,
            "document_count": doc_count,
            "model_name": self.model_name,
            "vector_dim": EXPECTED_DIMENSION,
            "collection_name": self.collection_name,
            "distance_metric": "cosine",
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        print(f"[Manifest] Saved index manifest to '{self.manifest_path}'")

    def build_index(self, force_rebuild: bool = False, batch_size: int = 32) -> int:
        """
        Builds or updates Chroma index from corpus.
        If index is already consistent and not force_rebuild, skips re-embedding.
        """
        is_consistent, msg, _ = self.check_index_consistency()
        if is_consistent and not force_rebuild:
            print(f"[Index] {msg}")
            print("[Index] Index is already up-to-date. Skipping re-encoding.")
            return self.collection.count()

        if force_rebuild:
            print(f"[Index] Force rebuild requested. Recreating collection '{self.collection_name}'...")
            existing_cols = [c.name for c in self.client.list_collections()]
            if self.collection_name in existing_cols:
                self.client.delete_collection(self.collection_name)
            self._collection = None

        docs = load_corpus_documents(self.data_path)
        print(f"[Corpus] Loaded {len(docs)} documents from '{self.data_path}'")

        # Inspect token lengths
        token_stats = self.inspect_corpus_tokens(docs)
        print(f"[Token Check] Max token length: {token_stats['max_token_length']}/{token_stats['model_max_tokens']} (Doc: {token_stats['max_doc_id']})")
        print(f"[Token Check] Average token length: {token_stats['avg_token_length']} tokens")
        if token_stats["truncated_count"] > 0:
            print(f"[WARNING] {token_stats['truncated_count']} documents exceed token limit!")
        else:
            print("[Token Check] No documents truncated. All passages fit within model capacity.")

        # Batch encoding with 'passage: ' prefix
        passages = [f"passage: {d['text'].strip()}" for d in docs]
        print(f"[Embedding] Batch encoding {len(passages)} passages on CPU (batch_size={batch_size})...")

        embeddings = self.model.encode(
            passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        actual_dim = embeddings_np.shape[1]
        if actual_dim != EXPECTED_DIMENSION:
            raise ValueError(f"Embedding dimension mismatch: expected {EXPECTED_DIMENSION}, got {actual_dim}")

        # Prepare records and metadata (ensuring no None values)
        ids = [d["id"] for d in docs]
        documents = [d["text"] for d in docs]
        metadatas = [
            {
                "topic": d["topic"],
                "title": d["title"],
                "source_id": d["source_id"],
                "source_url": d.get("source_url") or "",
                "chunk_index": int(d.get("chunk_index", 0)),
            }
            for d in docs
        ]

        print(f"[Chroma] Upserting {len(ids)} records into collection '{self.collection_name}'...")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_np.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

        total_count = self.collection.count()
        print(f"[Chroma] Ingestion complete. Total records in collection: {total_count}")

        # Update manifest
        dataset_sha = compute_file_sha256(self.data_path)
        self.write_manifest(total_count, dataset_sha)

        return total_count

"""
Dataset validation script for Vietnamese IT Corpus & Retrieval Benchmark (Task G1).
Validates schema integrity, foreign key references, splits, and quality constraints.
Uses strictly Python standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

DEFAULT_SOURCES_PATH = "data/corpus/sources.jsonl"
DEFAULT_DOCUMENTS_PATH = "data/corpus/documents.jsonl"
DEFAULT_QUERIES_PATH = "data/eval/queries.jsonl"
DEFAULT_QRELS_PATH = "data/eval/qrels_draft.jsonl"

VALID_TOPICS = {
    "Machine Learning",
    "Deep Learning",
    "Computer Vision",
    "Database",
    "Algorithms",
    "Computer Networks / Security",
}

VALID_CATEGORIES = {
    "semantic_paraphrase",
    "direct_technical",
    "ambiguous_confusable",
    "exact_match",
    "unanswerable",
}

VALID_SPLITS = {"dev", "test"}
VALID_PROVENANCE = {"adapted", "self_composed", "quoted"}
VALID_RELEVANCE = {0, 1, 2}
VALID_LABEL_STATUS = {"proposed"}


class DatasetValidator:
    def __init__(
        self,
        sources_path: str = DEFAULT_SOURCES_PATH,
        documents_path: str = DEFAULT_DOCUMENTS_PATH,
        queries_path: str = DEFAULT_QUERIES_PATH,
        qrels_path: str = DEFAULT_QRELS_PATH,
    ) -> None:
        self.sources_path = Path(sources_path)
        self.documents_path = Path(documents_path)
        self.queries_path = Path(queries_path)
        self.qrels_path = Path(qrels_path)

        self.errors: List[str] = []
        self.warnings: List[str] = []

        self.sources: Dict[str, Dict[str, Any]] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.queries: Dict[str, Dict[str, Any]] = {}
        self.qrels: List[Dict[str, Any]] = []

    def _load_jsonl(self, path: Path, file_desc: str) -> List[Tuple[int, Dict[str, Any]]]:
        if not path.exists():
            self.errors.append(f"[{file_desc}] File does not exist: {path}")
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        obj = json.loads(line_str)
                        records.append((line_num, obj))
                    except json.JSONDecodeError as e:
                        self.errors.append(f"[{file_desc}] Line {line_num}: JSON decode error: {e}")
        except Exception as e:
            self.errors.append(f"[{file_desc}] Failed to read file {path}: {e}")
        return records

    def validate_sources(self) -> None:
        records = self._load_jsonl(self.sources_path, "Sources")
        required_fields = {"source_id", "name", "author_or_org", "license", "provenance"}

        for line_num, src in records:
            missing = required_fields - set(src.keys())
            if missing:
                self.errors.append(f"[Sources] Line {line_num}: Missing required fields: {missing}")
                continue

            sid = src["source_id"]
            if not sid or not isinstance(sid, str):
                self.errors.append(f"[Sources] Line {line_num}: Invalid or empty source_id")
            elif sid in self.sources:
                self.errors.append(f"[Sources] Line {line_num}: Duplicate source_id '{sid}'")
            else:
                self.sources[sid] = src

            prov = src.get("provenance")
            if prov not in VALID_PROVENANCE:
                self.errors.append(f"[Sources] Line {line_num} ('{sid}'): Invalid provenance '{prov}'")

            if prov == "adapted" and not src.get("source_url"):
                self.warnings.append(f"[Sources] '{sid}': Adapted source missing source_url")

    def validate_documents(self) -> None:
        records = self._load_jsonl(self.documents_path, "Documents")
        required_fields = {"id", "source_id", "title", "text", "topic", "language", "chunk_index", "provenance"}

        seen_texts: Dict[str, str] = {}

        for line_num, doc in records:
            missing = required_fields - set(doc.keys())
            if missing:
                self.errors.append(f"[Documents] Line {line_num}: Missing required fields: {missing}")
                continue

            doc_id = doc["id"]
            if not doc_id or not isinstance(doc_id, str):
                self.errors.append(f"[Documents] Line {line_num}: Invalid or empty id")
            elif doc_id in self.documents:
                self.errors.append(f"[Documents] Line {line_num}: Duplicate document id '{doc_id}'")
            else:
                self.documents[doc_id] = doc

            text = doc.get("text", "")
            if not text or not text.strip():
                self.errors.append(f"[Documents] '{doc_id}': Document text is empty")
            else:
                norm_text = " ".join(text.lower().split())
                if norm_text in seen_texts:
                    self.errors.append(f"[Documents] '{doc_id}': Exact duplicate text with '{seen_texts[norm_text]}'")
                else:
                    seen_texts[norm_text] = doc_id

            sid = doc.get("source_id")
            if sid not in self.sources:
                self.errors.append(f"[Documents] '{doc_id}': Unknown source_id '{sid}' (not in sources registry)")

            topic = doc.get("topic")
            if topic not in VALID_TOPICS:
                self.errors.append(f"[Documents] '{doc_id}': Invalid topic '{topic}'. Must be one of {VALID_TOPICS}")

            lang = doc.get("language")
            if lang != "vi":
                self.errors.append(f"[Documents] '{doc_id}': Language must be 'vi', got '{lang}'")

            prov = doc.get("provenance")
            if prov not in VALID_PROVENANCE:
                self.errors.append(f"[Documents] '{doc_id}': Invalid provenance '{prov}'")

    def validate_queries(self) -> None:
        records = self._load_jsonl(self.queries_path, "Queries")
        required_fields = {"query_id", "query", "category", "split", "is_unanswerable"}

        seen_queries: Dict[str, str] = {}

        for line_num, q in records:
            missing = required_fields - set(q.keys())
            if missing:
                self.errors.append(f"[Queries] Line {line_num}: Missing required fields: {missing}")
                continue

            qid = q["query_id"]
            if not qid or not isinstance(qid, str):
                self.errors.append(f"[Queries] Line {line_num}: Invalid or empty query_id")
            elif qid in self.queries:
                self.errors.append(f"[Queries] Line {line_num}: Duplicate query_id '{qid}'")
            else:
                self.queries[qid] = q

            query_text = q.get("query", "")
            if not query_text or not query_text.strip():
                self.errors.append(f"[Queries] '{qid}': Query text is empty")
            else:
                norm_q = " ".join(query_text.lower().split())
                if norm_q in seen_queries:
                    self.errors.append(f"[Queries] '{qid}': Duplicate query text with '{seen_queries[norm_q]}'")
                else:
                    seen_queries[norm_q] = qid

            cat = q.get("category")
            if cat not in VALID_CATEGORIES:
                self.errors.append(f"[Queries] '{qid}': Invalid category '{cat}'. Must be one of {VALID_CATEGORIES}")

            split = q.get("split")
            if split not in VALID_SPLITS:
                self.errors.append(f"[Queries] '{qid}': Invalid split '{split}'. Must be one of {VALID_SPLITS}")

            if not isinstance(q.get("is_unanswerable"), bool):
                self.errors.append(f"[Queries] '{qid}': is_unanswerable must be boolean")

    def validate_qrels(self) -> None:
        records = self._load_jsonl(self.qrels_path, "Qrels")
        required_fields = {"query_id", "doc_id", "relevance", "label_status"}

        seen_pairs: Set[Tuple[str, str]] = set()
        query_positive_counts: Counter[str] = Counter()

        for line_num, qr in records:
            missing = required_fields - set(qr.keys())
            if missing:
                self.errors.append(f"[Qrels] Line {line_num}: Missing required fields: {missing}")
                continue

            qid = qr.get("query_id")
            did = qr.get("doc_id")
            rel = qr.get("relevance")
            status = qr.get("label_status")

            if qid not in self.queries:
                self.errors.append(f"[Qrels] Line {line_num}: Referenced query_id '{qid}' does not exist")

            if did not in self.documents:
                self.errors.append(f"[Qrels] Line {line_num}: Referenced doc_id '{did}' does not exist")

            pair = (str(qid), str(did))
            if pair in seen_pairs:
                self.errors.append(f"[Qrels] Line {line_num}: Duplicate query-document pair {pair}")
            else:
                seen_pairs.add(pair)

            if rel not in VALID_RELEVANCE:
                self.errors.append(f"[Qrels] Line {line_num}: Invalid relevance '{rel}'. Must be {VALID_RELEVANCE}")
            elif rel > 0:
                query_positive_counts[str(qid)] += 1

            if status not in VALID_LABEL_STATUS:
                self.errors.append(
                    f"[Qrels] Line {line_num}: label_status must be 'proposed' for draft labels, got '{status}'"
                )

            # Check consistency with unanswerable queries
            if qid in self.queries:
                is_unanswerable = self.queries[qid].get("is_unanswerable", False)
                if is_unanswerable and rel > 0:
                    self.errors.append(
                        f"[Qrels] Query '{qid}' is marked unanswerable but assigned positive relevance {rel} for '{did}'"
                    )

            self.qrels.append(qr)

        # Check answerable queries have at least one positive candidate
        for qid, q in self.queries.items():
            if not q.get("is_unanswerable", False):
                if query_positive_counts[qid] == 0:
                    self.warnings.append(f"[Qrels] Answerable query '{qid}' has no positive draft candidate (rel > 0)")

    def print_statistics(self) -> None:
        print("\n" + "=" * 60)
        print("DATASET VALIDATION SUMMARY & STATISTICS")
        print("=" * 60)

        # Corpus Stats
        print(f"\n[Corpus Documents]: {len(self.documents)} passages")
        topic_counts = Counter(doc["topic"] for doc in self.documents.values())
        for topic, count in sorted(topic_counts.items()):
            print(f"  - {topic}: {count}")

        # Sources Stats
        print(f"\n[Corpus Sources]: {len(self.sources)} sources")
        source_prov = Counter(src["provenance"] for src in self.sources.values())
        for prov, count in sorted(source_prov.items()):
            print(f"  - {prov}: {count}")

        # Queries Stats
        print(f"\n[Evaluation Queries]: {len(self.queries)} queries")
        cat_counts = Counter(q["category"] for q in self.queries.values())
        for cat, count in sorted(cat_counts.items()):
            print(f"  - Category '{cat}': {count}")

        split_counts = Counter(q["split"] for q in self.queries.values())
        for split, count in sorted(split_counts.items()):
            print(f"  - Split '{split}': {count}")

        unans_count = sum(1 for q in self.queries.values() if q.get("is_unanswerable"))
        print(f"  - Unanswerable queries: {unans_count}")

        # Qrels Stats
        print(f"\n[Draft Qrels]: {len(self.qrels)} relevance judgments")
        rel_counts = Counter(qr["relevance"] for qr in self.qrels)
        for rel in sorted(VALID_RELEVANCE):
            desc = {0: "0 (Irrelevant/Negative)", 1: "1 (Partially relevant)", 2: "2 (Directly relevant)"}.get(rel, "")
            print(f"  - Score {desc}: {rel_counts[rel]}")

        # Validation status
        print("\n" + "-" * 60)
        print(f"ERRORS:   {len(self.errors)}")
        print(f"WARNINGS: {len(self.warnings)}")
        print("-" * 60)

        if self.warnings:
            print("\nWARNING DETAILS:")
            for w in self.warnings:
                print(f"  [WARN] {w}")

        if self.errors:
            print("\nERROR DETAILS:")
            for e in self.errors:
                print(f"  [ERROR] {e}")

        if not self.errors:
            print("\n>>> VALIDATION RESULT: PASS (All checks passed successfully) <<<\n")
        else:
            print(f"\n>>> VALIDATION RESULT: FAIL ({len(self.errors)} errors found) <<<\n")

    def run(self) -> bool:
        self.validate_sources()
        self.validate_documents()
        self.validate_queries()
        self.validate_qrels()
        self.print_statistics()
        return len(self.errors) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Vietnamese IT Corpus & Qrels Dataset")
    parser.add_argument("--sources", default=DEFAULT_SOURCES_PATH, help="Path to sources.jsonl")
    parser.add_argument("--documents", default=DEFAULT_DOCUMENTS_PATH, help="Path to documents.jsonl")
    parser.add_argument("--queries", default=DEFAULT_QUERIES_PATH, help="Path to queries.jsonl")
    parser.add_argument("--qrels", default=DEFAULT_QRELS_PATH, help="Path to qrels_draft.jsonl")
    args = parser.parse_args()

    validator = DatasetValidator(
        sources_path=args.sources,
        documents_path=args.documents,
        queries_path=args.queries,
        qrels_path=args.qrels,
    )
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

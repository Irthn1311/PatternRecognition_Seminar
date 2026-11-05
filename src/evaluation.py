"""
Retrieval Evaluation Module (Task G3).
Implements exact IR metrics (Hit@5, MRR@5, nDCG@5) and the Verified-Label Gate.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def calculate_hit_at_k(relevances: List[int], k: int = 5) -> float:
    """
    Hit@K: 1.0 if at least one document in Top-K has primary relevance (grade 2), else 0.0.
    """
    top_k = relevances[:k]
    return 1.0 if any(r == 2 for r in top_k) else 0.0


def calculate_mrr_at_k(relevances: List[int], k: int = 5) -> float:
    """
    MRR@K (Reciprocal Rank): Reciprocal of the 1-based rank of the FIRST document
    with primary relevance (grade 2) in Top-K. Returns 0.0 if not found.
    """
    top_k = relevances[:k]
    for rank_idx, rel in enumerate(top_k, start=1):
        if rel == 2:
            return 1.0 / rank_idx
    return 0.0


def calculate_dcg_at_k(relevances: List[int], k: int = 5) -> float:
    """
    Discounted Cumulative Gain:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log_2(i + 1)
    """
    dcg = 0.0
    for rank_idx, rel in enumerate(relevances[:k], start=1):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(rank_idx + 1.0)
        dcg += gain / discount
    return dcg


def calculate_ndcg_at_k(
    retrieved_relevances: List[int],
    all_judged_relevances: List[int],
    k: int = 5,
) -> float:
    """
    Normalized Discounted Cumulative Gain:
        nDCG@K = DCG@K / IDCG@K
    where IDCG is computed on the sorted ideal relevances from all judged candidates in the pool.
    Returns 0.0 if IDCG@K is 0.
    """
    dcg = calculate_dcg_at_k(retrieved_relevances, k=k)
    ideal_sorted = sorted(all_judged_relevances, reverse=True)
    idcg = calculate_dcg_at_k(ideal_sorted, k=k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


class ReviewPoolGate:
    """
    Enforces the verified-label gate for retrieval evaluation.
    Only allows official metrics if all required candidate labels are verified by humans.
    """

    def __init__(self, review_pool_csv: str | Path) -> None:
        self.csv_path = Path(review_pool_csv)
        self.rows: List[Dict[str, str]] = []
        # query_id -> doc_id -> row
        self.pool_by_query: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Review pool CSV not found: {self.csv_path}")

        self.rows = []
        self.pool_by_query = {}

        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)
                qid = row["query_id"]
                did = row["doc_id"]
                if qid not in self.pool_by_query:
                    self.pool_by_query[qid] = {}
                self.pool_by_query[qid][did] = row

    def get_query_label_status(
        self, query_id: str, required_doc_ids: List[str]
    ) -> Tuple[bool, int, List[str]]:
        """
        Checks whether all required doc IDs for a query are verified.
        Returns (all_verified, missing_count, missing_doc_ids).
        """
        query_pool = self.pool_by_query.get(query_id, {})
        missing: List[str] = []

        for did in required_doc_ids:
            if did not in query_pool:
                missing.append(did)
                continue
            row = query_pool[did]
            status = row.get("review_status", "").strip().lower()
            rel_str = row.get("verified_relevance", "").strip()

            if status != "verified" or rel_str not in {"0", "1", "2"}:
                missing.append(did)

        return len(missing) == 0, len(missing), missing

    def get_verified_relevance(self, query_id: str, doc_id: str) -> Optional[int]:
        """Returns verified relevance integer if verified, else None."""
        row = self.pool_by_query.get(query_id, {}).get(doc_id)
        if not row:
            return None
        if row.get("review_status", "").strip().lower() == "verified":
            val = row.get("verified_relevance", "").strip()
            if val in {"0", "1", "2"}:
                return int(val)
        return None

    def get_proposed_relevance(self, query_id: str, doc_id: str) -> Optional[int]:
        """Returns proposed relevance integer from draft if available."""
        row = self.pool_by_query.get(query_id, {}).get(doc_id)
        if not row:
            return None
        val = row.get("proposed_relevance", "").strip()
        if val in {"0", "1", "2"}:
            return int(val)
        return None

    def get_all_verified_relevances_for_query(self, query_id: str) -> List[int]:
        """Returns all verified relevance scores for candidates of query_id in the pool."""
        query_pool = self.pool_by_query.get(query_id, {})
        relevances: List[int] = []
        for did, row in query_pool.items():
            if row.get("review_status", "").strip().lower() == "verified":
                val = row.get("verified_relevance", "").strip()
                if val in {"0", "1", "2"}:
                    relevances.append(int(val))
        return relevances

    def validate_gate(
        self,
        subset_query_ids: List[str],
        retrieved_rankings: Dict[str, Dict[str, List[str]]],
        top_k: int = 5,
    ) -> Tuple[bool, List[str]]:
        """
        Validates all conditions for official evaluation:
        1. top_k == 5
        2. All subset query IDs are present in pool
        3. No duplicate (query_id, doc_id) pairs
        4. All retrieved docs in Chroma and BM25 are present in pool
        5. All candidates in pool for subset queries have review_status='verified' and verified_relevance in {0, 1, 2}
        """
        errors: List[str] = []

        if top_k != 5:
            errors.append(f"Official evaluation is strictly standardized at top_k=5. Received top_k={top_k}.")

        # Check duplicate pairs
        seen_pairs: Set[Tuple[str, str]] = set()
        for r in self.rows:
            pair = (r["query_id"], r["doc_id"])
            if pair in seen_pairs:
                errors.append(f"Duplicate (query_id, doc_id) pair found in review pool: {pair}")
            seen_pairs.add(pair)

        # Check subset queries in pool
        for qid in subset_query_ids:
            if qid not in self.pool_by_query:
                errors.append(f"Query '{qid}' from evaluation subset is missing from review pool.")

        # Check retrieved docs and pool candidate verification
        for qid in subset_query_ids:
            q_pool = self.pool_by_query.get(qid, {})
            ranks = retrieved_rankings.get(qid, {})
            chroma_top = ranks.get("chroma", [])
            bm25_top = ranks.get("bm25", [])

            # Check retrieved docs exist in pool
            for did in chroma_top:
                if did not in q_pool:
                    errors.append(f"Query '{qid}': Chroma Top-5 doc '{did}' is missing from review pool.")
            for did in bm25_top:
                if did not in q_pool:
                    errors.append(f"Query '{qid}': BM25 Top-5 doc '{did}' is missing from review pool.")

            # Check all candidates for this query have verified relevance
            for did, row in q_pool.items():
                status = row.get("review_status", "").strip().lower()
                rel = row.get("verified_relevance", "").strip()
                if status != "verified" or rel not in {"0", "1", "2"}:
                    errors.append(
                        f"Query '{qid}', Doc '{did}': Not verified (status='{status}', verified_relevance='{rel}')."
                    )

        return len(errors) == 0, errors

    def get_coverage_statistics(self, query_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculates verification coverage across the requested queries."""
        target_rows = self.rows
        if query_ids is not None:
            q_set = set(query_ids)
            target_rows = [r for r in self.rows if r["query_id"] in q_set]

        total = len(target_rows)
        verified = sum(1 for r in target_rows if r.get("review_status", "").strip().lower() == "verified" and r.get("verified_relevance", "").strip() in {"0", "1", "2"})
        pending = total - verified

        # Distribution of verified vs proposed
        verified_dist: Dict[str, int] = {"0": 0, "1": 0, "2": 0}
        proposed_dist: Dict[str, int] = {"0": 0, "1": 0, "2": 0, "unjudged": 0}

        for r in target_rows:
            v_rel = r.get("verified_relevance", "").strip()
            if v_rel in verified_dist and r.get("review_status", "").strip().lower() == "verified":
                verified_dist[v_rel] += 1

            p_rel = r.get("proposed_relevance", "").strip()
            if p_rel in {"0", "1", "2"}:
                proposed_dist[p_rel] += 1
            else:
                proposed_dist["unjudged"] += 1

        return {
            "total_pairs": total,
            "verified_pairs": verified,
            "pending_pairs": pending,
            "completion_percentage": round(100.0 * verified / total, 1) if total > 0 else 0.0,
            "verified_distribution": verified_dist,
            "proposed_distribution": proposed_dist,
        }


def run_metrics_fixture_test() -> Dict[str, Any]:
    """
    Self-verification fixture test:
    Validates Hit@5, MRR@5, nDCG@5 against hand-calculated values.
    """
    # Ranking with relevances: [1, 0, 2, 0, 1]
    retrieved_rel = [1, 0, 2, 0, 1]
    # Pool has: [2, 2, 1, 1, 0, 0] -> Ideal Top-5 is [2, 2, 1, 1, 0]
    pool_rel = [2, 2, 1, 1, 0, 0]

    hit5 = calculate_hit_at_k(retrieved_rel, k=5)
    mrr5 = calculate_mrr_at_k(retrieved_rel, k=5)
    dcg5 = calculate_dcg_at_k(retrieved_rel, k=5)
    ndcg5 = calculate_ndcg_at_k(retrieved_rel, pool_rel, k=5)

    expected_hit = 1.0
    expected_mrr = 1.0 / 3.0  # 0.3333333333333333
    expected_dcg = 1.0 + 0.0 + (3.0 / 2.0) + 0.0 + (1.0 / math.log2(6))  # 2.886852807...
    idcg5 = (3.0 / 1.0) + (3.0 / math.log2(3)) + (1.0 / 2.0) + (1.0 / math.log2(5))  # 5.823465715...
    expected_ndcg = expected_dcg / idcg5

    assert abs(hit5 - expected_hit) < 1e-9, f"Hit@5 mismatch: {hit5} vs {expected_hit}"
    assert abs(mrr5 - expected_mrr) < 1e-9, f"MRR@5 mismatch: {mrr5} vs {expected_mrr}"
    assert abs(dcg5 - expected_dcg) < 1e-9, f"DCG@5 mismatch: {dcg5} vs {expected_dcg}"
    assert abs(ndcg5 - expected_ndcg) < 1e-9, f"nDCG@5 mismatch: {ndcg5} vs {expected_ndcg}"

    return {
        "status": "PASS",
        "hit5": round(hit5, 4),
        "mrr5": round(mrr5, 4),
        "dcg5": round(dcg5, 4),
        "idcg5": round(idcg5, 4),
        "ndcg5": round(ndcg5, 4),
    }

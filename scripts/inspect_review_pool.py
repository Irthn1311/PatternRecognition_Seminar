"""
Inspect and list the 105 candidate pairs grouped by query_id in data/eval/review_pool.csv.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

csv_path = Path("data/eval/review_pool.csv")

groups = defaultdict(list)
with open(csv_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        groups[row["query_id"]].append(row)

print(f"Total query groups: {len(groups)}")
print(f"Total candidate pairs: {sum(len(v) for v in groups.values())}")

for qid, rows in groups.items():
    qtext = rows[0]["query_text"]
    cat = rows[0]["category"]
    unans = rows[0]["is_unanswerable"]
    print(f"\n================================================================================")
    print(f"Query {qid} ({len(rows)} candidates) | Thể loại: {cat} | Unanswerable: {unans}")
    print(f"Nội dung: \"{qtext}\"")
    print("--------------------------------------------------------------------------------")
    for idx, r in enumerate(rows, 1):
        did = r["doc_id"]
        title = r["doc_title"]
        prop = r["proposed_relevance"] or "(trống)"
        c_rank = r["chroma_rank"] or "-"
        b_rank = r["bm25_rank"] or "-"
        note = r["proposed_note"] or ""
        print(f"  {idx:2d}. [{did}] {title}")
        print(f"      Chroma: #{c_rank} | BM25: #{b_rank} | Draft Relevance: {prop} | Ghi chú: {note}")

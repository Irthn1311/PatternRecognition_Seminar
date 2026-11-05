"""
Verification script for Streamlit Demo Acceptance Checks (Tasks G4 & G5).

Validates the 4 core acceptance criteria:
  Check 1: Dual Retrieval Search (Chroma vs BM25)
  Check 2: Metadata Topic Pre-Filtering
  Check 3: Isolated Mutable Collection CRUD (seminar_demo_mutable without affecting seminar_corpus_e5)
  Check 4: Experiment Results & Official Benchmark Gate Status
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_search import BM25Retriever
from src.chroma_search import ChromaRetriever
from src.indexer import DEFAULT_COLLECTION_NAME, CorpusIndexer
from src.ui_helpers import (
    DEFAULT_MUTABLE_COLLECTION,
    DEFAULT_OFFICIAL_JSON_PATH,
    DEFAULT_PREVIEW_JSON_PATH,
    DEFAULT_REVIEW_POOL_PATH,
    add_mutable_document,
    check_official_benchmark_eligibility,
    delete_mutable_document,
    get_cached_chroma_client,
    get_cached_embedding_model,
    get_or_create_mutable_collection,
    list_mutable_documents,
    load_preview_artifact,
    search_mutable_collection,
    update_mutable_document,
)


def run_acceptance_checks() -> None:
    print("=" * 80)
    print("RUNNING G4+G5 DEMO ACCEPTANCE CHECKS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Check 1: Search on Chroma and BM25
    # -------------------------------------------------------------------------
    print("\n[Check 1: Dual Retrieval Search]")
    query = "mạng nơ-ron tích chập CNN trích xuất đặc trưng hình ảnh"
    chroma_retriever = ChromaRetriever()
    bm25_retriever = BM25Retriever()

    c_res = chroma_retriever.search(query, top_k=5)
    b_res = bm25_retriever.search(query, top_k=5)

    assert len(c_res) == 5, f"Expected 5 Chroma results, got {len(c_res)}"
    assert len(b_res) == 5, f"Expected 5 BM25 results, got {len(b_res)}"
    assert "distance" in c_res[0], "Chroma result missing distance"
    assert "score" in b_res[0], "BM25 result missing score"

    print(f"  Chroma Top-1: [{c_res[0]['id']}] {c_res[0]['title']} (Cosine Dist: {c_res[0]['distance']:.4f})")
    print(f"  BM25 Top-1:   [{b_res[0]['id']}] {b_res[0]['title']} (BM25 Score: {b_res[0]['score']:.4f})")
    print("Check 1 PASS: Both retrievers executed successfully on real data.")

    # -------------------------------------------------------------------------
    # Check 2: Metadata Filtering
    # -------------------------------------------------------------------------
    print("\n[Check 2: Metadata Topic Filtering]")
    topic = "Database"
    c_filtered = chroma_retriever.search(query, top_k=4, topic=topic)
    b_filtered = bm25_retriever.search(query, top_k=4, topic=topic)

    assert len(c_filtered) > 0, "Chroma filtered search returned 0 results"
    assert len(b_filtered) > 0, "BM25 filtered search returned 0 results"
    assert all(item["topic"] == topic for item in c_filtered), "Chroma result returned non-matching topic"
    assert all(item["topic"] == topic for item in b_filtered), "BM25 result returned non-matching topic"

    print(f"  Chroma returned {len(c_filtered)} results, all topic='{topic}'.")
    print(f"  BM25 returned {len(b_filtered)} results, all topic='{topic}'.")
    print("Check 2 PASS: Metadata pre-filtering strictly enforced on both retrievers.")

    # -------------------------------------------------------------------------
    # Check 3: Isolated Mutable Collection CRUD (Safe in Temporary DB)
    # -------------------------------------------------------------------------
    print("\n[Check 3: Isolated Mutable Collection CRUD]")
    import tempfile
    import chromadb

    # Inspect real DB official collection in read-only mode before test
    real_client = get_cached_chroma_client()
    official_col = real_client.get_collection(DEFAULT_COLLECTION_NAME)
    initial_official_count = official_col.count()
    assert initial_official_count == 114, f"Expected 114 official records, got {initial_official_count}"
    print(f"  Official collection count before CRUD: {initial_official_count} (seminar_corpus_e5)")

    # Record real demo collection IDs in real DB before test
    real_demo_col = get_or_create_mutable_collection(real_client, DEFAULT_MUTABLE_COLLECTION)
    real_demo_ids_before = set(d["id"] for d in list_mutable_documents(real_demo_col))

    # Execute CRUD inside an isolated temporary directory to protect user's real demo data
    with tempfile.TemporaryDirectory(prefix="chroma_demo_crud_test_", ignore_cleanup_errors=True) as tmp_dir:
        tmp_client = chromadb.PersistentClient(path=str(Path(tmp_dir) / "test_chroma"))
        test_demo_col = get_or_create_mutable_collection(tmp_client, "test_mutable_collection")
        model = get_cached_embedding_model()

        # Step A: Add
        test_id = "demo_test_crud_01"
        add_mutable_document(
            test_demo_col,
            model,
            doc_id=test_id,
            title="Học Tăng Cường Demo",
            topic="Machine Learning",
            text="Học tăng cường (Reinforcement Learning) là nhánh học máy tối ưu chính sách thông qua phần thưởng.",
        )
        docs_after_add = list_mutable_documents(test_demo_col)
        assert any(d["id"] == test_id for d in docs_after_add), "Added document not found in demo collection"
        print(f"  [CRUD Add] Added '{test_id}', isolated collection count: {len(docs_after_add)}")

        # Step B: Read & Search
        search_res = search_mutable_collection(test_demo_col, model, "học tăng cường phần thưởng", top_k=1)
        assert len(search_res) == 1, "Search on demo collection failed"
        assert search_res[0]["id"] == test_id, f"Expected {test_id}, got {search_res[0]['id']}"
        print(f"  [CRUD Search] Found '{search_res[0]['id']}' with cosine distance {search_res[0]['distance']:.4f}")

        # Step C: Update
        updated_title = "Học Tăng Cường Nâng Cao"
        updated_text = "Học tăng cường nâng cao với thuật toán Deep Q-Network và Policy Gradient."
        update_mutable_document(
            test_demo_col,
            model,
            doc_id=test_id,
            title=updated_title,
            topic="Machine Learning",
            text=updated_text,
        )
        docs_after_update = list_mutable_documents(test_demo_col)
        target_doc = next(d for d in docs_after_update if d["id"] == test_id)
        assert target_doc["title"] == updated_title, "Title was not updated"
        assert target_doc["text"] == updated_text, "Text was not updated"
        print(f"  [CRUD Update] Updated '{test_id}', title is now: '{target_doc['title']}'")

        # Step D: Delete
        delete_mutable_document(test_demo_col, test_id)
        docs_after_delete = list_mutable_documents(test_demo_col)
        assert not any(d["id"] == test_id for d in docs_after_delete), "Deleted document still present in demo collection"
        print(f"  [CRUD Delete] Deleted '{test_id}', isolated collection count: {len(docs_after_delete)}")

    # Verify real DB demo collection and official collection remain completely untouched
    real_demo_ids_after = set(d["id"] for d in list_mutable_documents(real_demo_col))
    assert real_demo_ids_after == real_demo_ids_before, "Real demo collection was modified during test!"

    final_official_count = official_col.count()
    assert final_official_count == 114, f"Official collection corrupted! Expected 114, got {final_official_count}"
    print(f"  Official collection count after CRUD: {final_official_count} (unchanged)")
    print("Check 3 PASS: CRUD operations executed in isolated temp DB; real demo & official collections 100% untouched.")

    # -------------------------------------------------------------------------
    # Check 4: Experiment Results & Official Benchmark Gate Status
    # -------------------------------------------------------------------------
    print("\n[Check 4: Experiment Results & Gate Status]")
    preview = load_preview_artifact(DEFAULT_PREVIEW_JSON_PATH)
    assert preview is not None, f"Preview artifact not found at {DEFAULT_PREVIEW_JSON_PATH}"
    assert "eval_results" in preview, "Preview artifact missing eval_results"
    assert len(preview["eval_results"]) == 12, f"Expected 12 queries in preview, got {len(preview['eval_results'])}"
    print(f"  Loaded preview artifact with {len(preview['eval_results'])} test queries.")

    is_eligible, reason, official_data = check_official_benchmark_eligibility(
        review_pool_path=DEFAULT_REVIEW_POOL_PATH,
        official_json_path=DEFAULT_OFFICIAL_JSON_PATH,
    )
    print(f"  Official benchmark eligibility: {is_eligible}")
    print(f"  Gate reason: \"{reason}\"")
    assert not is_eligible, "Official benchmark should be INELIGIBLE when 105 labels are pending!"
    assert official_data is None, "Official data should not be loaded when pending"
    print("Check 4 PASS: Experiment tab displays preview and correctly locks official benchmark.")

    print("\n" + "=" * 80)
    print("ALL 4 DEMO ACCEPTANCE CHECKS PASSED SUCCESSFULLY (PASS)")
    print("=" * 80)


if __name__ == "__main__":
    run_acceptance_checks()

"""
Two-process persistence verification for ChromaDB (Task G0 - Phase E).
Executes Process A (ingest & exit) and Process B (fresh process reopen & query).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PERSIST_TEST_DB = "./chroma_db_persist_test"
PYTHON_EXE = sys.executable


def run_process(args: list[str], desc: str) -> subprocess.CompletedProcess[str]:
    print(f"\n--- [Process Step] {desc} ---")
    cmd = [PYTHON_EXE, "scripts/smoke_chroma.py", "--db-path", PERSIST_TEST_DB] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(f"[stderr]: {result.stderr.strip()}", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed with exit code {result.returncode}")
    return result


def main() -> None:
    # Cleanup any previous test db
    test_db_path = Path(PERSIST_TEST_DB)
    if test_db_path.exists():
        shutil.rmtree(test_db_path)

    try:
        # Process A: Write data and exit completely
        print("[Check 2] Starting Process A: Ingest data into persistent storage...")
        res_a = run_process(["--ingest"], "Process A - Ingestion")
        assert "Upsert completed. Total records in collection: 16" in res_a.stdout, (
            "Process A failed to upsert 16 documents"
        )
        print("[Check 2] Process A exited successfully.")

        # Process B: Completely independent Python process reading from disk
        print("\n[Check 2] Starting Process B: Separate process opening database and querying...")
        res_b = run_process(
            ["--count", "--query", "thuật toán Dijkstra tìm đường đi ngắn nhất đồ thị"],
            "Process B - Reopen & Query",
        )
        assert "record count: 16" in res_b.stdout, "Process B did not find 16 records on disk"
        assert "doc_algo_01" in res_b.stdout, "Process B failed to retrieve doc_algo_01"
        print("[Check 2] Process B successfully reopened database, verified count 16, and queried correctly.")

        # Check 3: Idempotency check in a new invocation
        print("\n[Check 3] Starting Idempotency Test: Ingest again into the same DB...")
        res_c = run_process(["--ingest", "--count"], "Process C - Re-ingest (Idempotency)")
        assert "record count: 16" in res_c.stdout, "Record count increased after re-ingestion!"
        print("[Check 3] Idempotency PASS: Record count remained exactly 16 after duplicate upsert.")

        print("\n" + "=" * 50)
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (PASS)")
        print("=" * 50)

    finally:
        # Cleanup test folder
        if test_db_path.exists():
            shutil.rmtree(test_db_path, ignore_errors=True)


if __name__ == "__main__":
    main()

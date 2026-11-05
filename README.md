# PatternRecognition_Seminar

Dự án Seminar môn Nhận diện Mẫu (Pattern Recognition) - Nghiên cứu Chroma VectorDB, Embedding và Semantic Search cho học liệu tiếng Việt.

- **GitHub Issue #1 (TASK G0):** [Issue #1 — Bootstrap Windows repository + Chroma smoke test](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/1)
- **GitHub Issue #3 (TASK G1):** [Issue #3 — Curate Vietnamese IT corpus, evaluation queries and draft qrels](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/3)
- **GitHub Issue #5 (TASK G2):** [Issue #5 — Index 114 passages and implement Chroma + Vietnamese BM25 retrieval CLI](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/5)
- **Embedding Model:** `intfloat/multilingual-e5-small` (384 chiều, tiền tố `passage: ` và `query: `, chuẩn hóa L2)
- **Vector Database:** Chroma (`chromadb.PersistentClient`) với metric Cosine Distance
- **Lexical Baseline:** BM25Okapi (`rank_bm25`) kết hợp phân đoạn từ tiếng Việt `pyvi`

---

## 1. Yêu cầu môi trường

- **Hệ điều hành:** Windows 10/11 (win32)
- **Python:** Python 3.12 (khuyến nghị `3.12.x`)
- **Shell:** Windows PowerShell 5.1+ hoặc PowerShell 7+
- **Phần cứng:** CPU thông thường (không yêu cầu GPU)

---

## 2. Cài đặt môi trường (PowerShell)

### Bước 2.1: Tạo môi trường ảo `.venv`

```powershell
python -m venv .venv
```

### Bước 2.2: Kích hoạt môi trường ảo

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Bước 2.3: Cài đặt dependencies

```powershell
pip install -r requirements.txt
```

Các thư viện chính được sử dụng:
- `chromadb==1.5.9`
- `sentence-transformers==6.2.0`
- `numpy==2.5.4`
- `rank-bm25==0.2.2`
- `pyvi==0.1.1`

---

## 3. Quản lý Index Corpus (ChromaDB)

Tập học liệu chính thức gồm 114 đoạn văn bản tiếng Việt thuộc 6 chủ đề CNTT được lưu tại `data/corpus/documents.jsonl`.

### 3.1 Xây dựng hoặc kiểm tra Index

Script tự động kiểm tra token length, mã hóa batch theo lô (batch_size=32) với tiền tố `passage: ` và lưu manifest nhất quán (`chroma_db/index_manifest.json`):

```powershell
# Xây dựng index (nếu index đã tồn tại và khớp SHA256 thì tự động bỏ qua re-encode)
python scripts/build_index.py
```

### 3.2 Kiểm tra tính nhất quán của Index

```powershell
# Kiểm tra manifest, SHA256 dữ liệu và số lượng bản ghi
python scripts/build_index.py --check
```

### 3.3 Tái tạo Index có kiểm soát (Rebuild)

Chỉ xóa và tạo lại collection `seminar_corpus_e5`, không xóa các tệp/thư mục khác:

```powershell
python scripts/build_index.py --rebuild
```

---

## 4. Truy vấn & Đối sánh Truy hồi (Chroma vs BM25)

Công cụ `scripts/compare_search.py` hỗ trợ tìm kiếm đồng thời qua hai phương pháp trên cùng 114 tài liệu.

### 4.1 Tìm kiếm theo nội dung tự do

```powershell
# So sánh cả hai phương pháp (Chroma & BM25) Top-5 mặc định
python scripts/compare_search.py --query "chỉ mục B-Tree trong cơ sở dữ liệu quan hệ"

# Chỉ chạy tìm kiếm ngữ nghĩa Chroma
python scripts/compare_search.py --query "mạng nơ-ron tích chập CNN" --mode chroma

# Chỉ chạy tìm kiếm từ khóa BM25 tiếng Việt
python scripts/compare_search.py --query "thuật toán Dijkstra tìm đường đi ngắn nhất" --mode bm25
```

### 4.2 Tìm kiếm theo Query ID từ tập đánh giá G1

```powershell
# Truy vấn q_001 (Dev split)
python scripts/compare_search.py --query-id q_001 --top-k 5

# Truy vấn q_019 (So sánh L1 vs L2 regularization)
python scripts/compare_search.py --query-id q_019 --top-k 5
```

### 4.3 Lọc theo chủ đề (Metadata Filtering)

Lọc ứng viên hợp lệ trước khi lấy Top-K (áp dụng nhất quán cho cả Chroma `where` và BM25 candidate filtering):

```powershell
python scripts/compare_search.py --query "thuật toán đồ thị và tìm kiếm" --topic "Algorithms" --top-k 5
```

> **Lưu ý về độ đo:**  
> - Chroma sử dụng **Cosine Distance** (khoảng cách [0..2], giá trị càng nhỏ càng tương đồng).  
> - BM25 sử dụng **BM25 Score** (điểm số đối sánh từ vựng không giới hạn, giá trị càng cao càng khớp).  
> - Hai thang đo có bản chất toán học khác nhau, không được gộp chung thành % độ tin cậy.

---

## 5. Cấu trúc thư mục dự án

```text
PatternRecognition_Seminar/
├── data/
│   ├── corpus/
│   │   ├── documents.jsonl      # 114 đoạn tài liệu tiếng Việt (6 chủ đề)
│   │   └── sources.jsonl        # 13 nguồn tài liệu kỹ thuật có thể truy vết
│   ├── eval/
│   │   ├── queries.jsonl        # 32 câu hỏi đánh giá (dev/test split)
│   │   ├── qrels_draft.jsonl    # 86 nhãn tương quan dự thảo {0, 1, 2}
│   │   └── label_review.csv     # Bảng Excel hỗ trợ người kiểm duyệt rà nhãn
│   ├── smoke_documents.json     # 16 tài liệu thử nghiệm G0 (giữ nguyên độc lập)
│   └── README.md                # Đặc tả schema dữ liệu G1
├── src/
│   ├── __init__.py
│   ├── indexer.py               # Quản lý Chroma collection, batch embedding, manifest
│   ├── chroma_search.py         # Module tìm kiếm ngữ nghĩa Chroma
│   └── bm25_search.py           # Module tìm kiếm từ khóa BM25 tiếng Việt (PyVi)
├── scripts/
│   ├── build_index.py           # CLI xây dựng, kiểm tra và rebuild index Chroma
│   ├── compare_search.py        # CLI đối sánh tìm kiếm Chroma vs BM25
│   ├── validate_dataset.py      # Validator kiểm tra toàn vẹn dữ liệu G1
│   ├── smoke_chroma.py          # Smoke test G0
│   └── verify_persistence.py    # Kiểm chứng persistence G0 (2 processes)
├── reports/
│   ├── g0-smoke.md              # Báo cáo thực nghiệm G0
│   ├── g1-dataset.md            # Báo cáo tập dữ liệu G1
│   └── g2-retrieval.md          # Báo cáo kỹ thuật đối sánh truy hồi G2
├── .gitignore                   # Loại trừ .venv, chroma_db, model weights, cache
├── requirements.txt             # Danh sách dependencies đã kiểm tra
└── README.md                    # Hướng dẫn thiết lập và thực thi dự án
```

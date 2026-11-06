# PatternRecognition_Seminar

Dự án Seminar môn Nhận diện Mẫu (Pattern Recognition) - Nghiên cứu Chroma VectorDB, Embedding và Semantic Search cho học liệu tiếng Việt.

- **GitHub Issue #1 (TASK G0):** [Issue #1 — Bootstrap Windows repository + Chroma smoke test](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/1)
- **GitHub Issue #3 (TASK G1):** [Issue #3 — Curate Vietnamese IT corpus, evaluation queries and draft qrels](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/3)
- **GitHub Issue #5 (TASK G2):** [Issue #5 — Index 114 passages and implement Chroma + Vietnamese BM25 retrieval CLI](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/5)
- **GitHub Issue #7 (TASK G3):** [Issue #7 — Reproducible retrieval evaluation and pooled relevance review](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/7)
- **GitHub Issue #9 (TASK G4+G5):** [Issue #9 — Build Streamlit Semantic Search Lab + demo acceptance checks](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/9)
- **GitHub Issue #12 (TASK UI/UX + Audit):** [Issue #12 — UI/UX hotfix + dev-only retrieval relevance audit](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/12)
- **GitHub Issue #14 (TASK Notebook Thực nghiệm):** [Issue #14 — Jupyter experiment notebook aligned with slides](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/14)
- **GitHub Issue #11 (TASK G6):** [Issue #11 — Human-verified labels, official metrics and seminar package](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/11)
- **Embedding Model:** `intfloat/multilingual-e5-small` (384 chiều, tiền tố `passage: ` và `query: `, chuẩn hóa L2)
- **Vector Database:** Chroma (`chromadb.PersistentClient`) với metric Cosine Distance
- **Lexical Baseline:** BM25Okapi (`rank_bm25`) kết hợp phân đoạn từ tiếng Việt `pyvi`
- **Interactive UI:** Streamlit (`app.py`) với cấu hình Dark Theme chuẩn trình chiếu (`.streamlit/config.toml`)
- **Jupyter Notebook:** `notebooks/Chroma_Seminar_Experiments.ipynb` (5 phần thực nghiệm đối sánh theo slide)
- **Official Benchmark Metrics:** Macro-Average Hit@5: 90.91%, MRR@5: 79.09% (Chroma) vs 81.82% (BM25), nDCG@5: 69.28% (Chroma) vs 74.94% (BM25)

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

## 5. Đánh Giá Truy Hồi & Kiểm Duyệt Nhãn (Task G3)

Quy trình đánh giá có thể tái lập trên tập 12 test queries cố định (`data/eval/evaluation_subset.json`).

### 5.1 Tạo Candidate Review Pool

Hợp nhất Top-5 của Chroma, Top-5 của BM25 và các ứng viên draft qrels thành 105 cặp query-doc độc nhất chờ người kiểm duyệt:

```powershell
python scripts/prepare_eval_pool.py
```

### 5.2 Chạy Kiểm tra Metrics Fixture

Kiểm chứng các công thức toán học (`Hit@5`, `MRR@5`, `nDCG@5`) trên fixture mẫu đối chiếu giá trị tính tay:

```powershell
python scripts/run_evaluation.py --fixture-test
```

### 5.3 Chạy Đánh giá Xem trước (Preview Mode)

Xuất kết quả đối sánh Top-1, phân tích định tính câu hỏi unanswerable `q_030` và lưu metadata tái lập tại `results/g3/rankings_preview.json`:

```powershell
python scripts/run_evaluation.py
```

### 5.4 Chạy Đánh giá Chính thức (Verified-Label Gate)

Chế độ nghiêm ngặt yêu cầu 100% nhãn ứng viên trong pool phải được người kiểm duyệt xác nhận (`review_status='verified'`). Nếu còn nhãn pending, hệ thống sẽ từ chối và báo lỗi minh bạch:

```powershell
python scripts/run_evaluation.py --official
```

---

## 6. Khởi Chạy Ứng Dụng Streamlit Demo (Task G4+G5)

Giao diện tương tác **Chroma Semantic Search Lab** hỗ trợ trình diễn trực quan trước lớp học với 3 tabs:
1. **Semantic Search:** Truy vấn tự do, so sánh song song Chroma vs BM25, lọc theo chủ đề.
2. **Collection Explorer:** Xem 114 tài liệu chính thức (`seminar_corpus_e5`, chỉ đọc) và thao tác Thêm/Sửa/Xóa/Tìm kiếm trên collection demo riêng (`seminar_demo_mutable`).
3. **Experiment Results:** Đối soát 12 test queries, phân tích câu hỏi unanswerable, hiển thị trạng thái cổng Verified-Label Gate.

### 6.1 Chuẩn bị và Kiểm tra Chỉ mục

Trước khi khởi chạy ứng dụng, bảo đảm chỉ mục đã được lập:

```powershell
python scripts/build_index.py --check
```

### 6.2 Khởi chạy Ứng dụng Streamlit

```powershell
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ mặc định: `http://localhost:8501`.

### 6.3 Chạy Kiểm chứng Tối thiểu cho Demo (Acceptance Checks)

```powershell
python scripts/verify_demo_acceptance.py
```

---

## 7. Jupyter Notebook Thực Nghiệm Seminar (Task G6 / Issue #14)

Tệp notebook `notebooks/Chroma_Seminar_Experiments.ipynb` được thiết kế ngắn gọn, trực quan, chạy được từ đầu đến cuối và lưu sẵn đầy đủ kết quả bảng/hình, dùng làm **minh chứng thực nghiệm trực tiếp cho 24 slide bài giảng Seminar**:

### 7.1 Cài đặt môi trường chạy Notebook

```powershell
pip install -r requirements-notebook.txt
```

### 7.2 Khởi chạy Jupyter

```powershell
jupyter lab
# hoặc
jupyter notebook
```
Sau đó mở tệp `notebooks/Chroma_Seminar_Experiments.ipynb` và chọn **Kernel -> Restart Kernel and Run All Cells** (hoặc xem trực tiếp kết quả đã được chạy sẵn).

### 7.3 Tái lập hoặc chạy lại tự động từ dòng lệnh

```powershell
python scripts/build_notebook.py
```

### 7.4 Bản đồ liên kết 5 phần thực nghiệm với Slide bài giảng:
1. **Thí nghiệm 1 (Slide 15–16):** Truy vấn vector mẫu $a, b, c, q$ trên mặt phẳng và khảo sát phép lọc siêu dữ liệu (`where={"topic": "Y"}`).
2. **Thí nghiệm 2 (Slide 5–6, 8–9, 17):** Trích xuất vector embedding 384D cho 114 đoạn tài liệu CNTT tiếng Việt và quản lý lưu trữ trong Chroma.
3. **Thí nghiệm 3 (Slide 3–4):** Đối sánh tìm kiếm ngữ nghĩa (Chroma) vs tìm kiếm từ khóa (BM25) trên 3 câu hỏi dev, phân tích case `q_001`.
4. **Thí nghiệm 4 (Slide 11–13, 19):** Đối chiếu chỉ mục đồ thị HNSW của Chroma với phép tìm kiếm cosine chính xác vét cạn (Exact Cosine trong RAM), đo chỉ số ANN Recall@5.
5. **Thí nghiệm 5 (Slide 19–20):** Quy trình đánh giá khoa học, candidate review pool (105 nhãn pending), cơ chế Verified-Label Gate và phân tích định tính truy vấn unanswerable `q_030`.

---

## 8. Cấu trúc thư mục dự án

```text
PatternRecognition_Seminar/
├── .streamlit/
│   └── config.toml              # Cấu hình Dark Theme Streamlit chuẩn máy chiếu
├── app.py                       # Ứng dụng Streamlit: Chroma Semantic Search Lab
├── notebooks/
│   └── Chroma_Seminar_Experiments.ipynb # Jupyter Notebook 5 phần thực nghiệm theo slide
├── data/
│   ├── corpus/
│   │   ├── documents.jsonl      # 114 đoạn tài liệu tiếng Việt (6 chủ đề)
│   │   └── sources.jsonl        # 13 nguồn tài liệu kỹ thuật có thể truy vết
│   ├── eval/
│   │   ├── queries.jsonl        # 32 câu hỏi đánh giá (dev/test split)
│   │   ├── qrels_draft.jsonl    # 86 nhãn tương quan dự thảo {0, 1, 2}
│   │   ├── label_review.csv     # Bảng Excel hỗ trợ người kiểm duyệt rà nhãn G1
│   │   ├── evaluation_subset.json # 12 câu truy vấn test cố định phục vụ benchmark G3
│   │   └── review_pool.csv      # 105 cặp ứng viên pooled (Chroma + BM25 + Draft)
│   ├── smoke_documents.json     # 16 tài liệu thử nghiệm G0 (giữ nguyên độc lập)
│   └── README.md                # Đặc tả schema dữ liệu G1 & G3
├── src/
│   ├── __init__.py
│   ├── indexer.py               # Quản lý Chroma collection, batch embedding, manifest
│   ├── chroma_search.py         # Module tìm kiếm ngữ nghĩa Chroma
│   ├── bm25_search.py           # Module tìm kiếm từ khóa BM25 tiếng Việt (PyVi)
│   ├── evaluation.py            # Tính toán metrics (Hit@5, MRR@5, nDCG@5) & Verified Gate
│   └── ui_helpers.py            # Quản lý cache tài nguyên, collection demo CRUD và gate check
├── scripts/
│   ├── build_index.py           # CLI xây dựng, kiểm tra và rebuild index Chroma
│   ├── compare_search.py        # CLI đối sánh tìm kiếm Chroma vs BM25
│   ├── prepare_eval_pool.py     # CLI tạo candidate review pool 12 test queries
│   ├── run_evaluation.py        # CLI chạy đánh giá truy hồi và kiểm tra cổng gate
│   ├── verify_demo_acceptance.py# Tự động hóa kiểm chứng 4 tiêu chuẩn acceptance demo G4+G5
│   ├── diagnose_q001.py         # Chẩn đoán truy vấn q_001 và thử nghiệm giả thuyết dev
│   ├── build_notebook.py        # Tự động tạo và thực thi tuần tự Jupyter Notebook
│   ├── validate_dataset.py      # Validator kiểm tra toàn vẹn dữ liệu G1
│   ├── smoke_chroma.py          # Smoke test G0
│   └── verify_persistence.py    # Kiểm chứng persistence G0 (2 processes)
├── results/
│   └── g3/
│       ├── rankings_preview.json # Báo cáo kết quả và metadata có thể tái lập
│       └── official_metrics.json # Điểm số benchmark chính thức sau thẩm định nhãn
├── reports/
│   ├── g0-smoke.md              # Báo cáo thực nghiệm G0
│   ├── g1-dataset.md            # Báo cáo tập dữ liệu G1
│   ├── g2-retrieval.md          # Báo cáo kỹ thuật đối sánh truy hồi G2
│   ├── g3-evaluation.md         # Báo cáo quy trình đánh giá và candidate pool G3
│   ├── g5-demo.md               # Báo cáo nghiệm thu ứng dụng Streamlit Lab G4+G5
│   ├── ui-retrieval-audit.md    # Báo cáo cải tiến UI/UX và chẩn đoán truy hồi Issue #12
│   ├── g6-label-review.md       # Báo cáo kết quả thẩm định 105 nhãn tương quan Task G6
│   └── g6-official-evaluation.md# Báo cáo đánh giá thực nghiệm chính thức Task G6
├── .gitignore                   # Loại trừ .venv, chroma_db, model weights, cache
├── requirements.txt             # Danh sách dependencies cốt lõi đã kiểm tra
├── requirements-notebook.txt    # Danh sách dependencies bổ sung cho Jupyter & Matplotlib
└── README.md                    # Hướng dẫn thiết lập và thực thi dự án
```

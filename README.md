# PatternRecognition_Seminar

Dự án Seminar môn Nhận diện Mẫu (Pattern Recognition) - Nghiên cứu Chroma VectorDB, Embedding và Semantic Search cho học liệu tiếng Việt.

- **GitHub Issue #1 (TASK G0):** [Issue #1 — Bootstrap Windows repository + Chroma multilingual semantic-search smoke test](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/1)
- **Embedding Model:** `intfloat/multilingual-e5-small` (384 dimensions)
- **Vector Database:** Chroma (`chromadb.PersistentClient`) với cosine distance metric

---

## 1. Yêu cầu môi trường

- **Hệ điều hành:** Windows 10/11 (hoặc tương đương)
- **Python:** Python 3.12 (khuyến nghị `3.12.x`)
- **Shell:** Windows PowerShell 5.1+ hoặc PowerShell 7+
- **Phần cứng:** CPU thông thường (không yêu cầu GPU)

---

## 2. Cài đặt môi trường (PowerShell)

### Bước 2.1: Tạo môi trường ảo `.venv`

```powershell
# Chạy tại thư mục gốc repository
python -m venv .venv
```

### Bước 2.2: Kích hoạt môi trường ảo

```powershell
# Cho phép thực thi script nếu bị chặn bởi ExecutionPolicy
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Kích hoạt .venv
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

---

## 3. Chạy Smoke Test & Tìm kiếm ngữ nghĩa

Tập dữ liệu mẫu gồm 16 đoạn văn bản tiếng Việt thuộc 6 chủ đề kỹ thuật được lưu tại `data/smoke_documents.json`.

### 3.1 Chạy Ingestion & Demo 3 truy vấn mẫu

Chạy toàn bộ quy trình: nạp model E5, tính vector embeddings (tiền tố `passage: ` và chuẩn hóa L2), upsert vào Chroma PersistentClient (idempotent), và thực thi 3 truy vấn thử nghiệm:

```powershell
python scripts/smoke_chroma.py --ingest --queries
```

### 3.2 Tùy chọn truy vấn đơn lẻ

```powershell
python scripts/smoke_chroma.py --query "mạng nơ-ron tích chập CNN trích xuất đặc trưng hình ảnh"
```

### 3.3 Kiểm tra số lượng bản ghi hiện có

```powershell
python scripts/smoke_chroma.py --count
```

---

## 4. Kiểm chứng kỹ thuật (Minimal Verification)

Script `scripts/verify_persistence.py` tự động hóa 3 bài kiểm tra kỹ thuật cốt lõi:
1. **End-to-End:** Load model `intfloat/multilingual-e5-small`, kiểm tra vector dimension = 384, upsert và truy vấn Top-3.
2. **Persistence qua hai tiến trình độc lập:** Tiến trình A nạp dữ liệu vào thư mục lưu trữ rồi thoát hoàn toàn. Tiến trình B độc lập khởi tạo sau đó, mở cơ sở dữ liệu trên đĩa, xác nhận 16 bản ghi và thực thi truy vấn thành công.
3. **Idempotency:** Nạp lại dữ liệu cũ, xác nhận số lượng ID không bị trùng lặp hay tăng sai lệch.

```powershell
python scripts/verify_persistence.py
```

---

## 5. Cấu trúc thư mục

```text
PatternRecognition_Seminar/
├── data/
│   └── smoke_documents.json     # 16 tài liệu tiếng Việt mẫu (6 chủ đề)
├── scripts/
│   ├── smoke_chroma.py          # Script nạp vector và truy vấn Chroma PersistentClient
│   └── verify_persistence.py    # Kiểm chứng persistence qua 2 tiến trình độc lập
├── reports/
│   └── g0-smoke.md              # Báo cáo kỹ thuật chi tiết và bằng chứng thực nghiệm
├── .gitignore                   # Loại trừ .venv, DB files, model cache, weights
├── requirements.txt             # Danh sách dependencies đã kiểm tra
└── README.md                    # Hướng dẫn thiết lập và thực thi dự án
```

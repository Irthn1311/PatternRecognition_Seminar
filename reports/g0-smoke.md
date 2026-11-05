# Báo Cáo Kỹ Thuật G0: Chroma Feasibility & Bootstrap Smoke Test

- **Dự án:** PatternRecognition_Seminar
- **Nhiệm vụ:** TASK G0 — Repository Bootstrap & Chroma Technical Feasibility
- **GitHub Issue:** [#1 — Bootstrap Windows repository + Chroma multilingual semantic-search smoke test](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/1)
- **Thời gian thực hiện:** Tháng 11/2026
- **Môi trường:** Windows 11 Home / Windows PowerShell 5.1

---

## 1. Thông số môi trường và phiên bản thư viện

| Thành phần | Thông số / Phiên bản | Ghi chú |
| :--- | :--- | :--- |
| **Hệ điều hành** | Windows 11 (win32) | Chạy trên môi trường local |
| **Python** | Python 3.12.16 | Sử dụng môi trường ảo `.venv` |
| **ChromaDB** | 1.5.9 | `chromadb.PersistentClient` |
| **Sentence Transformers** | 6.2.0 | Tương thích PyTorch 2.15.0+cpu |
| **PyTorch (Backend)** | 2.15.0+cpu | Thực thi hoàn toàn bằng CPU, không cần GPU |
| **NumPy** | 2.5.4 | Xử lý vector mảng |
| **Embedding Model** | `intfloat/multilingual-e5-small` | Hugging Face Transformers |
| **Vector Dimension** | 384 dimensions | Đo đạc thực tế: `(1, 384)` |
| **Normalization** | `normalize_embeddings=True` | Chuẩn hóa L2 trước khi đưa vào Chroma |
| **Distance Metric** | Cosine Distance | Cấu hình collection: `{"hnsw:space": "cosine"}` |
| **Số lượng bản ghi** | 16 documents | Tự biên soạn phủ 6 chủ đề kỹ thuật |

---

## 2. Quy tắc tiền tố E5 (Prefix Rules)

Theo quy chuẩn của mô hình Multilingual-E5:
- **Tài liệu nạp (Passages):** Phải có tiền tố `passage: ` (ví dụ: `passage: Hồi quy tuyến tính...`)
- **Câu truy vấn (Queries):** Phải có tiền tố `query: ` (ví dụ: `query: mạng nơ-ron tích chập CNN...`)

Quy tắc này đã được triển khai nhất quán và tự động trong lớp `ChromaEmbeddingManager` tại `scripts/smoke_chroma.py`.

---

## 3. Các lệnh đã thực thi

```powershell
# 1. Khởi tạo Git và cấu hình remote
git init -b main
git remote add origin https://github.com/Irthn1311/PatternRecognition_Seminar.git

# 2. Khởi tạo môi trường ảo và cài đặt dependencies
python -m venv .venv
.\.venv\Scripts\pip.exe install chromadb sentence-transformers numpy

# 3. Chạy ingestion và demo 3 truy vấn tiếng Việt
python scripts/smoke_chroma.py --ingest --queries

# 4. Kiểm chứng Persistence (hai tiến trình độc lập) và Idempotency
python scripts/verify_persistence.py
```

---

## 4. Kết quả thực nghiệm 3 truy vấn tiếng Việt

### Truy vấn 1: Khác từ vựng với tài liệu gốc (Lexically varied)
- **Câu hỏi:** `làm sao để tránh việc mô hình học vẹt và ghi nhớ quá mức dữ liệu huấn luyện`
- **Kết quả Top-3:**
  - **Rank 1:** `doc_ml_03` (Cosine Distance: `0.1036`)  
    *Nội dung:* Overfitting xảy ra khi mô hình học máy ghi nhớ chi tiết và nhiễu của tập huấn luyện, làm suy giảm khả năng tổng quát hóa trên tập dữ liệu kiểm thử thực tế.  
    *Chủ đề:* Machine Learning
  - **Rank 2:** `doc_cv_03` (Cosine Distance: `0.1459`)  
    *Nội dung:* Tăng cường dữ liệu hình ảnh (Image Augmentation) bằng phép xoay, lật, thay đổi độ sáng giúp làm phong phú tập huấn luyện...  
    *Chủ đề:* Computer Vision
  - **Rank 3:** `doc_dl_03` (Cosine Distance: `0.1626`)  
    *Nội dung:* Hàm kích hoạt phi tuyến tính như ReLU hoặc GELU giúp mạng học sâu...  
    *Chủ đề:* Deep Learning
- **Nhận xét:** Truy vấn dùng cụm từ "học vẹt", "ghi nhớ quá mức dữ liệu huấn luyện" nhưng mô hình truy xuất chính xác tài liệu định nghĩa hiện tượng "Overfitting" ở vị trí số 1 với khoảng cách cosine chỉ 0.1036.

---

### Truy vấn 2: Chứa thuật ngữ kỹ thuật trực tiếp (Direct technical terms)
- **Câu hỏi:** `mạng nơ-ron tích chập CNN trích xuất đặc trưng không gian từ hình ảnh`
- **Kết quả Top-3:**
  - **Rank 1:** `doc_dl_01` (Cosine Distance: `0.0580`)  
    *Nội dung:* Mạng nơ-ron tích chập (Convolutional Neural Network - CNN) sử dụng các tầng convolution và pooling để trích xuất các đặc trưng không gian đa mức từ tín hiệu hình ảnh.  
    *Chủ đề:* Deep Learning
  - **Rank 2:** `doc_cv_01` (Cosine Distance: `0.1682`)  
    *Nội dung:* Phát hiện đối tượng (Object Detection) kết hợp bài toán định vị bounding box và phân loại nhãn thực thể trong ảnh...  
    *Chủ đề:* Computer Vision
  - **Rank 3:** `doc_dl_03` (Cosine Distance: `0.1779`)  
    *Nội dung:* Hàm kích hoạt phi tuyến tính như ReLU hoặc GELU giúp mạng học sâu...  
    *Chủ đề:* Deep Learning
- **Nhận xét:** Khoảng cách cosine rất nhỏ (0.0580), tài liệu CNN khớp hoàn hảo nội dung ngữ nghĩa và từ khóa chuyên ngành.

---

### Truy vấn 3: Chủ đề mạng máy tính (Alternative domain)
- **Câu hỏi:** `hệ thống phân giải tên miền và định tuyến địa chỉ IP trong mạng máy tính`
- **Kết quả Top-3:**
  - **Rank 1:** `doc_net_02` (Cosine Distance: `0.1020`)  
    *Nội dung:* Hệ thống phân giải tên miền DNS (Domain Name System) chuyển đổi tên miền dạng chữ dễ nhớ mà con người sử dụng thành địa chỉ IP số để các thiết bị mạng định tuyến gói tin.  
    *Chủ đề:* Computer Networks
  - **Rank 2:** `doc_db_01` (Cosine Distance: `0.1721`)  
    *Nội dung:* Chỉ mục B-Tree trong cơ sở dữ liệu quan hệ...  
    *Chủ đề:* Database
  - **Rank 3:** `doc_dl_03` (Cosine Distance: `0.1880`)  
    *Nội dung:* Hàm kích hoạt phi tuyến tính như ReLU hoặc GELU...  
    *Chủ đề:* Deep Learning
- **Nhận xét:** Tài liệu DNS và định tuyến IP được xếp hạng 1 chính xác với độ tương đồng vượt trội so với các chủ đề khác.

---

## 5. Kết quả kiểm chứng kỹ thuật (Acceptance Verification)

| Hạng mục kiểm tra | Tiêu chuẩn đánh giá | Kết quả thực tế | Trạng thái |
| :--- | :--- | :--- | :--- |
| **Check 1: End-to-End** | Load model E5 CPU, embedding shape 384, upsert Chroma thành công, query trả về Top-3 đầy đủ ID, metadata, distance. | Hoàn thành, 16 bản ghi được nhúng vector 384 chiều, tìm kiếm trả về Top-3 khớp đúng nội dung. | **PASS** |
| **Check 2: Persistence (2 processes)** | Tiến trình A ghi dữ liệu và thoát. Tiến trình B độc lập khởi tạo mới, đọc lại dữ liệu trên đĩa tại cùng đường dẫn, kiểm tra đúng 16 bản ghi và query thành công. | Script `scripts/verify_persistence.py` chạy qua 2 subprocesses độc lập sử dụng thư mục tạm cô lập (`tempfile.TemporaryDirectory`). Tiến trình B đọc đúng 16 bản ghi và tìm kiếm ra `doc_algo_01` (Dijkstra) với distance `0.0613`. Tự động dọn dẹp sau khi kiểm chứng hoàn tất. | **PASS** |
| **Check 3: Idempotency** | Chạy lại ingestion dữ liệu cũ trên collection hiện có, tổng số bản ghi không tăng hoặc trùng lặp ID. | Chạy lại lệnh ingest: số bản ghi trước và sau đều là 16. Cơ chế `upsert` với stable IDs hoạt động chuẩn xác. | **PASS** |

---

## 6. Các giới hạn và vấn đề còn tồn tại

1. **Cảnh báo môi trường Windows:**
   - Hugging Face Hub đưa ra cảnh báo: `UserWarning: huggingface_hub cache-system uses symlinks by default...` do Windows không bật Developer Mode. Cảnh báo này chỉ liên quan đến symlink cache của HF, không ảnh hưởng đến việc tải và suy luận mô hình.
   - Khi không truyền `HF_TOKEN`, hub hiển thị cảnh báo `unauthenticated requests`, nhưng tốc độ tải và quyền truy cập model công khai `multilingual-e5-small` hoàn toàn bình thường.
2. **Hiệu năng và phần cứng:**
   - Toàn bộ pipeline chạy trên CPU nhằm đảm bảo tính tương thích và dễ tái tạo trên mọi máy tính. Thời gian nhúng 16 tài liệu chỉ mất khoảng 1.5 giây. Đối với tập dữ liệu lớn hơn trong tương lai, cần bổ sung batching hoặc GPU / ONNX runtime.
3. **Phạm vi dữ liệu:**
   - 16 đoạn tài liệu hiện tại là tập dữ liệu mẫu (smoke dataset) do nhóm tự biên soạn nhằm kiểm tra tính khả thi kỹ thuật. Dự án chưa có tập ground truth / qrels chính thức, do đó chưa đo đạc các chỉ số đánh giá IR (như MRR, MAP, NDCG) hay so sánh với BM25 (thuộc các giai đoạn tiếp theo).

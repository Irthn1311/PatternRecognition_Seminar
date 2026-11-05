# Báo Cáo Kỹ Thuật G4+G5: Nghiệm Thu Ứng Dụng Streamlit Chroma Semantic Search Lab

- **Dự án:** PatternRecognition_Seminar
- **Nhiệm vụ:** TASK G4+G5 — Streamlit Chroma Semantic Search Lab & Demo Acceptance Checks
- **GitHub Issue:** [#9 — G4+G5: Build Streamlit Semantic Search Lab + essential demo acceptance checks](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/9)
- **Thời gian thực hiện:** Tháng 11/2026
- **Môi trường:** Windows 11 Home / Python 3.12.16 / Streamlit 1.66.1

---

## 1. Tổng Quan Ứng Dụng Đã Triển Khai

Ứng dụng web tương tác **Chroma Semantic Search Lab** (`app.py`) đã được xây dựng hoàn chỉnh bằng Streamlit, phục vụ trình diễn trước lớp học cho buổi Seminar môn Nhận diện Mẫu:
- **Tên ứng dụng:** Chroma Semantic Search Lab
- **Mô hình Embedding:** `intfloat/multilingual-e5-small` (384 chiều, tiền tố `passage: ` và `query: `, chuẩn hóa L2).
- **Cơ sở dữ liệu Vector:** Chroma (`chromadb.PersistentClient`) với độ đo Cosine Distance.
- **Baseline Từ vựng:** BM25Okapi (`rank_bm25`) kết hợp phân đoạn từ ghép tiếng Việt `pyvi`.
- **Hiệu năng & Caching:** Tối ưu hóa bằng `@st.cache_resource` cho mô hình embedding và bộ chỉ mục BM25, đảm bảo không nạp lại trọng số khi người dùng tương tác trên giao diện.

Giao diện được thiết kế trực quan, độ tương phản cao, bao gồm **3 tabs chức năng chính**:

---

## 2. Chi Tiết 3 Tabs Giao Diện

### Tab 1: 🔍 Semantic Search (Tìm Kiếm Ngữ Nghĩa & BM25)
- **Hộp nhập truy vấn:** Cho phép nhập tự do văn bản tiếng Việt hoặc chọn nhanh các câu hỏi gợi ý để demo.
- **Lựa chọn chế độ:**
  - `Both (Chroma vs BM25)`: Hiển thị hai cột song song để đối sánh trực tiếp.
  - `Chroma Semantic Search`: Tìm kiếm thuần ngữ nghĩa.
  - `Vietnamese BM25`: Tìm kiếm thuần từ khóa.
- **Thanh trượt Top-K:** Điều chỉnh linh hoạt số lượng kết quả từ 1 đến 10 (mặc định: 5).
- **Bộ lọc chủ đề (Metadata Filtering):** Lọc ứng viên trước khi xếp hạng (All Topics, Machine Learning, Deep Learning, Computer Vision, Database, Algorithms, Computer Networks / Security).
- **Phân định thang đo rõ ràng:**
  - Chroma: **Cosine Distance** (khoảng cách [0..2], càng nhỏ càng tương đồng ngữ nghĩa).
  - BM25: **BM25 Score** (điểm số đối sánh từ vựng không giới hạn, càng lớn càng khớp).
  - Có khung cảnh báo minh bạch: không quy đổi hay gộp chung hai thang đo thành % độ tin cậy.

### Tab 2: 🗄️ Collection Explorer (Khám Phá & Thao Tác Trực Tiếp)
Gồm 2 phân hệ tách biệt hoàn toàn:
1. **Kho học liệu chính thức (`seminar_corpus_e5` — Chỉ đọc):**
   - Hiển thị thông số: 114 đoạn tài liệu, 384 dimensions, Cosine metric, trạng thái bất biến (Read-Only).
   - Bộ lọc xem theo chủ đề và ô tìm kiếm nhanh văn bản.
   - Bảng dữ liệu tương tác xem toàn bộ 114 passages.
2. **Collection thao tác trực tiếp (`seminar_demo_mutable` — CRUD Demo):**
   - Collection độc lập dành riêng cho các thao tác thử nghiệm trên lớp, **tuyệt đối không làm thay đổi kho 114 tài liệu chính thức**.
   - **Thêm (Add):** Nhập ID, tiêu đề, chủ đề, nội dung $\rightarrow$ tự động tính embedding E5 (`passage: `) và nạp vào Chroma.
   - **Xem & Tìm kiếm (Read & Search):** Duyệt danh sách các tài liệu demo và thử nghiệm tìm kiếm vector trực tiếp trên demo collection.
   - **Sửa (Update):** Chọn ID $\rightarrow$ sửa tiêu đề, chủ đề, nội dung $\rightarrow$ tự động tái tính embedding E5 và cập nhật Chroma.
   - **Xóa (Delete):** Chọn ID $\rightarrow$ xác nhận hộp kiểm an toàn $\rightarrow$ xóa khỏi collection.

### Tab 3: 📊 Experiment Results (Kết Quả Thực Nghiệm & Cổng Kiểm Duyệt)
- **Thống kê tổng quan:** 12 test queries, 105 cặp ứng viên pooled, tỷ lệ đồng thuận Top-1 đạt 58.3%.
- **Trạng thái cổng Verified-Label Gate:**
  - Khi 105 nhãn đang ở trạng thái `pending`, giao diện hiển thị cảnh báo rõ ràng:  
    `⏳ Official benchmark is pending human verification (105/105 candidate labels pending).`
  - Từ chối xuất bản bảng điểm chính thức chưa được con người chứng nhận.
- **Đối soát chi tiết từng câu hỏi:** Dropdown chọn xem bất kỳ câu hỏi nào trong 12 câu hỏi đánh giá $\rightarrow$ hiển thị phân loại, trạng thái đồng thuận, và bảng Top-5 song song giữa Chroma và BM25.
- **Phân tích định tính câu hỏi Unanswerable (`q_030`):**
  - Giải thích bản chất thuật toán nearest-neighbor không tự động từ chối (abstain).
  - Cả hai hệ thống đều trả về láng giềng gần nhất (`algo_014` và `ml_001`), làm rõ lý do tại sao câu hỏi này được phân tích độc lập và loại khỏi mẫu số tính điểm trung bình.

---

## 3. Kết Quả 4 Bài Kiểm Chứng Nghiệm Thu (Acceptance Checks)

Script tự động `scripts/verify_demo_acceptance.py` đã thực thi kiểm chứng toàn bộ 4 tiêu chí chấp nhận:

```powershell
python scripts/verify_demo_acceptance.py
```

### Log thực tế:
```text
================================================================================
RUNNING G4+G5 DEMO ACCEPTANCE CHECKS
================================================================================

[Check 1: Dual Retrieval Search]
  Chroma Top-1: [cv_001] Tầng Tích chập Convolutional Layer và Bộ lọc Kernel (Cosine Dist: 0.1224)
  BM25 Top-1:   [cv_010] Phát hiện Đối tượng Hai giai đoạn Faster R-CNN (BM25 Score: 18.4074)
Check 1 PASS: Both retrievers executed successfully on real data.

[Check 2: Metadata Topic Filtering]
  Chroma returned 4 results, all topic='Database'.
  BM25 returned 4 results, all topic='Database'.
Check 2 PASS: Metadata pre-filtering strictly enforced on both retrievers.

[Check 3: Isolated Mutable Collection CRUD]
  Official collection count before CRUD: 114 (seminar_corpus_e5)
  [CRUD Add] Added 'demo_test_crud_01', demo collection count: 1
  [CRUD Search] Found 'demo_test_crud_01' with cosine distance 0.0846
  [CRUD Update] Updated 'demo_test_crud_01', title is now: 'Học Tăng Cường Nâng Cao'
  [CRUD Delete] Deleted 'demo_test_crud_01', demo collection count: 0
  Official collection count after CRUD: 114 (unchanged)
Check 3 PASS: CRUD operations isolated to demo collection; official corpus perfectly preserved.

[Check 4: Experiment Results & Gate Status]
  Loaded preview artifact with 12 test queries.
  Official benchmark eligibility: False
  Gate reason: "Official benchmark is pending human verification (105/105 candidate labels pending)."
Check 4 PASS: Experiment tab displays preview and correctly locks official benchmark.

================================================================================
ALL 4 DEMO ACCEPTANCE CHECKS PASSED SUCCESSFULLY (PASS)
================================================================================
```

---

## 4. Ba Kịch Bản Trình Diễn Đề Xuất Trước Lớp Học

Dành cho người trình bày khi báo cáo Seminar:

### Kịch bản 1: Đối sánh Ngữ nghĩa (Dense) vs Từ khóa (Sparse)
- **Thao tác:** Tại Tab **Semantic Search**, chọn câu hỏi gợi ý `q_002`:  
  *"cách giải quyết khi mô hình học máy thuộc lòng dữ liệu huấn luyện và dự đoán kém trên thực tế"*
- **Hiện tượng quan sát:**
  - Người dùng không hề dùng từ tiếng Anh *"Overfitting"* hay *"điều chuẩn"*.
  - **Chroma:** Nhận diện xuất sắc khái niệm "học thuộc lòng dữ liệu" và đưa tài liệu `ml_015` (Hiện tượng Quá khớp và các Biện pháp khắc phục) lên Hạng 1 (Cosine Dist: `0.1197`).
  - **BM25:** Dựa vào các từ khóa "học máy", "mô hình", "dữ liệu", đưa `ml_015` lên Hạng 1 nhờ tần suất từ vựng, nhưng các thứ hạng sau bị phân tán sang các chủ đề khác có từ "mô hình".
- **Điểm nhấn giảng dạy:** Thể hiện rõ ưu thế của vector embedding trong việc nắm bắt ý định ngữ nghĩa tự nhiên của người dùng.

### Kịch bản 2: Lọc Siêu Dữ Liệu theo Ngữ Cảnh (Metadata Filtering)
- **Thao tác:** Nhập câu hỏi *"tìm kiếm dữ liệu và thuật toán cây phân cấp"*, chọn Lọc theo chủ đề: `Database`.
- **Hiện tượng quan sát:**
  - Câu hỏi vừa có từ "thuật toán" (thuộc Algorithms), vừa có từ "dữ liệu" (thuộc Database).
  - Nhờ cơ chế pre-filtering, 100% kết quả của cả Chroma và BM25 đều được giới hạn trong chủ đề `Database` (trả về các bài về *Chỉ mục B-Tree `db_003`* và *Chỉ mục Vector HNSW `db_016`*).
- **Điểm nhấn giảng dạy:** Giải thích cách kết hợp giữa tìm kiếm tương đồng vector và bộ lọc thuộc tính cấu trúc trong các hệ thống Vector Database thực tế.

### Kịch bản 3: Thao Tác Trực Tiếp trên Vector Database (Live CRUD)
- **Thao tác:** Chuyển sang Tab **Collection Explorer** $\rightarrow$ chọn phân hệ `Demo Collection (seminar_demo_mutable)`.
- **Các bước demo:**
  1. Thêm một tài liệu mới (ví dụ ID: `demo_rl_01`, Tiêu đề: *Học tăng cường*, Nội dung: *Học tăng cường tối ưu hóa hành động thông qua cơ chế phần thưởng...*).
  2. Xem số lượng bản ghi tăng từ 0 lên 1 và kiểm tra kết quả tìm kiếm vector ngay lập tức.
  3. Cập nhật nội dung và chứng kiến vector được tái tính toán tức thì.
  4. Xóa tài liệu và xác nhận số bản ghi quay về 0.
  5. Đối chiếu lại kho chính thức `seminar_corpus_e5`: vẫn vẹn nguyên đúng 114 tài liệu.
- **Điểm nhấn giảng dạy:** Minh họa trực quan vòng đời dữ liệu (Data Lifecycle) trong Vector Database mà không lo ngại làm hỏng kho dữ liệu chuẩn.

---

## 5. Giới Hạn Còn Tồn Tại
- Toàn bộ 105 cặp ứng viên trong `data/eval/review_pool.csv` hiện vẫn ở trạng thái `pending` (chưa thẩm định thủ công).
- Ứng dụng tuân thủ nghiêm ngặt nguyên tắc khoa học: **chưa hiển thị bảng điểm benchmark chính thức** trên giao diện khi nhãn chưa được con người phê duyệt.

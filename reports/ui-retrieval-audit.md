# Báo Cáo Kỹ Thuật: Cải Tiến Giao Diện UI/UX & Chẩn Đoán Chất Lượng Truy Hồi

- **Dự án:** PatternRecognition_Seminar
- **Nhiệm vụ:** TASK — UI/UX Redesign + Retrieval Relevance Audit
- **GitHub Issue:** [#12 — UI/UX hotfix + dev-only retrieval relevance audit for Chroma Seminar demo](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/12)
- **Thời gian thực hiện:** Tháng 11/2026
- **Môi trường:** Windows 11 Home / Python 3.12.16 / Streamlit 1.66.1

---

## 1. Cải Tiến Giao Diện Người Dùng (UI/UX Redesign)

### 1.1 Khắc phục lỗi tương phản màu sắc & Cấu hình Theme nhất quán
- **Vấn đề ban đầu:** Tiêu đề màu xanh đậm (`#1E3A8A`) trên nền đen gây khó đọc; các thẻ kết quả màu xanh/xanh dương nhạt với chữ trắng hoặc chữ xám mờ trên nền sáng gây chói và mất chữ khi trình chiếu trên máy chiếu lớp học.
- **Giải pháp triển khai:**
  - Thiết lập tệp cấu hình chuẩn `.streamlit/config.toml` với hệ màu Dark Theme chuyên nghiệp:
    - `primaryColor = "#38BDF8"` (Sky Blue nổi bật làm màu nhấn).
    - `backgroundColor = "#0F172A"` (Slate-900 / Navy charcoal tối dễ chịu).
    - `secondaryBackgroundColor = "#1E293B"` (Slate-800 cho bề mặt sidebar và container).
    - `textColor = "#F8FAFC"` (Slate-50 trắng sáng đạt chuẩn tương phản cao).
    - Giữ nguyên cơ chế bảo vệ CORS mặc định an toàn của Streamlit (loại bỏ `enableCORS = false`).
  - Tái cấu trúc các thẻ kết quả: Sử dụng `st.container(border=True)` kết hợp các badge có màu tương phản tường minh (`badge-rank-chroma`, `badge-rank-bm25`, `badge-topic-tag`).
  - Tiêu đề tài liệu (`result-title-text`) được định dạng chữ lớn, in đậm và sáng rõ (`#F8FAFC`), nổi bật hơn hẳn so với ID và chủ đề.
  - **Bảo vệ chống chèn HTML chưa escape (FIX 1):** Toàn bộ các giá trị text động (`title`, `topic`, `preview`) đều được xử lý qua hàm `safe_html()` (`html.escape(..., quote=True)`) trước khi đưa vào các thẻ hiển thị. Các ký tự đặc biệt `<`, `>`, `&`, `"`, `'` được escape thành thực thể HTML an toàn, loại trừ triệt để nguy cơ lỗi cú pháp hoặc XSS.

### 1.2 Tối ưu bố cục trình diễn lớp học (Classroom Presentation Layout)
- **Sidebar gọn gàng:** Tinh gọn các thông số kỹ thuật (Model ID, số chiều 384, tokenizer, metric) vào một `st.expander("🛠️ Chi tiết cấu hình kỹ thuật")`, giúp thanh điều hướng thoáng đãng và tập trung vào trạng thái chỉ mục.
- **Trạng thái khởi tạo thân thiện (Empty State):** Khi chưa thực hiện tìm kiếm, thay vì để khoảng trống lớn vô nghĩa, giao diện hiển thị khung hướng dẫn chào mừng trực quan với các bước thao tác cụ thể.
- **Gợi ý câu hỏi mẫu:** Bố trí thanh chọn câu hỏi mẫu ngay phía trên ô tìm kiếm để người thuyết trình có thể chọn nhanh mà không cần gõ bàn phím trước lớp.

### 1.3 Khắc phục cơ chế Search Interaction
- **Vấn đề ban đầu:** Điều kiện `if search_btn or query_input.strip()` khiến Streamlit tự động kích hoạt tính toán vector embedding E5 mỗi khi người dùng thay đổi bất kỳ widget nào (thanh trượt Top-K, bộ lọc chủ đề, hoặc khi vừa chọn câu hỏi mẫu).
- **Giải pháp triển khai:**
  - Chuyển toàn bộ cụm điều khiển tìm kiếm vào `st.form("search_form", clear_on_submit=False)` kết hợp quản lý trạng thái phiên qua `st.session_state`.
  - Chọn câu hỏi mẫu chỉ điền văn bản vào ô tìm kiếm, **chỉ thực thi suy luận khi người dùng nhấn nút '🚀 Tìm kiếm'**.
  - Việc thay đổi thanh trượt Top-K, đổi bộ lọc chủ đề hoặc đổi câu hỏi mẫu không còn tự động kích hoạt tính toán vector E5 lại.
- **Bảo toàn ngữ cảnh phiên (`st.session_state`):**
  - Lưu giữ câu hỏi, chế độ, Top-K, chủ đề và danh sách kết quả của lần submit gần nhất.
  - Khi người dùng tương tác với các thành phần khác, kết quả tìm kiếm không bị biến mất hay bị gán nhầm cho cấu hình chưa submit.
- **Xử lý lỗi rõ ràng:**
  - Nếu Chroma index bị thiếu hoặc không đồng bộ, giao diện hiển thị thông báo lỗi `st.error` rõ ràng kèm mã lệnh sửa chữa; ở chế độ `Both`, BM25 vẫn tiếp tục hiển thị kết quả độc lập thay vì nhầm lẫn thành "Không có kết quả".

### 1.4 Khắc phục Official Explorer & Experiment Results
- **Official Corpus Explorer:** Chuyển từ việc đọc tệp `documents.jsonl` sang đọc trực tiếp dữ liệu và metadata từ collection Chroma thật `seminar_corpus_e5` qua `col.get(include=['documents', 'metadatas'])`, phản ánh đúng `col.count() = 114`.
- **An toàn thao tác CRUD:** Script nghiệm thu chuyển sang thực thi trên database tạm thời cô lập (`tempfile.TemporaryDirectory`), bảo vệ 100% dữ liệu trong collection thật của người dùng. Hàm `add_mutable_document()` từ chối nếu ID đã tồn tại, và `update_mutable_document()` yêu cầu ID phải tồn tại.
- **Experiment Results:** Tự động đọc số lượng nhãn pending/verified động từ `data/eval/review_pool.csv` (không hardcode `0/105`). Tích hợp cơ chế kiểm tra đồng thời cả `corpus_sha256` và `review_pool_sha256`; nếu file nhãn bị sửa sau khi tính toán, hệ thống sẽ tự động khóa để chống kết quả lỗi thời.

---

## 2. Chẩn Đoán Chất Lượng Truy Hồi Trên Truy Vấn `q_001`

### 2.1 Truy vấn thực tế qua Chroma HNSW Search và BM25 Retrieval (Top-30 Inspection)
- **Câu hỏi (`q_001`):** *"làm thế nào để máy tính tự động tìm ra các vật thể và vị trí của chúng trong một bức ảnh?"*
- **Thứ hạng thực tế của các tài liệu mục tiêu:**

| Tài liệu | Tên bài & Lĩnh vực | Hạng Chroma (HNSW Vector Search) | Cosine Distance | Hạng BM25 (PyVi Lexical Search) | BM25 Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `cv_009` | Phân loại Ảnh (Image Classification) | **1** | **0.1452** | **1** | **13.1526** |
| `cv_014` | Phân đoạn Cá thể (Instance Segmentation & Mask R-CNN) | **2** | **0.1504** | **18** | **7.7063** |
| `cv_011` | Phát hiện Đối tượng Một giai đoạn (YOLO) | **11** | **0.1767** | **29** | **6.8481** |
| `cv_010` | Phát hiện Đối tượng Hai giai đoạn (Faster R-CNN) | **23** | **0.1874** | **17** | **7.7637** |

### 2.2 Phân biệt giữa Chroma HNSW Search và E5 Exact Cosine trong RAM (FIX 2)
Trong script chẩn đoán `scripts/diagnose_q001.py`, nhóm nghiên cứu phân định tường minh giữa:
1. **Truy vấn Chroma thật (HNSW ANN Search):** Gọi qua `ChromaRetriever.search(q_001, top_k=30)` truy xuất trên cấu trúc đồ thị HNSW phân cấp lưu trên đĩa.
2. **Truy vấn Cosine chính xác lý thuyết trong RAM (Exact In-Memory Cosine):** Tính toán tích vô hướng vét cạn $1 - \mathbf{u} \cdot \mathbf{v}$ trực tiếp trên mảng vector NumPy trong bộ nhớ.

**Kết quả đối chiếu:**
- Trên tập học liệu 114 tài liệu, Top-5 của Chroma HNSW (`cv_009`, `cv_014`, `cv_018`, `cv_008`, `ml_009`) **trùng khớp 100% về cả thứ tự lẫn khoảng cách** so với tính toán cosine chính xác vét cạn trong RAM.
- Điều này chứng minh rằng: **Chroma HNSW không hề gặp hiện tượng sai số xấp xỉ hay thất lạc láng giềng gần nhất** trên quy mô này. Việc `cv_009` đứng trên `cv_010`/`cv_011` hoàn toàn là do hình học không gian vector của mô hình `intfloat/multilingual-e5-small`.

### 2.3 Phân tích nguyên nhân ngữ nghĩa & từ vựng
1. **Độ bão hòa từ vựng (BM25):** Đoạn văn `cv_009` chứa mật độ từ trùng khớp cao nhất với câu hỏi: *"bức ảnh"*, *"ảnh"*, *"thị giác máy tính"*, *"máy tính"*, *"một"*, *"các"*, *"của"*, *"để"*, *"và"* (8/17 tokens trùng khớp). Trong khi đó, đoạn `cv_010` thậm chí không chứa từ *"ảnh"* hay *"bức ảnh"* trong phần thân văn bản.
2. **Khoảng cách phong cách diễn đạt (Semantic Gap):** Câu hỏi `q_001` dùng văn phong tự nhiên đời thường của người học (*"tìm ra các vật thể và vị trí của chúng trong một bức ảnh"*), trong khi `cv_010` và `cv_011` sử dụng hệ thống thuật ngữ học thuật chuyên sâu (*"phát hiện đối tượng"*, *"vùng ứng viên"*, *"mạng đề xuất vùng RPN"*, *"hộp neo anchor boxes"*, *"tọa độ hộp bao bounding box"*).
3. **Lý do `cv_014` đứng Hạng 2 ở Chroma:** Đoạn `cv_014` có chứa cụm từ *"định vị chính xác vị trí"*, *"phát hiện đối tượng"* và *"xuất hiện trong ảnh"*. Nhờ đó, mô hình E5 nhận diện sự tương đồng ngữ nghĩa về "vị trí vật thể trong ảnh" và đưa `cv_014` lên vị trí Hạng 2 với khoảng cách cosine `0.1504` (rất sát Hạng 1 `0.1452`).

---

## 3. Thử Nghiệm Các Biến Thể Cải Thiện Trên Tập Dev Queries

Để kiểm chứng xem việc bổ sung tiêu đề có cải thiện khả năng phân biệt hay không, nhóm nghiên cứu đã thiết lập thử nghiệm đối chứng trên 3 câu hỏi thuộc tập dev (`q_001`, `q_005`, `q_019`):
- **Biến thể A:** BM25 lập chỉ mục bổ sung Tiêu đề (`title + ". " + text`).
- **Biến thể B:** E5 Exact Cosine mã hóa bổ sung Tiêu đề (`passage: title + ". " + text`).

### Kết quả đo đạc thực nghiệm:

| Query | Phương pháp | Baseline (Chỉ thân văn bản) Top-1 & Top-2 | Biến thể (Tiêu đề + Thân văn bản) Top-1 & Top-2 | Đánh giá biến thể |
| :--- | :--- | :--- | :--- | :--- |
| **`q_001`** (Vị trí vật thể ảnh) | **BM25** | #1: `cv_009` (13.15), #2: `cv_003` (12.09) | #1: `cv_008` (14.74), #2: `cv_003` (14.28) | Không đưa được `cv_010`/`cv_011` lên Top. |
| | **E5 In-Memory** | #1: `cv_009` (0.1452), #2: `cv_014` (0.1504) | #1: `cv_009` (0.1496), #2: `cv_014` (0.1513) | Thứ hạng không đổi; khoảng cách giữ nguyên. |
| **`q_005`** (Mã độc phá CSDL) | **BM25** | #1: `net_011` SQLi (13.32), #2: `net_013` (12.78) | #1: `net_013` CSRF (15.67), #2: `net_011` (12.81) | **Hồi quy (Regression):** `net_011` bị tụt xuống #2. |
| | **E5 In-Memory** | #1: `net_012` XSS (0.1319), #2: `net_011` (0.1344) | #1: `net_012` XSS (0.1272), #2: `net_013` (0.1404) | **Hồi quy (Regression):** `net_011` bị tụt xuống #3. |
| **`q_019`** (L1 vs L2) | **BM25** | #1: `dl_008` (14.30), #2: `ml_005` (13.78) | #1: `ml_005` Ridge (21.27), #2: `ml_004` (21.15) | **Cải thiện:** L1 và L2 lên thẳng Top 1 và 2. |
| | **E5 In-Memory** | #1: `ml_005` Ridge (0.1310), #2: `ml_004` (0.1326) | #1: `ml_005` Ridge (0.1117), #2: `ml_004` (0.1260) | Giữ nguyên Top 1 và Top 2 (khoảng cách tốt hơn). |

### Kết luận khoa học từ thử nghiệm:
1. **Không áp dụng thay thế âm thầm vào Production Baseline:**
   - Việc ghép tiêu đề vào đoạn văn tuy có lợi cho các câu hỏi chứa thuật ngữ đối chiếu trực tiếp như `q_019` (L1/L2), nhưng lại gây **hiện tượng hồi quy rõ rệt trên `q_005`** (làm trôi dạt tài liệu mục tiêu SQL Injection `net_011` xuống dưới CSRF), và **hoàn toàn không giải quyết được vấn đề của `q_001`**.
   - Việc thay đổi cách mã hóa văn bản hoặc xếp hạng của hệ thống sẽ làm sai lệch tập ứng viên trong candidate pool 105 nhãn đã trích xuất ở G3, đòi hỏi phải tái tạo toàn bộ pool và thẩm định lại từ đầu.
2. **Nguyên tắc liêm chính học thuật:**
   - Tuyệt đối không can thiệp hardcode thứ hạng hay sửa đổi nội dung corpus chỉ để tạo kết quả demo đẹp mắt.
   - Giữ nguyên hệ thống retrieval baseline đã được kiểm duyệt của G2, báo cáo trung thực sự khác biệt giữa văn phong người dùng tự nhiên và thuật ngữ học thuật chuyên sâu để phục vụ bài thảo luận phân tích trong Seminar.

---

## 4. Bằng Chứng Kiểm Tra Nghiệm Thu (Minimal Verification)

### 4.1 Kiểm tra tự động các bài test demo (`scripts/verify_demo_acceptance.py`):
- **Check 1 (Search):** **PASS** — Cả Chroma và BM25 truy xuất dữ liệu thật thành công.
- **Check 2 (Metadata Filtering):** **PASS** — Lọc theo chủ đề `Database` cho kết quả 100% thuộc chủ đề.
- **Check 3 (Isolated Collection CRUD):** **PASS** — Thực thi chu trình Add/Search/Update/Delete hoàn toàn trong thư mục tạm cô lập; collection chính thức `seminar_corpus_e5` giữ nguyên vẹn 114 tài liệu; demo collection thật của người dùng không bị tác động.
- **Check 4 (Experiment Results & Gate Lock):** **PASS** — Đọc động thống kê nhãn; khi 105 nhãn đang pending, UI khóa chặt không hiển thị official benchmark.

### 4.2 Kiểm tra các chốt bảo vệ an toàn (Safeguards):
- **Kiểm tra Escape HTML (`safe_html`):** Chuỗi `Test <script>alert(1)</script> & foo` được escape thành `Test &lt;script&gt;alert(1)&lt;/script&gt; &amp; foo` $\rightarrow$ **PASS**.
- **Từ chối trùng ID khi Add:** `ValueError: Document ID 'id_1' already exists in demo collection.` $\rightarrow$ **PASS**.
- **Yêu cầu ID tồn tại khi Update:** `ValueError: Document ID 'non_existent_id' not found in demo collection.` $\rightarrow$ **PASS**.
- **Kiểm tra tính nhất quán mã băm (Double-Hash Gate):**
  - Khớp cả `corpus_sha256` và `review_pool_sha256`: `is_eligible == True` $\rightarrow$ **PASS**.
  - Review pool bị chỉnh sửa sau khi tính toán (SHA-256 mismatch): `is_eligible == False` kèm cảnh báo khóa kết quả $\rightarrow$ **PASS**.

### 4.3 Trạng thái kiểm tra giao diện trên trình duyệt:
- **Xác nhận kỹ thuật:** Server Streamlit khởi chạy thành công, endpoint kiểm tra sức khỏe HTTP phản hồi mã trạng thái 200 OK.
- **Ghi nhận minh bạch:** `NOT VERIFIED IN BROWSER DIRECTLY` — Do môi trường dòng lệnh của AI Coding Agent không có môi trường đồ họa trình duyệt (Headless/No GUI) để thực hiện click chuột người dùng trực tiếp. Khuyến nghị người dùng chạy lệnh `streamlit run app.py` trên máy cá nhân để trải nghiệm trực quan giao diện Dark Theme mới.

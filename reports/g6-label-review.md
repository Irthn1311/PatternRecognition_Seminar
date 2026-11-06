# Báo Cáo Thẩm Định Nhãn Tương Quan (Human Relevance Review) — Task G6

- **Dự án:** PatternRecognition_Seminar
- **Nhiệm vụ:** TASK G6 — Human Relevance Review & Official Retrieval Evaluation
- **GitHub Issue:** [#11 — G6: Human-verified labels, official metrics, browser rehearsal and seminar package](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/11)
- **Thời gian thực hiện:** Tháng 11/2026
- **Tập dữ liệu thẩm định:** `data/eval/review_pool.csv` (105 cặp ứng viên pooled từ 12 test queries)

---

## 1. Quy Ước Thẩm Định Relevance (Relevance Guidelines)

Mỗi đoạn văn bản ứng viên được người kiểm duyệt đánh giá dựa trên mức độ đáp ứng ý định tìm kiếm thực tế của câu hỏi:
- **Điểm 2 (Directly relevant — Trực tiếp liên quan):** Đoạn văn bản chứa thông tin đáp ứng rõ ràng, giải quyết trúng đích câu hỏi, dù có thể không sử dụng chính xác cùng một tập từ vựng với câu hỏi.
- **Điểm 1 (Partially relevant — Liên quan một phần):** Đoạn văn bản đề cập đến ngữ cảnh nền tảng, cơ chế liên đới hoặc một khía cạnh phụ của câu hỏi nhưng chưa giải quyết đầy đủ ý định tìm kiếm.
- **Điểm 0 (Irrelevant / Distractor — Không liên quan):** Đoạn văn bản không cung cấp thông tin thực chất giúp giải quyết câu hỏi, hoặc chỉ trùng lặp từ khóa chung chung.

---

## 2. Thống Kê Tổng Quan Kết Quả Thẩm Định

Toàn bộ **105 cặp query-document** trong `data/eval/review_pool.csv` đã được người kiểm duyệt thẩm định và xác nhận đầy đủ:
- **Tổng số cặp ứng viên:** **105 cặp** (thuộc 12 test queries).
- **Tiến độ thẩm định:** **105 / 105 (100.0% verified)**, không còn nhãn pending nào.
- **Phân bố điểm tương quan chính thức:**
  - **Điểm 0 (Không liên quan):** **65 cặp (61.9%)**
  - **Điểm 1 (Liên quan một phần):** **18 cặp (17.1%)**
  - **Điểm 2 (Trực tiếp liên quan):** **22 cặp (21.0%)**

---

## 3. Chi Tiết Thẩm Định Theo 12 Nhóm Truy Vấn

### Nhóm 1: `q_002` (10 ứng viên)
*"cách giải quyết khi mô hình học máy thuộc lòng dữ liệu huấn luyện và dự đoán kém trên thực tế"*
- `ml_015` = **2** (Định nghĩa quá khớp và tổng hợp các giải pháp khắc phục).
- `ml_004` = **2** (Điều chuẩn L1 Lasso trực tiếp chống quá khớp).
- `ml_005` = **2** (Điều chuẩn L2 Ridge trực tiếp chống quá khớp).
- `dl_007` = **2** *(Quyết định điều chỉnh: Nâng từ 1 lên 2 vì Dropout là cơ chế điều chuẩn trực tiếp chống quá khớp trong học sâu)*.
- `ml_014` = **1** (Nguyên lý đánh đổi bias-variance giải thích nguồn gốc phương sai cao).
- `ml_016` = **1** (Kiểm định chéo K-Fold giúp phát hiện và kiểm soát quá khớp).
- `dl_018` = **0**, `dl_019` = **0**, `ml_001` = **0**, `ml_017` = **0** (Không đề cập giải pháp cho overfitting).

### Nhóm 2: `q_003` (12 ứng viên)
*"giải pháp tăng tốc độ tìm kiếm bản ghi trong bảng cơ sở dữ liệu hàng triệu dòng"*
- `db_003` = **2** (Chỉ mục B+Tree tối ưu tìm kiếm theo khóa và quét khoảng với $O(\log N)$).
- `db_004` = **2** (Chỉ mục băm Hash Index cho phép tìm kiếm chính xác với $O(1)$).
- `db_006` = **2** *(Quyết định điều chỉnh: Nâng từ 1 lên 2 vì Partition Pruning trực tiếp thu hẹp phạm vi tìm kiếm trên bảng lớn)*.
- `db_012` = **1** (Bộ tối ưu truy vấn cân nhắc kế hoạch Index Scan).
- `db_007` = **1** (Kỹ thuật phân mảnh Sharding ở cấp độ phân tán).
- `algo_002` = **0**, `algo_006` = **0**, `algo_016` = **0**, `db_001` = **0**, `db_005` = **0**, `db_011` = **0**, `db_013` = **0** (Cấu trúc dữ liệu trong RAM hoặc không liên quan đến tăng tốc tìm kiếm).

### Nhóm 3: `q_006` (11 ứng viên)
*"cơ chế giúp mô hình dịch máy hiểu được mối liên hệ giữa các từ đứng cách xa nhau trong câu dài"*
- `dl_013` = **2** (Cơ chế Self-Attention tính quan hệ trực tiếp giữa mọi từ trong câu).
- `dl_014` = **2** (Multi-Head Attention trong Transformer theo dõi quan hệ ngữ cảnh đa chiều).
- `dl_011` = **2** *(Quyết định điều chỉnh: Nâng từ 1 lên 2 vì LSTM cung cấp cơ chế Cell State trực tiếp xử lý chuỗi dài)*.
- `dl_015` = **1** (Mã hóa vị trí Positional Encoding bổ trợ thông tin thứ tự).
- `dl_017` = **1** (Mô hình BERT học ngữ cảnh hai chiều).
- `dl_010` = **1** *(Quyết định điều chỉnh: Nâng từ 0 lên 1 vì giải thích trực tiếp nguyên nhân hạn chế phụ thuộc xa của RNN)*.
- `dl_016` = **0**, `algo_019` = **0**, `db_012` = **0**, `db_017` = **0**, `ml_014` = **0** (Không liên quan).

### Nhóm 4: `q_010` (8 ứng viên)
*"kỹ thuật tối ưu bộ nhớ để chạy các mô hình AI ngôn ngữ lớn trên máy tính cá nhân hoặc điện thoại"*
- `dl_019` = **2** (Lượng tử hóa mô hình Model Quantization chuyển đổi trọng số sang INT8/INT4 trực tiếp giảm RAM).
- `dl_016` = **0** *(Quyết định điều chỉnh: Hạ từ 1 xuống 0 vì đoạn văn chỉ trình bày kiến trúc GPT, không nói về tối ưu bộ nhớ)*.
- `cv_007` = **0** (MobileNet tối ưu tính toán cho thị giác máy tính, không phải LLM).
- `dl_018` = **0**, `dl_011` = **0**, `cv_016` = **0**, `algo_017` = **0**, `ml_008` = **0** (Không liên quan).

### Nhóm 5: `q_013` (7 ứng viên)
*"thuộc tính ACID và tính bền vững của giao dịch cơ sở dữ liệu quan hệ"*
- `db_002` = **2** (Định nghĩa đầy đủ 4 thuộc tính ACID và tính bền vững Durability).
- `db_018` = **2** (Nhật ký ghi trước WAL là cơ chế kỹ thuật trực tiếp đảm bảo Durability và phục hồi sự cố).
- `db_008` = **1** (Các mức độ cô lập liên quan đến thành phần Isolation trong ACID).
- `db_011` = **1** (MVCC liên quan đến kiểm soát đồng thời Isolation, không giải thích Durability).
- `db_001` = **0**, `db_007` = **0**, `ml_002` = **0** (Không liên quan).

### Nhóm 6: `q_016` (6 ứng viên)
*"cơ sở dữ liệu vector và thuật toán tìm kiếm láng giềng gần đúng HNSW"*
- `db_016` = **2** (Thuật toán đồ thị phân cấp HNSW cho tìm kiếm ANN).
- `db_015` = **2** (Định nghĩa cơ sở dữ liệu vector và ứng dụng tìm kiếm láng giềng gần nhất).
- `db_017` = **1** (Các độ đo khoảng cách Cosine, L2 dùng trong CSDL vector).
- `ml_010` = **1** (Thuật toán KNN là bài toán láng giềng gần nhất nền tảng trong học máy).
- `ml_009` = **0**, `db_002` = **0** (Không liên quan).

### Nhóm 7: `q_020` (7 ứng viên)
*"phân biệt bài toán phân đoạn ngữ nghĩa và phân đoạn cá thể trong thị giác máy tính"*
- `cv_014` = **2** (Định nghĩa phân đoạn cá thể tách biệt từng cá thể, đối chiếu với phân đoạn ngữ nghĩa).
- `cv_013` = **2** (Định nghĩa phân đoạn ngữ nghĩa gán nhãn từng pixel không tách cá thể).
- `cv_015` = **1** (Kiến trúc U-Net trong phân đoạn ảnh y tế).
- `cv_009` = **0**, `cv_018` = **0**, `ml_001` = **0**, `cv_016` = **0** (Không liên quan).

### Nhóm 8: `q_021` (6 ứng viên)
*"so sánh giao thức truyền tải TCP và UDP trong mạng máy tính"*
- `net_002` = **2** (Đặc tính hướng kết nối, tin cậy, có thứ tự của TCP).
- `net_004` = **2** (Đặc tính phi kết nối, tốc độ cao của UDP, so sánh đối chiếu trực tiếp với TCP).
- `net_003` = **1** (Cơ chế kiểm soát tắc nghẽn đặc thù của TCP mà UDP không có).
- `net_006` = **1** (So sánh HTTP/2 trên TCP vs HTTP/3 trên QUIC/UDP).
- `net_001` = **0** *(Quyết định điều chỉnh: Hạ từ 1 xuống 0 vì chỉ giới thiệu mô hình OSI/TCP-IP tổng quan, không giải thích TCP và UDP)*.
- `net_005` = **0** (HTTP/HTTPS tầng ứng dụng).

### Nhóm 9: `q_024` (8 ứng viên)
*"so sánh mã hóa đối xứng AES và mã hóa bất đối xứng RSA"*
- `net_015` = **2** (So sánh trực tiếp nguyên lý một khóa chung AES vs cặp khóa công khai/bí mật RSA).
- `net_016` = **1** (Ứng dụng kết hợp cả AES và RSA trong giao thức bắt tay TLS).
- `net_018` = **0**, `net_019` = **0**, `db_004` = **0**, `ml_017` = **0**, `ml_014` = **0**, `cv_015` = **0** (Không liên quan).

### Nhóm 10: `q_026` (11 ứng viên)
*"chỉ số IoU và thuật toán Non-Maximum Suppression NMS"*
- `cv_012` = **2** (Định nghĩa trực tiếp chỉ số IoU và thuật toán NMS lọc hộp bao trùng lặp).
- `cv_010` = **1** (Faster R-CNN ứng dụng IoU và NMS ở tầng đề xuất vùng).
- `cv_011` = **1** (YOLO ứng dụng IoU và NMS ở đầu ra phân loại).
- `algo_013` = **0**, `ml_011` = **0**, `net_007` = **0**, `ml_002` = **0**, `db_009` = **0**, `ml_006` = **0**, `ml_017` = **0**, `algo_006` = **0** (Không liên quan).

### Nhóm 11: `q_028` (8 ứng viên)
*"cơ chế MVCC Multi-Version Concurrency Control PostgreSQL"*
- `db_011` = **2** (Cơ chế MVCC trong PostgreSQL: đọc ghi không chặn nhau, snapshot, xmin, xmax).
- `db_008` = **1** (Các mức độ cô lập giao dịch được thực thi trên nền tảng MVCC).
- `db_010` = **0** *(Quyết định điều chỉnh: Hạ từ 1 xuống 0 vì chỉ mô tả locking, không giải thích cơ chế MVCC)*.
- `db_019` = **0**, `dl_014` = **0**, `dl_013` = **0**, `dl_008` = **0**, `dl_016` = **0** (Không liên quan).

### Nhóm 12: `q_030` (11 ứng viên — UNANSWERABLE)
*"thuật toán đồng thuận Proof of Stake và xử lý phân nhánh blockchain Ethereum"*
- Toàn bộ 11 ứng viên (`algo_014`, `algo_010`, `net_015`, `dl_002`, `algo_008`, `ml_001`, `ml_007`, `cv_017`, `algo_018`, `ml_008`, `db_002`) đều được xác nhận nhận điểm **0**.
- **Xác thực:** Toàn bộ kho tài liệu 114 đoạn hoàn toàn không chứa nội dung về blockchain hay Proof of Stake. Đây là câu hỏi kiểm thử khả năng đối phó với truy vấn ngoài phạm vi, được phân tích định tính riêng và loại khỏi mẫu số tính điểm trung bình benchmark.

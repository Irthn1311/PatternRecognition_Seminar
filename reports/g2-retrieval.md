# Báo Cáo Kỹ Thuật G2: Lập Chỉ Mục Corpus & Đối Sánh Truy Hồi Chroma vs Vietnamese BM25

- **Dự án:** PatternRecognition_Seminar
- **Nhiệm vụ:** TASK G2 — Chroma Core, Indexing & BM25 Retrieval
- **GitHub Issue:** [#5 — G2: Index 114 passages and implement Chroma + Vietnamese BM25 retrieval CLI](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/5)
- **Thời gian thực hiện:** Tháng 11/2026
- **Môi trường:** Windows 11 Home / Python 3.12.16 / PowerShell 5.1

---

## 1. Thông Số Môi Trường và Thư Viện

| Thành phần | Phiên bản / Thông số | Vai trò |
| :--- | :--- | :--- |
| **Hệ điều hành** | Windows 11 (win32) | Môi trường thử nghiệm local |
| **Python** | Python 3.12.16 | Môi trường ảo `.venv` |
| **ChromaDB** | 1.5.9 | `chromadb.PersistentClient` (collection: `seminar_corpus_e5`) |
| **Sentence Transformers** | 6.2.0 | Tương thích PyTorch 2.15.0+cpu |
| **PyTorch (Backend)** | 2.15.0+cpu | Suy luận embedding hoàn toàn trên CPU |
| **NumPy** | 2.5.4 | Xử lý vector mảng nhúng |
| **rank-bm25** | 0.2.2 | Thuật toán truy xuất từ vựng BM25Okapi |
| **pyvi** | 0.1.1 | `pyvi.ViTokenizer` phân đoạn từ ghép tiếng Việt |
| **Embedding Model** | `intfloat/multilingual-e5-small` | 384 dimensions, tiền tố `passage: ` / `query: `, chuẩn hóa L2 |
| **Metric Vector** | Cosine Distance | Cấu hình collection: `{"hnsw:space": "cosine"}` |
| **Quy mô Corpus** | 114 passages | `data/corpus/documents.jsonl` (cân bằng 19 đoạn / 6 chủ đề) |

---

## 2. Kiểm Tra Giới Hạn Token & Mã Hóa Batch

Trước khi đưa vào lưu trữ, toàn bộ 114 tài liệu được kiểm tra độ dài token bằng tokenizer chính thức của mô hình `intfloat/multilingual-e5-small`:
- **Độ dài token lớn nhất đo được:** **147 tokens** (đoạn `dl_001` - Perceptron Đa tầng MLP).
- **Độ dài token trung bình:** **129.32 tokens**.
- **Giới hạn tối đa của mô hình (max_seq_length):** **512 tokens**.
- **Số tài liệu bị cắt ngắn (Truncated count):** **0** (100% tài liệu nằm trọn vẹn trong phạm vi tiếp nhận).
- **Cơ chế Batching:** Quá trình tính vector embedding được thực thi theo từng lô nhỏ (`batch_size=32`), hoàn thành nạp 114 tài liệu trong khoảng 2.1 giây trên CPU.

---

## 3. Quản Lý Tính Nhất Quán & Tái Lập (Index Consistency)

Để loại trừ triệt để nguy cơ sử dụng chỉ mục cũ (stale/misaligned index) khi dữ liệu hoặc cấu hình thay đổi:
- **Tệp Manifest (`chroma_db/index_manifest.json`):** Lưu trữ SHA-256 của tập dữ liệu `documents.jsonl`, tên mô hình, số chiều vector, số lượng tài liệu và thời điểm tạo.
- **Đối soát ba bên (Three-way Consistency Check):** Phương thức `check_index_consistency()` đối chiếu đồng thời:
  1. Số lượng tài liệu trong collection Chroma.
  2. Số lượng tài liệu ghi nhận trong manifest.
  3. **Số lượng tài liệu thực tế đọc từ tệp corpus trên đĩa.**
  4. Mã băm SHA-256 của tệp corpus hiện tại so với manifest.
- **Ngăn chặn Silent Upsert:**
  - Nếu collection đã tồn tại nhưng manifest bị thiếu hoặc dữ liệu/cấu hình bị thay đổi (ví dụ có tài liệu bị xóa khỏi corpus), lệnh build thông thường sẽ **chủ động từ chối và báo lỗi ngay lập tức** (thay vì âm thầm upsert đè lên collection cũ khiến các ID cũ bị mồ côi/stale).
  - Người dùng bắt buộc phải truyền `--rebuild` (`python scripts/build_index.py --rebuild`) để hệ thống xóa và tái tạo sạch sẽ collection mục tiêu.
- **Bảo vệ Truy vấn:** `ChromaRetriever` tự động kiểm tra tính nhất quán trước khi thực thi truy vấn. Nếu index bị lỗi thời hoặc không khớp dữ liệu, truy vấn sẽ ném ngoại lệ rõ ràng thay vì trả về kết quả sai lệch.

---

## 4. Kết Quả Kiểm Chứng Tối Thiểu (Minimal Verification)

### Check 1 — Indexing & Idempotency trên 114 passages: PASS
- Chạy lần đầu: nạp 114 tài liệu, tạo manifest thành công.
- Chạy lần thứ hai: hệ thống nhận diện `Index 'seminar_corpus_e5' is consistent (114 records, SHA256: a9633f3a)`, thông báo `Index is already up-to-date. Skipping re-encoding`. Số lượng bản ghi giữ nguyên chính xác 114.

### Check 2 — Search Reopen (Tiến trình độc lập): PASS
- Đóng tiến trình index hoàn toàn, khởi tạo tiến trình Python mới từ dòng lệnh.
- Mở lại cơ sở dữ liệu `chroma_db` đã lưu và thực thi truy vấn thành công mà không cần nạp lại corpus.

### Check 3 — Kiểm chứng Ngăn Chặn Stale Index (Thư mục tạm cô lập): PASS
Thực nghiệm kiểm chứng độc lập trên tập dữ liệu và thư mục database tạm thời:
1. Tạo corpus gồm 3 tài liệu và build index thành công (Count = 3).
2. Xóa 1 tài liệu khỏi corpus (corpus chỉ còn 2 tài liệu).
3. Chạy `build_index.py --check`: Phát hiện lỗi mismatch SHA-256 và trả về exit code `1`.
4. Chạy `build_index.py` thông thường không có `--rebuild`: Từ chối nạp đè và báo lỗi `[Build Error]: Existing index is inconsistent... Ingestion aborted to prevent stale or orphaned records. Please run with '--rebuild'` (Exit code `1`).
5. Chạy `build_index.py --rebuild`: Thực thi thành công, xóa sạch collection cũ và tạo lại collection mới. Kiểm tra Chroma xác nhận chỉ còn đúng 2 tài liệu (`['t_001', 't_002']`).

### Check 4 — Kiểm chứng CLI Error Code & Input Guard: PASS
1. **Kiểm tra Exit Code khi có lỗi:** Chạy `compare_search.py --db-path ./non_existent_db --mode both --query "kiểm tra lỗi"`. Hệ thống báo lỗi Chroma không tìm thấy collection, BM25 vẫn chạy để xuất thông tin gỡ lỗi, và toàn bộ lệnh kết thúc với **exit code `1`** (không còn trả về exit code 0 giả mạo).
2. **Kiểm tra Top-K Guard:** Chạy với `--top-k 0` và `--top-k -1`: Báo lỗi `Error: --top-k must be a positive integer greater than 0.` và kết thúc ngay với exit code `1`, ngăn ngừa lỗi negative slicing trong Python.

### Check 5 — BM25 & Chroma Comparison trên 3 truy vấn Dev: PASS
Thực nghiệm đối sánh chi tiết Top-5 trên 3 câu hỏi thuộc tập `dev` (`q_001`, `q_005`, `q_019`):

#### Truy vấn 1 (`q_001` — Semantic Paraphrase, Dev):
*"làm thế nào để máy tính tự động tìm ra các vật thể và vị trí của chúng trong một bức ảnh?"*

| Hạng | Chroma (Cosine Distance - nhỏ hơn là gần hơn) | BM25 (BM25 Score - lớn hơn là khớp hơn) |
| :---: | :--- | :--- |
| **Top 1** | `cv_009` (0.1452) — Phân loại Ảnh Image Classification | `cv_009` (13.1526) — Phân loại Ảnh Image Classification |
| **Top 2** | `cv_014` (0.1504) — Phân đoạn Cá thể Mask R-CNN | `cv_003` (12.0950) — Trường Thụ nhận Receptive Field |
| **Top 3** | `cv_018` (0.1614) — Trích xuất Đặc trưng SIFT và HOG | `cv_008` (11.6684) — Vision Transformer ViT |
| **Top 4** | `cv_008` (0.1663) — Vision Transformer ViT | `net_017` (11.3555) — Tấn công Từ chối Dịch vụ DDoS |
| **Top 5** | `ml_009` (0.1681) — Máy vector hỗ trợ SVM | `cv_004` (10.7518) — Mạng AlexNet ImageNet |

*Nhận xét quan sát:*
- Cả hai phương pháp đều đưa `cv_009` lên vị trí số 1.
- Chroma đưa tài liệu định vị vật thể `cv_014` (Mask R-CNN) lên Top 2 nhờ bắt được ngữ nghĩa "vị trí của vật thể trong ảnh".
- BM25 xếp `net_017` (DDoS) vào Top 4 do sự trùng khớp ngẫu nhiên của các từ vựng phổ biến như "máy tính" và "ảnh" trong nội dung.

---

#### Truy vấn 2 (`q_005` — Semantic Paraphrase, Dev):
*"làm sao để ngăn chặn người dùng nhập mã độc vào ô tìm kiếm nhằm phá hoại cơ sở dữ liệu web"*

| Hạng | Chroma (Cosine Distance) | BM25 (BM25 Score) |
| :---: | :--- | :--- |
| **Top 1** | `net_012` (0.1319) — Lỗ hổng Cross-Site Scripting XSS | `net_011` (13.3263) — Lỗ hổng SQL Injection |
| **Top 2** | `net_011` (0.1344) — Lỗ hổng SQL Injection | `net_013` (12.7854) — Lỗ hổng CSRF |
| **Top 3** | `net_017` (0.1452) — Tấn công Từ chối Dịch vụ DDoS | `net_012` (11.4850) — Lỗ hổng Cross-Site Scripting XSS |
| **Top 4** | `net_013` (0.1546) — Lỗ hổng CSRF | `algo_006` (10.5543) — Bảng Băm Hash Table |
| **Top 5** | `algo_006` (0.1606) — Bảng Băm Hash Table | `db_015` (6.7904) — Cơ sở Dữ liệu Vector |

*Nhận xét quan sát:*
- BM25 xếp trúng đích `net_011` (SQL Injection) ở Top 1 nhờ trùng khớp cụm từ "cơ sở dữ liệu".
- Chroma xếp `net_012` (XSS) ở Top 1 và `net_011` ở Top 2 (khoảng cách cosine rất sít sao: 0.1319 vs 0.1344); cả 4 kết quả hàng đầu của Chroma đều thuộc nhóm an ninh ứng dụng web.

---

#### Truy vấn 3 (`q_019` — Ambiguous / Confusable, Dev):
*"sự khác biệt giữa điều chuẩn L1 Lasso và điều chuẩn L2 Ridge"*

| Hạng | Chroma (Cosine Distance) | BM25 (BM25 Score) |
| :---: | :--- | :--- |
| **Top 1** | `ml_005` (0.1310) — Điều chuẩn L2 Ridge | `dl_008` (14.3032) — Chuẩn hóa Batch Normalization |
| **Top 2** | `ml_004` (0.1326) — Điều chuẩn L1 Lasso | `ml_005` (13.7851) — Điều chuẩn L2 Ridge |
| **Top 3** | `db_017` (0.1658) — Khoảng cách Vector Cosine & L2 | `ml_004` (13.0586) — Điều chuẩn L1 Lasso |
| **Top 4** | `dl_009` (0.1716) — Chuẩn hóa Layer Normalization | `cv_004` (11.4004) — Mạng AlexNet ImageNet |
| **Top 5** | `db_001` (0.1719) — CSDL Quan hệ và Chuẩn hóa | `ml_015` (10.7109) — Hiện tượng Quá khớp |

*Nhận xét quan sát:*
- Chroma xếp hoàn hảo hai tài liệu mục tiêu `ml_005` (Ridge) và `ml_004` (Lasso) ở Top 1 và Top 2.
- BM25 bị ảnh hưởng bởi tần suất từ "chuẩn hóa / điều chuẩn", khiến đoạn `dl_008` (Batch Normalization) nhận điểm số cao hơn cả hai đoạn mục tiêu.

---

### Check 6 — Metadata Topic Filtering: PASS
Thực hiện truy vấn với bộ lọc `--topic "Algorithms"`:
```powershell
python scripts/compare_search.py --query "tìm kiếm phân cấp đồ thị" --topic "Algorithms" --top-k 5
```
Kết quả ghi nhận:
- **100% kết quả** của Chroma (`algo_009`, `algo_008`, `algo_002`, `algo_013`, `algo_012`) đều thuộc chủ đề `Algorithms`.
- **100% kết quả** của BM25 (`algo_009`, `algo_008`, `algo_002`, `algo_011`, `algo_010`) đều thuộc chủ đề `Algorithms`.
- Cơ chế lọc ứng viên trước khi xếp hạng (pre-filtering) hoạt động chính xác trên cả hai công cụ.

---

## 5. Phân Biệt Thang Đo và Giới Hạn Nghiên Cứu

1. **Tính chất thang đo:**
   - Khoảng cách Cosine của Chroma là độ đo khoảng cách hình học trong không gian vector (giá trị từ 0 đến 2, giá trị càng nhỏ càng tương đồng).
   - Điểm số BM25 là hàm đo tần suất từ vựng (TF-IDF phi tuyến tính) không bị chặn trên.
   - Tuyệt đối không quy đổi hay so sánh số học trực tiếp giữa hai thang đo này thành phần trăm độ tin cậy.
2. **Nguyên tắc khoa học về đánh giá:**
   - Các ví dụ quan sát trên đây là bằng chứng kỹ thuật chứng minh cả hai hệ thống hoạt động đúng chức năng trên cùng tập dữ liệu.
   - **Không được kết luận mô hình nào vượt trội hơn** chỉ dựa vào 3 câu hỏi trên. Việc kết luận hiệu năng retrieval đòi hỏi tập nhãn chính thức (sau khi thẩm định qrels) và tính toán các chỉ số thống kê tiêu chuẩn (MRR@K, nDCG@K, Recall@K) sẽ được thực hiện ở G3.

# Báo Cáo Kỹ Thuật G3: Quy Trình Đánh Giá Truy Hồi & Candidate Review Pool

- **Dự án:** PatternRecognition_Seminar
- **Nhiệm vụ:** TASK G3 — Retrieval Evaluation & Relevance Review
- **GitHub Issue:** [#7 — G3: Reproducible retrieval evaluation and pooled relevance review](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/7)
- **Thời gian thực hiện:** Tháng 11/2026
- **Môi trường:** Windows 11 Home / Python 3.12.16 / PowerShell 5.1

---

## 1. Mục Tiêu & Phương Pháp Luận G3

Mục tiêu cốt lõi của Task G3 là xây dựng **hạ tầng đánh giá truy hồi có thể tái lập một cách khoa học**, phân định rạch ròi giữa:
1. **Hạ tầng thuật toán và xếp hạng (Rankings & Code):** Được kiểm tra tính đúng đắn toán học thông qua các fixture test.
2. **Dữ liệu nhãn thẩm định (Ground Truth Labels):** Nhãn tương quan do AI đề xuất ban đầu (Draft Labels) chỉ là bản nháp gợi ý. **Tuyệt đối không được tự động công nhận nhãn AI làm ground truth chính thức.**

Chính vì vậy, G3 thiết lập cơ chế **Verified-Label Gate** (Cổng kiểm soát nhãn đã xác thực): Hệ thống từ chối tính toán và xuất các chỉ số benchmark chính thức cho đến khi toàn bộ các ứng viên trong pool được con người kiểm duyệt và gán trạng thái `verified`.

---

## 2. Cố Định Tập Truy Vấn Đánh Giá (Evaluation Subset)

Tập truy vấn đánh giá được chốt trước tại tệp `data/eval/evaluation_subset.json` gồm đúng **12 test queries** đại diện đầy đủ cho 5 thể loại (chốt trước khi chạy xếp hạng nhằm ngăn chặn thiên kiến chọn lọc câu hỏi):

| Mã Query | Thể loại (Category) | Tập chia (Split) | Câu hỏi truy vấn |
| :--- | :--- | :---: | :--- |
| `q_002` | `semantic_paraphrase` | test | cách giải quyết khi mô hình học máy thuộc lòng dữ liệu huấn luyện và dự đoán kém trên thực tế |
| `q_003` | `semantic_paraphrase` | test | giải pháp tăng tốc độ tìm kiếm bản ghi trong bảng cơ sở dữ liệu hàng triệu dòng |
| `q_006` | `semantic_paraphrase` | test | cơ chế giúp mô hình dịch máy hiểu được mối liên hệ giữa các từ đứng cách xa nhau trong câu dài |
| `q_010` | `semantic_paraphrase` | test | kỹ thuật tối ưu bộ nhớ để chạy các mô hình AI ngôn ngữ lớn trên máy tính cá nhân hoặc điện thoại |
| `q_013` | `direct_technical` | test | thuộc tính ACID và tính bền vững của giao dịch cơ sở dữ liệu quan hệ |
| `q_016` | `direct_technical` | test | cơ sở dữ liệu vector và thuật toán tìm kiếm láng giềng gần đúng HNSW |
| `q_020` | `ambiguous_confusable` | test | phân biệt bài toán phân đoạn ngữ nghĩa và phân đoạn cá thể trong thị giác máy tính |
| `q_021` | `ambiguous_confusable` | test | so sánh giao thức truyền tải TCP và UDP trong mạng máy tính |
| `q_024` | `ambiguous_confusable` | test | so sánh mã hóa đối xứng AES và mã hóa bất đối xứng RSA |
| `q_026` | `exact_match` | test | chỉ số IoU và thuật toán Non-Maximum Suppression NMS |
| `q_028` | `exact_match` | test | cơ chế MVCC Multi-Version Concurrency Control PostgreSQL |
| `q_030` | `unanswerable` | test | thuật toán đồng thuận Proof of Stake và xử lý phân nhánh blockchain Ethereum |

- **Tổng số:** 12 queries (11 câu hỏi có thể trả lời trong corpus, 1 câu hỏi unanswerable).
- Các câu hỏi `dev` được giữ riêng phục vụ kiểm thử và tinh chỉnh, không trộn lẫn vào tập benchmark chính thức.

---

## 3. Xây Dựng Candidate Review Pool

Script `scripts/prepare_eval_pool.py` tiến hành hợp nhất các ứng viên tiềm năng cho 12 câu hỏi:
1. Top-5 ứng viên trả về từ Chroma Semantic Search (`seminar_corpus_e5`).
2. Top-5 ứng viên trả về từ Vietnamese BM25 (`pyvi` + `rank_bm25`).
3. Các ứng viên đã có trong nhãn dự thảo `qrels_draft.jsonl`.
4. Trích xuất toàn văn đoạn văn bản (`doc_text`), tiêu đề và URL nguồn từ corpus.

### Kết quả xuất ra tệp `data/eval/review_pool.csv`:
- **Tổng số cặp query-document độc nhất:** **105 cặp**.
- **Trung bình mỗi câu hỏi:** ~8.75 ứng viên cần kiểm duyệt.
- **Trạng thái khởi tạo:**
  - `verified_relevance`: **Để trống hoàn toàn**.
  - `review_status`: Khởi tạo mặc định là `pending`.
- **Cơ chế an toàn bảo vệ dữ liệu:** Script tự động phát hiện nếu tệp `review_pool.csv` đã có nhãn verified và từ chối ghi đè trừ khi có cờ `--overwrite`.

---

## 4. Công Thức & Kiểm Chứng Toán Học Các Metrics

Module `src/evaluation.py` triển khai các độ đo IR tiêu chuẩn theo định nghĩa toán học chính xác:

### 4.1 Công thức tính toán
- **`Hit@K` (với K=5):**
  $$Hit@K = \begin{cases} 1.0 & \text{nếu tồn tại ít nhất một tài liệu có } rel = 2 \text{ trong Top-K} \\ 0.0 & \text{ngược lại} \end{cases}$$
- **`MRR@K` (Mean Reciprocal Rank với K=5):**
  $$RR@K = \begin{cases} \frac{1}{r} & \text{với } r \le K \text{ là thứ hạng xuất hiện đầu tiên của tài liệu có } rel = 2 \\ 0.0 & \text{nếu không có tài liệu } rel = 2 \text{ trong Top-K} \end{cases}$$
- **`nDCG@K` (Normalized Discounted Cumulative Gain với K=5):**
  - Mức tăng ích (Graded Gain): $gain(rel) = 2^{rel} - 1$ (với $rel \in \{0, 1, 2\}$)
  - Hệ số chiết khấu vị trí: $discount(i) = \log_2(i + 1)$
  - $DCG@K = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i + 1)}$
  - $IDCG@K$ được tính bằng cách sắp xếp giảm dần toàn bộ các nhãn đã được kiểm duyệt trong pool ứng viên của câu hỏi đó:
    $$IDCG@K = \sum_{i=1}^{\min(K, |judged\_positives|)} \frac{2^{rel_i^*} - 1}{\log_2(i + 1)}$$
  - $nDCG@K = \frac{DCG@K}{IDCG@K}$ (nếu $IDCG@K = 0$ thì $nDCG@K = 0.0$).

### 4.2 Kết quả kiểm chứng Fixture Toán học (`--fixture-test`)
Kiểm nghiệm một trường hợp xếp hạng mẫu Top-5 với vector nhãn thực tế $retrieved = [1, 0, 2, 0, 1]$ và tập nhãn trong pool $pool = [2, 2, 1, 1, 0, 0]$:
- Tính tay:
  - $Hit@5 = 1.0$ (tài liệu $rel=2$ xuất hiện ở hạng 3)
  - $MRR@5 = \frac{1}{3} \approx 0.3333$
  - $DCG@5 = \frac{1}{\log_2 2} + 0 + \frac{3}{\log_2 4} + 0 + \frac{1}{\log_2 6} \approx 1.0 + 1.5 + 0.3869 = 2.8869$
  - $IDCG@5 = \frac{3}{\log_2 2} + \frac{3}{\log_2 3} + \frac{1}{\log_2 4} + \frac{1}{\log_2 5} + 0 \approx 3.0 + 1.8928 + 0.5 + 0.4307 = 5.8235$
  - $nDCG@5 = \frac{2.8869}{5.8235} \approx 0.4957$
- **Kết quả chạy lệnh `python scripts/run_evaluation.py --fixture-test`:**
  ```text
  [Fixture] Result: {'status': 'PASS', 'hit5': 1.0, 'mrr5': 0.3333, 'dcg5': 2.8869, 'idcg5': 5.8235, 'ndcg5': 0.4957}
  Metrics implementation matches hand-calculated values exactly. PASS.
  ```

---

## 5. Mức Độ Hoàn Thành Nhãn, Cổng Verified-Label Gate & Cơ Chế Official Evaluation

Thống kê độ phủ kiểm duyệt hiện tại trên 105 cặp ứng viên trong `data/eval/review_pool.csv`:
- **Số cặp đã kiểm duyệt (Verified):** **0 / 105 (0.0%)**
- **Số cặp đang chờ kiểm duyệt (Pending):** **105 / 105 (100.0%)**
- **Phân bố nhãn dự thảo trong pool:**
  - Nhãn đề xuất 2: 19 cặp
  - Nhãn đề xuất 1: 13 cặp
  - Nhãn đề xuất 0: 4 cặp
  - Ứng viên mới chưa có nhãn dự thảo (unjudged): 69 cặp

### 5.1 Phân biệt hai chế độ thực thi:
1. **Chế độ Xem trước (Preview Mode — mặc định không có cờ `--official`):**
   - Chạy bình thường ngay cả khi 100% nhãn đang ở trạng thái `pending`.
   - Xuất bảng so sánh thứ hạng Top-1 giữa Chroma và BM25, phân tích định tính câu hỏi unanswerable `q_030`.
   - Xuất artifact tái lập tại `results/g3/rankings_preview.json`.
2. **Chế độ Đánh giá Chính thức (Official Benchmark Mode — kích hoạt với cờ `--official`):**
   - Thực thi cơ chế kiểm duyệt nghiêm ngặt (Verified-Label Gate).
   - Yêu cầu cấu hình chuẩn hóa đúng `--top-k 5`.
   - Kiểm tra toàn bộ 12 test queries và mọi tài liệu Top-5 của cả hai phương pháp đều phải có mặt trong review pool với `review_status='verified'` và `verified_relevance` thuộc `{0, 1, 2}`.
   - Tuyệt đối không coi các tài liệu unjudged là điểm 0.
   - Nếu còn bất kỳ nhãn nào pending/missing: **từ chối tính toán**, không tạo hoặc ghi đè file `official_metrics.json`, và thoát với **exit code `1`**.
   - Khi toàn bộ nhãn đã được thẩm định đầy đủ: tính toán per-query metrics (Hit@5, MRR@5, nDCG@5), tính macro-average trên đúng 11 câu hỏi answerable (loại trừ `q_030`), và xuất kết quả chính thức tại `results/g3/official_metrics.json`.

---

## 6. Kết Quả 4 Kiểm Chứng Tối Thiểu (Minimal Verification)

### Check 1 — Metrics Correctness (Fixture Test): PASS
```powershell
python scripts/run_evaluation.py --fixture-test
```
- Vector Top-5 $retrieved = [1, 0, 2, 0, 1]$, $pool = [2, 2, 1, 1, 0, 0]$.
- Tính toán thực tế: $Hit@5 = 1.0$, $MRR@5 = 0.3333$, $DCG@5 = 2.8869$, $IDCG@5 = 5.8235$, $nDCG@5 = 0.4957$.
- Kết quả: Khớp chính xác với giá trị tính tay lý thuyết. **PASS**.

### Check 2 — Gate Refusal trên Review Pool Thật (105 Nhãn Pending): PASS
```powershell
python scripts/run_evaluation.py --official
```
- Hệ thống phát hiện 105 nhãn chưa được duyệt trong `data/eval/review_pool.csv`.
- Báo lỗi rõ ràng: `[VERIFIED-LABEL GATE REFUSAL] Official benchmark evaluation locked! Total gate violations found: 105`.
- Thoát với **exit code `1`** (`$LASTEXITCODE` = 1). Không tạo file official giả mạo. **PASS**.

### Check 3 — Kiểm chứng Nhánh Official-Success với Synthetic Verified CSV: PASS
Thực hiện trong thư mục tạm cô lập (sử dụng tệp CSV kiểm thử được đánh dấu rõ là *test-only*, không sửa đổi hay commit vào dữ liệu thật):
- Gán nhãn verified giả định cho toàn bộ 105 cặp ứng viên và chạy `run_evaluation.py --official`.
- Kết quả: Cổng gate mở thành công (**exit code `0`**).
- Tạo thành công tệp `official_metrics.json` chứa đầy đủ:
  - Per-query metrics (Hit@5, MRR@5, nDCG@5) cho cả Chroma và BM25.
  - Macro-average được tính toán chính xác trên **đúng 11 câu hỏi answerable** (loại trừ `q_030`).
  - Ghi nhận đầy đủ thông tin metadata, corpus SHA-256, dependencies và ghi chú giới hạn pooled judgments. **PASS**.

### Check 4 — Gate Refusal khi Thiếu Nhãn / Cặp Ứng Viên: PASS
Thực hiện trong thư mục tạm:
- Chuyển 1 nhãn bất kỳ trong tệp CSV kiểm thử về trạng thái `pending` và để trống relevance.
- Chạy `run_evaluation.py --official`.
- Kết quả: Gate ngay lập tức phát hiện vi phạm, từ chối tính toán, không xuất file `official_metrics.json`, và thoát với **exit code `1`**.
- Thử nghiệm với `--top-k 3`: Gate từ chối với thông báo `Official evaluation is strictly standardized at top_k=5. Received top_k=3` (**exit code `1`**). **PASS**.

---

## 7. Đối Sánh Thứ Hạng Xếp Hạng Xem Trước (Preview Rankings)

Chạy xem trước với `python scripts/run_evaluation.py` ghi nhận bảng so sánh Top-1 giữa Chroma và BM25 trên 12 queries:

| Query ID | Thể loại | Chroma Top-1 | BM25 Top-1 | Trạng thái Top-1 |
| :---: | :--- | :---: | :---: | :---: |
| `q_002` | `semantic_paraphrase` | `ml_015` | `ml_015` | **AGREE** |
| `q_003` | `semantic_paraphrase` | `algo_016` | `db_001` | **DIFFER** |
| `q_006` | `semantic_paraphrase` | `dl_017` | `dl_015` | **DIFFER** |
| `q_010` | `semantic_paraphrase` | `dl_016` | `dl_019` | **DIFFER** |
| `q_013` | `direct_technical` | `db_002` | `db_002` | **AGREE** |
| `q_016` | `direct_technical` | `db_016` | `db_016` | **AGREE** |
| `q_020` | `ambiguous_confusable` | `cv_014` | `cv_014` | **AGREE** |
| `q_021` | `ambiguous_confusable` | `net_004` | `net_006` | **DIFFER** |
| `q_024` | `ambiguous_confusable` | `net_015` | `net_015` | **AGREE** |
| `q_026` | `exact_match` | `cv_012` | `cv_012` | **AGREE** |
| `q_028` | `exact_match` | `db_011` | `db_011` | **AGREE** |
| `q_030` | `unanswerable` | `algo_014` | `ml_001` | **DIFFER (Unans)** |

- **Tỷ lệ đồng thuận Top-1:** **7 / 12 queries (58.3%)**
- **Các nhóm có độ đồng thuận cao:** Nhóm thuật ngữ trực tiếp (`direct_technical`), khớp chính xác (`exact_match`) và phân biệt khái niệm (`ambiguous_confusable`) đạt độ đồng thuận rất cao.
- **Các nhóm có độ phân kỳ:** Nhóm diễn đạt tự nhiên (`semantic_paraphrase`) ghi nhận sự khác biệt rõ rệt giữa cơ chế nhúng ngữ nghĩa dày đặc (dense) và từ vựng thưa thớt (sparse lexical).

---

## 8. Phân Tích Định Tính Câu Hỏi Unanswerable (`q_030`)

- **Nội dung câu hỏi:** `"thuật toán đồng thuận Proof of Stake và xử lý phân nhánh blockchain Ethereum"`
- **Kết quả tìm kiếm:**
  - Chroma Top-1: `algo_014` (Thuật toán Prim Tìm Cây khung Nhỏ nhất) với Cosine Distance `0.1513`
  - BM25 Top-1: `ml_001` (Học có giám sát và Học không giám sát) với BM25 Score `9.0677`
- **Quan sát phân tích:**
  - Cả hai hệ thống tìm kiếm đều **không có cơ chế từ chối (abstain)**; chúng bắt buộc phải trả về Top-K láng giềng gần nhất có trong cơ sở dữ liệu.
  - Việc vector nearest-neighbor trả về khoảng cách cosine 0.1513 không đồng nghĩa với việc mô hình đã tìm thấy câu trả lời đúng.
  - Vì vậy, câu hỏi `q_030` được tách riêng trong phân tích định tính và **không đưa vào mẫu số tính điểm trung bình của các chỉ số Hit/MRR/nDCG** để tránh làm sai lệch độ đo.

---

## 9. Các Giới Hạn Cần Lưu Ý

1. **Nhãn đang ở trạng thái Pending:** Toàn bộ 105 cặp ứng viên hiện đang chờ người kiểm duyệt mở tệp `data/eval/review_pool.csv` để đối soát và điền nhãn. Chưa có bảng xếp hạng hay điểm số chính thức nào được công bố tại G3.
2. **Đánh giá dựa trên Candidate Pool (Pooled Judgments):** Điểm $IDCG@5$ được tính toán trên các ứng viên có mặt trong pool kiểm duyệt. Điều này phản ánh tính tương đối của pool, chưa bảo đảm bao quát 100% mọi đoạn văn liên quan tiềm năng trong toàn bộ corpus.
3. **Quy mô tập mẫu:** Tập 12 test queries được cố định là mẫu thử nghiệm có kiểm soát cho đề tài Seminar, không đại diện cho toàn bộ hành vi tìm kiếm trên Internet.
4. **Tệp Artifacts tái lập:** Toàn bộ kết quả xếp hạng, thông số phiên bản thư viện và mã băm SHA-256 dữ liệu được lưu tự động tại `results/g3/rankings_preview.json`.

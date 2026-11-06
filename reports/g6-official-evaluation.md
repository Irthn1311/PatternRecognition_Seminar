# Báo Cáo Đánh Giá Thực Nghiệm Chính Thức (Official Retrieval Benchmark) — Task G6

- **Dự án:** PatternRecognition_Seminar
- **Nhiệm vụ:** TASK G6 — Human Relevance Review & Official Retrieval Evaluation
- **GitHub Issue:** [#11 — G6: Human-verified labels, official metrics, browser rehearsal and seminar package](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/11)
- **Thời gian thực hiện:** Tháng 11/2026
- **Môi trường thực nghiệm:** Windows 11 Home / Python 3.12.16 / CPU
- **Mã băm dữ liệu (Data SHA-256):**
  - Corpus (`data/corpus/documents.jsonl`): `a9633f3a20ffdc66199d196cb41ba31497baa10385ca854b8db1fac707536dd3`
  - Review Pool (`data/eval/review_pool.csv`): `2cf8d264267e817c3761c568f5a6538f71aa5a3ba23c7b2a632df3df95a024df`

---

## 1. Thiết Lập Đánh Giá Thực Nghiệm

Đánh giá chính thức chất lượng truy hồi được thực hiện trên tập dữ liệu chuẩn:
- **Kho tài liệu (Corpus):** 114 đoạn văn bản CNTT tiếng Việt (cân bằng 19 đoạn / 6 chủ đề).
- **Hệ thống truy xuất đối sánh:**
  1. **Chroma Semantic Search:** Vector embeddings 384 chiều từ mô hình `intfloat/multilingual-e5-small` (chuẩn hóa L2, khoảng cách Cosine trên chỉ mục HNSW).
  2. **Vietnamese BM25 Baseline:** Thuật toán BM25Okapi (`rank_bm25`) kết hợp bộ tách từ ghép tiếng Việt `pyvi.ViTokenizer`.
- **Tập truy vấn đánh giá cố định:** 12 test queries (`data/eval/evaluation_subset.json`), gồm 11 câu hỏi answerable và 1 câu hỏi unanswerable (`q_030`).
- **Tập nhãn thẩm định (Ground Truth):** 105 cặp query-document độc nhất trong `data/eval/review_pool.csv`, được người kiểm duyệt xác nhận 100% (22 nhãn 2, 18 nhãn 1, 65 nhãn 0).
- **Cơ chế Verified-Label Gate:** Toàn bộ quá trình tính toán được bảo vệ bởi cổng gate kiểm tra tính toàn vẹn mã băm (Double-Hash Gate), đảm bảo không có nhãn pending và loại bỏ hoàn toàn các số liệu giả định.

---

## 2. Kết Quả Benchmark Tổng Hợp (Macro-Averages)

Các chỉ số được tính toán chuẩn hóa trên **đúng 11 câu hỏi answerable** (câu hỏi unanswerable `q_030` được tách riêng trong phân tích định tính):

| Chỉ số đánh giá | Chroma (E5 Small) | BM25 (PyVi) | Chênh lệch (Chroma − BM25) | Ý nghĩa thống kê |
| :--- | :---: | :---: | :---: | :--- |
| **`Hit@5`** | **0.9091** (10/11) | **0.9091** (10/11) | **+0.0000** | Cả hai phương pháp đều đưa được ít nhất 1 tài liệu đạt chuẩn (rel=2) vào Top-5 ở 10 trên 11 câu hỏi (90.9%). |
| **`MRR@5`** | **0.7909** | **0.8182** | **−0.0273** | BM25 nhỉnh hơn nhẹ về tốc độ đưa tài liệu đúng lên vị trí số 1 ở các câu hỏi chứa từ khóa kỹ thuật khớp trực tiếp. |
| **`nDCG@5`** | **0.6928** | **0.7494** | **−0.0566** | BM25 đạt điểm tích lũy chiết khấu cao hơn ở các câu hỏi có sự hỗ trợ của từ vựng trùng khớp mạnh trong văn bản. |

---

## 3. Bảng Điểm Chi Tiết Từng Câu Hỏi (Per-Query Metrics)

| Query ID | Thể loại câu hỏi | C-Hit@5 | C-MRR@5 | C-nDCG@5 | B-Hit@5 | B-MRR@5 | B-nDCG@5 | Nhận xét so sánh |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `q_002` | `semantic_paraphrase` | 1.00 | 1.000 | 0.425 | 1.00 | 1.000 | 0.647 | Cả hai đều đưa `ml_015` lên Hạng 1; BM25 giữ thêm các tài liệu L1/L2 ở Top-5 tốt hơn. |
| `q_003` | `semantic_paraphrase` | **1.00** | **0.200** | **0.221** | 0.00 | 0.000 | 0.054 | **Chroma vượt trội:** Chroma tìm thấy B-Tree (`db_003`) ở Top-5; BM25 hoàn toàn bỏ lỡ B-Tree/Hash Index trong Top-5. |
| `q_006` | `semantic_paraphrase` | 0.00 | 0.000 | 0.208 | **1.00** | **0.500** | **0.580** | **BM25 tốt hơn:** BM25 đưa Multi-Head Attention (`dl_014`) lên Hạng 2; Chroma bị phân tán vào BERT/GPT. |
| `q_010` | `semantic_paraphrase` | 1.00 | 0.500 | 0.631 | **1.00** | **1.000** | **1.000** | **BM25 tốt hơn:** BM25 tìm trúng Lượng tử hóa (`dl_019`) ở ngay Hạng 1; Chroma xếp GPT ở Hạng 1 và `dl_019` ở Hạng 2. |
| `q_013` | `direct_technical` | 1.00 | 1.000 | 0.675 | 1.00 | 1.000 | 0.800 | Cả hai đều đưa `db_002` lên Hạng 1; BM25 gom thêm `db_018` (WAL) vào Top-5. |
| `q_016` | `direct_technical` | **1.00** | **1.000** | **0.981** | 1.00 | 1.000 | 0.926 | Cả hai đều đạt điểm gần như tuyệt đối; Chroma nhỉnh hơn ở thứ hạng tài liệu liên quan phụ (`db_017`). |
| `q_020` | `ambiguous_confusable` | 1.00 | 1.000 | 0.907 | 1.00 | 1.000 | 0.987 | Cả hai đều tìm thấy cả hai mặt đối lập (`cv_014` và `cv_013`) ở Top 1 và 2. |
| `q_021` | `ambiguous_confusable` | **1.00** | **1.000** | **0.914** | 1.00 | 0.500 | 0.696 | **Chroma vượt trội:** Chroma đưa cả TCP (`net_002`) và UDP (`net_004`) lên Top 1 và 2; BM25 bị trôi `net_006` lên đầu. |
| `q_024` | `ambiguous_confusable` | 1.00 | 1.000 | 0.933 | 1.00 | 1.000 | 1.000 | Cả hai đều tìm trúng tài liệu so sánh AES/RSA (`net_015`) ở ngay Hạng 1. |
| `q_026` | `exact_match` | 1.00 | 1.000 | 0.726 | 1.00 | 1.000 | 0.726 | **Tương đương tuyệt đối:** Cả hai phương pháp cho kết quả và thứ hạng Top-5 giống hệt nhau (`cv_012` ở Hạng 1). |
| `q_028` | `exact_match` | **1.00** | **1.000** | **1.000** | 1.00 | 1.000 | 0.826 | **Chroma vượt trội:** Chroma đạt điểm nDCG hoàn hảo (1.000); BM25 bị xen lẫn các tài liệu Deep Learning do từ khóa "PostgreSQL". |

---

## 4. Phân Tích Chuyên Sâu Các Nhóm Truy Vấn

### 4.1 Những trường hợp Chroma (Semantic Search) vượt trội
1. **Truy vấn diễn đạt trừu tượng không trùng từ khóa (`q_003`):**
   - Câu hỏi: *"giải pháp tăng tốc độ tìm kiếm bản ghi trong bảng cơ sở dữ liệu hàng triệu dòng"*.
   - Người dùng hỏi về "tăng tốc tìm kiếm", không nhắc đến thuật ngữ "chỉ mục" hay "B-Tree".
   - **Chroma:** Nắm bắt được ngữ nghĩa khái niệm lưu trữ và đưa tài liệu chỉ mục B-Tree (`db_003`) vào Top-5 (Hit=1.0).
   - **BM25:** Bị phụ thuộc vào các từ khóa xuất hiện nhiều ("bảng", "cơ sở dữ liệu", "bản ghi"), kéo theo các bài về chuẩn hóa RDBMS (`db_001`), khóa chính (`db_005`), MVCC (`db_011`), MongoDB (`db_013`) và Sharding (`db_007`), **hoàn toàn bỏ lỡ tài liệu chỉ mục B-Tree trong Top-5 (Hit=0.0)**.
2. **Truy vấn so sánh hai khái niệm đối lập (`q_021`):**
   - Câu hỏi: *"so sánh giao thức truyền tải TCP và UDP trong mạng máy tính"*.
   - **Chroma:** Đưa cả tài liệu UDP (`net_004`, Hạng 1) và TCP (`net_002`, Hạng 2) lên đầu bảng, đạt nDCG = 0.914.
   - **BM25:** Bị cụm từ "giao thức" kéo bài HTTP/2 và HTTP/3 (`net_006`) lên Hạng 1, đẩy TCP xuống Hạng 5 (nDCG = 0.696).

### 4.2 Những trường hợp BM25 (Từ khóa) chiếm ưu thế
1. **Truy vấn có từ khóa kỹ thuật định danh mạnh (`q_006`, `q_010`):**
   - Trên `q_010` (*"kỹ thuật tối ưu bộ nhớ để chạy các mô hình AI ngôn ngữ lớn"*): BM25 nhận diện chính xác tài liệu Lượng tử hóa (`dl_019`) ở vị trí Hạng 1 nhờ tần suất từ vựng, đạt nDCG hoàn hảo 1.000. Trong khi đó, Chroma xếp mô hình GPT (`dl_016`) ở Hạng 1 do độ tương đồng ngữ nghĩa của cụm "mô hình ngôn ngữ lớn", đẩy `dl_019` xuống Hạng 2.
   - Trên `q_006` (*"dịch máy hiểu liên hệ từ xa trong câu dài"*): BM25 đưa kiến trúc Transformer Multi-Head (`dl_014`) lên Hạng 2 (Hit=1.0), trong khi Chroma bị hút mạnh về phía mô hình BERT (`dl_017`) ở Hạng 1 (tài liệu BERT chỉ đạt rel=1).

### 4.3 Những trường hợp hai phương pháp tương đương
- Các câu hỏi thuộc nhóm khớp thuật ngữ chính xác (`exact_match`) như `q_026` (*"chỉ số IoU và thuật toán Non-Maximum Suppression NMS"*) hoặc đối chiếu mã hóa `q_024` (*"so sánh mã hóa đối xứng AES và mã hóa bất đối xứng RSA"*). Cả hai phương pháp đều đưa tài liệu đích lên Hạng 1 với độ tin cậy tuyệt đối.

---

## 5. Phân Tích Định Tính Truy Vấn Ngoài Phạm Vi (`q_030`)

- **Câu hỏi (`q_030`):** *"thuật toán đồng thuận Proof of Stake và xử lý phân nhánh blockchain Ethereum"*
- **Kết quả trả về:**
  - Chroma Top-1: `algo_014` (Thuật toán Prim) với khoảng cách Cosine `0.1513`.
  - BM25 Top-1: `ml_001` (Học có giám sát và không giám sát) với BM25 Score `9.0677`.
- **Quan sát phân tích khoa học:**
  - Cả Chroma và BM25 đều **không có cơ chế tự từ chối (abstain)**. Khi nhận câu hỏi ngoài phạm vi, hệ thống bắt buộc phải trả về các tài liệu gần nhất trong kho dữ liệu.
  - Việc vector láng giềng gần nhất có khoảng cách tương đối nhỏ không đồng nghĩa với việc câu trả lời là đúng.
  - Do đó, việc loại trừ `q_030` khỏi mẫu số tính điểm trung bình (chỉ tính trên 11 câu answerable) là quyết định phương pháp luận bắt buộc để bảo vệ tính chính xác của các chỉ số hiệu năng IR.

---

## 6. Giới Hạn Nghiên Cứu & Kết Luận Khoa Học

1. **Giới hạn quy mô dữ liệu:** Đánh giá được thực hiện trên 11 câu hỏi answerable và 114 đoạn tài liệu. Kết quả phản ánh chính xác hành vi của hai phương pháp trên kho học liệu chuyên đề này, không thể khái quát thành kết luận phổ quát cho môi trường web quy mô hàng triệu tài liệu.
2. **Giới hạn của Candidate Pool (Pooled Judgments):** Điểm $IDCG@5$ được tính toán dựa trên các ứng viên có mặt trong pool thẩm định. Điều này phản ánh tính tương đối của tập ứng viên được chọn, chưa bảo đảm vét cạn 100% mọi đoạn văn bản tiềm năng trong toàn bộ kho dữ liệu.
3. **Kết luận khoa học:**
   - Cả Chroma Semantic Search và BM25 Lexical Search đều đạt hiệu năng thực tế rất ấn tượng trên tập dữ liệu học liệu CNTT tiếng Việt (cùng đạt Hit@5 = 90.9%).
   - Hai phương pháp mang **tính chất bổ trợ sâu sắc**: Tìm kiếm vector xuất sắc trong việc nắm bắt ý định khái niệm trừu tượng, trong khi tìm kiếm từ khóa cực kỳ tin cậy khi câu hỏi chứa các thuật ngữ định danh chính xác.
   - Kết quả này gợi mở hướng phát triển hệ thống tìm kiếm lai (Hybrid Search: kết hợp Reciprocal Rank Fusion giữa Chroma và BM25) trong tương lai.

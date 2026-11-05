# Báo Cáo Kỹ Thuật G1: Xây Dựng Corpus Học Liệu CNTT Tiếng Việt & Bộ Dữ Liệu Đánh Giá Truy Hồi

- **Dự án:** PatternRecognition_Seminar
- **Nhiệm vụ:** TASK G1 — Vietnamese IT Corpus, Evaluation Queries & Draft Qrels
- **GitHub Issue:** [#3 — G1: Curate Vietnamese IT corpus, evaluation queries and draft qrels](https://github.com/Irthn1311/PatternRecognition_Seminar/issues/3)
- **Thời gian thực hiện:** Tháng 11/2026
- **Môi trường:** Windows 11 Home / Python 3.12.16

---

## 1. Tổng quan Số Liệu Thực Tế

Bộ dữ liệu G1 đã hoàn thành xây dựng với các số liệu thực tế đo đạc qua script kiểm tra tự động:

| Hạng mục dữ liệu | Số lượng thực tế | Quy cách lưu trữ | Ghi chú |
| :--- | :---: | :--- | :--- |
| **Passages tài liệu (Corpus)** | **114** | `data/corpus/documents.jsonl` | Cân bằng 19 đoạn / nhóm trên 6 chủ đề |
| **Registry nguồn (Sources)** | **13** | `data/corpus/sources.jsonl` | 12 nguồn adapted có URL chính thức, 1 self_composed |
| **Câu truy vấn đánh giá (Queries)** | **32** | `data/eval/queries.jsonl` | 5 thể loại, phân chia 9 dev / 23 test |
| **Nhãn tương quan dự thảo (Qrels)** | **86** | `data/eval/qrels_draft.jsonl` | Thang điểm 0, 1, 2; trạng thái `proposed` |
| **Bảng kiểm duyệt trực quan** | **86 dòng** | `data/eval/label_review.csv` | Định dạng CSV UTF-8 with BOM tiện rà trên Excel |
| **Bộ kiểm tra toàn vẹn** | **1 script** | `scripts/validate_dataset.py` | 100% Python Standard Library, 0 dependencies |

---

## 2. Phân Bố Chi Tiết Dữ Liệu

### 2.1 Phân bố 114 passages theo 6 nhóm chủ đề
Mỗi chủ đề chứa đúng 19 đoạn văn bản độc lập, có ngữ cảnh trọn vẹn, độ dài từ 350 đến 700 ký tự (phù hợp giới hạn 512 token của mô hình E5):
- **Machine Learning:** 19 đoạn (`ml_001` - `ml_019`)
- **Deep Learning:** 19 đoạn (`dl_001` - `dl_019`)
- **Computer Vision:** 19 đoạn (`cv_001` - `cv_019`)
- **Database:** 19 đoạn (`db_001` - `db_019`)
- **Algorithms:** 19 đoạn (`algo_001` - `algo_019`)
- **Computer Networks / Security:** 19 đoạn (`net_001` - `net_019`)

### 2.2 Phân bố 32 câu truy vấn theo thể loại và tập chia
- **Thể loại truy vấn (Category):**
  - `semantic_paraphrase`: 11 câu (34.4%) — Dùng lối diễn đạt tự nhiên đời thường, từ đồng nghĩa hoặc câu hỏi của sinh viên.
  - `direct_technical`: 7 câu (21.9%) — Câu hỏi rõ ràng về nguyên lý kỹ thuật cốt lõi.
  - `ambiguous_confusable`: 6 câu (18.7%) — So sánh và phân biệt các cặp khái niệm dễ nhầm lẫn.
  - `exact_match`: 4 câu (12.5%) — Chứa các từ khóa viết tắt và thuật ngữ đặc thù mà BM25 có ưu thế.
  - `unanswerable`: 4 câu (12.5%) — Các câu hỏi nằm ngoài phạm vi tài liệu (Quantum Computing, Blockchain, LLVM, Saga Pattern).
- **Phân chia tập (Split):**
  - `dev`: 9 câu (28.1%) — Dành cho quá trình tinh chỉnh mô hình và kiểm tra sơ bộ.
  - `test`: 23 câu (71.9%) — Dành cho đánh giá benchmark độc lập cuối cùng.

### 2.3 Phân bố 86 nhãn tương quan dự thảo (Draft Qrels)
- **Điểm 2 (Directly relevant):** 43 nhãn (50.0%) — Trả lời trực tiếp nội dung câu hỏi.
- **Điểm 1 (Partially relevant):** 34 nhãn (39.5%) — Cung cấp ngữ cảnh nền tảng hoặc kiến thức liên đới.
- **Điểm 0 (Irrelevant / Distractor):** 9 nhãn (10.5%) — Đoạn văn bản gây nhiễu được đưa vào để kiểm tra khả năng phân biệt âm tính.

---

## 3. Nguồn Gốc và Phân Loại Provenance

Dữ liệu tuân thủ nghiêm ngặt nguyên tắc sở hữu trí tuệ và có thể truy vết:
- **`adapted` (12 nguồn):** Diễn giải và tổng hợp súc tích bằng tiếng Việt từ các tài liệu chuẩn kỹ thuật, bài giảng mở và tài liệu chính thức:
  - Scikit-learn Consortium Documentation (BSD-3-Clause)
  - Machine Learning Cơ Bản (Vũ Hữu Tiệp - Open Educational Blog)
  - Deep Learning Book (Ian Goodfellow, Yoshua Bengio, Aaron Courville - MIT Press)
  - PyTorch Official Documentation (Linux Foundation / Modified BSD)
  - Stanford CS231n: Convolutional Neural Networks for Visual Recognition
  - OpenCV Official Documentation (Apache-2.0)
  - PostgreSQL Official Documentation (PostgreSQL License)
  - SQLite Architecture and Query Planner (Public Domain)
  - Open Data Structures (Pat Morin, Carleton University - CC-BY-2.5-CA)
  - GeeksforGeeks Algorithms Reference
  - IETF RFC 9293 (Transmission Control Protocol Specification - Open Standard)
  - OWASP Top Ten Security Risks (CC-BY-SA 4.0)
- **`self_composed` (1 nguồn):** Nhóm nghiên cứu Seminar Nhận diện Mẫu (SGU) tự biên soạn tổng hợp các phần đánh đổi kỹ thuật và đối sánh chuyên sâu.
- **Không có dữ liệu sao chép nguyên văn vi phạm bản quyền (`quoted`).**

---

## 4. Minh Họa Ví Dụ Dữ Liệu

### 4.1 Ví dụ Passage (Computer Vision - `cv_006`)
```json
{
  "id": "cv_006",
  "source_id": "src_cv_01",
  "title": "Mạng ResNet và Kết nối Tắt Skip Connections",
  "text": "Mạng thặng dư ResNet (Residual Networks, He et al., 2015) giải quyết vấn đề suy thoái hiệu năng khi xây dựng các mạng nơ-ron cực sâu. Bằng cách bổ sung các kết nối tắt (Skip Connections hay Identity Shortcut), mạng học ánh xạ thặng dư F(x) = H(x) - x thay vì học trực tiếp hàm mục tiêu H(x). Gradient có thể chạy thẳng không suy giảm qua các kết nối tắt trong quá trình lan truyền ngược, cho phép huấn luyện thành công các mạng vượt quá 152 tầng.",
  "topic": "Computer Vision",
  "language": "vi",
  "chunk_index": 0,
  "source_url": "https://cs231n.github.io/",
  "provenance": "adapted"
}
```

### 4.2 Ví dụ Câu hỏi Paraphrase (`q_002`) và Qrels đối sánh
- **Câu hỏi:** `"cách giải quyết khi mô hình học máy thuộc lòng dữ liệu huấn luyện và dự đoán kém trên thực tế"` (Dùng cụm từ "thuộc lòng dữ liệu huấn luyện", không dùng từ "overfitting").
- **Nhãn gán tương quan:**
  - `ml_015` (Hiện tượng Quá khớp và các Biện pháp khắc phục): **Điểm 2** (Giải đáp trực tiếp hiện tượng và các giải pháp).
  - `ml_004` (Điều chuẩn L1 Lasso): **Điểm 2** (Phương pháp điều chuẩn giải quyết vấn đề).
  - `ml_005` (Điều chuẩn L2 Ridge): **Điểm 2** (Phương pháp điều chuẩn giải quyết vấn đề).
  - `ml_014` (Đánh đổi Độ chệch và Phương sai): **Điểm 1** (Giải thích nguồn gốc phương sai cao).
  - `dl_007` (Điều chuẩn Dropout): **Điểm 1** (Giải pháp tương tự trong Deep Learning).

---

## 5. Kết Quả Kiểm Tra Tính Toàn Vẹn Thực Tế

Toàn bộ các ràng buộc kỹ thuật đã được xác thực thông qua lệnh thực thi tự động:

```powershell
python scripts/validate_dataset.py
```

### Log đầu ra thực tế của validator:
```text
============================================================
DATASET VALIDATION SUMMARY & STATISTICS
============================================================

[Corpus Documents]: 114 passages
  - Algorithms: 19
  - Computer Networks / Security: 19
  - Computer Vision: 19
  - Database: 19
  - Deep Learning: 19
  - Machine Learning: 19

[Corpus Sources]: 13 sources
  - adapted: 12
  - self_composed: 1

[Evaluation Queries]: 32 queries
  - Category 'ambiguous_confusable': 6
  - Category 'direct_technical': 7
  - Category 'exact_match': 4
  - Category 'semantic_paraphrase': 11
  - Category 'unanswerable': 4
  - Split 'dev': 9
  - Split 'test': 23
  - Unanswerable queries: 4

[Draft Qrels]: 86 relevance judgments
  - Score 0 (Irrelevant/Negative): 9
  - Score 1 (Partially relevant): 34
  - Score 2 (Directly relevant): 43

------------------------------------------------------------
ERRORS:   0
WARNINGS: 0
------------------------------------------------------------

>>> VALIDATION RESULT: PASS (All checks passed successfully) <<<
```

---

## 6. Rà Soát Thủ Công Một Số Cặp Tiêu Biểu & Giới Hạn Cần Lưu Ý

### 6.1 Ghi nhận từ lượt rà soát thủ công:
1. **Các câu hỏi nhóm `ambiguous_confusable` (như `q_019`, `q_020`, `q_021`, `q_022`):**
   - Đều có ít nhất 2 passages đạt điểm 2 tương ứng với hai mặt của khái niệm cần phân biệt (ví dụ `q_022` hỏi so sánh B-Tree và Hash Index thì `db_003` và `db_004` đều đạt điểm 2).
   - Điều này tạo ra thách thức thực tế cho mô hình tìm kiếm phải hồi xuất được cả hai góc nhìn để người dùng có cái nhìn trọn vẹn.
2. **Các câu hỏi nhóm `unanswerable` (`q_029` - `q_032`):**
   - Được kiểm tra kỹ lưỡng để đảm bảo toàn bộ 114 đoạn tài liệu trong corpus hoàn toàn không chứa nội dung trả lời (như máy tính lượng tử, Proof of Stake, LLVM, Saga Pattern).
   - Đã gán nhãn 0 cho một số đoạn văn bản gần gũi về mặt từ khóa tổng quan (distractors) để kiểm tra khả năng không bị ảo giác điểm cao của hệ thống IR.

### 6.2 Các giới hạn của tập dữ liệu:
1. **Trạng thái nhãn là Dự thảo (`label_status: proposed`):**
   - Các nhãn relevance hiện tại do AI đề xuất dựa trên phân tích nội dung, **chưa phải là ground truth được con người chứng nhận (verified)**.
   - Thầy cô và người kiểm duyệt cần rà soát qua tệp `data/eval/label_review.csv` trước khi sử dụng kết quả này để báo cáo các chỉ số IR chính thức (MRR, NDCG@K).
2. **Quy mô tập dữ liệu (114 đoạn văn):**
   - Quy mô này phù hợp hoàn hảo cho việc đối sánh kỹ thuật và tìm kiếm ngữ nghĩa trong phạm vi Seminar chuyên đề, đảm bảo chất lượng từng đoạn cao và không bị loãng. Tuy nhiên, kích thước này chưa phản ánh toàn diện hiện tượng nhiễu quy mô lớn (large-scale web noise) của các tập dữ liệu như MS MARCO.
3. **Quy ước Unjudged:**
   - Trong 114 tài liệu, những tài liệu không được gán nhãn cho một query cụ thể phải được coi là *unjudged*, không tự động xem là hoàn toàn không liên quan (tránh làm sai lệch Recall@K nếu sau này phát hiện thêm các góc độ trả lời mới).

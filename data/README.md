# Vietnamese IT Corpus & Evaluation Benchmark Dataset (Task G1)

Thư mục này chứa tập dữ liệu học liệu Công nghệ Thông tin tiếng Việt, tập câu hỏi đánh giá và nhãn tương quan dự thảo phục vụ nghiên cứu Semantic Search và BM25 cho Seminar Nhận diện Mẫu (Pattern Recognition).

---

## 1. Cấu trúc thư mục dữ liệu

```text
data/
├── corpus/
│   ├── sources.jsonl            # Registry nguồn tài liệu và giấy phép (13 nguồn)
│   └── documents.jsonl          # 114 passages tiếng Việt thuộc 6 chủ đề
├── eval/
│   ├── queries.jsonl            # 32 câu truy vấn đánh giá (dev/test split)
│   ├── qrels_draft.jsonl        # 86 nhãn tương quan dự thảo (0, 1, 2)
│   ├── label_review.csv         # Bảng tổng hợp hỗ trợ người kiểm duyệt rà nhãn G1
│   ├── evaluation_subset.json   # 12 câu truy vấn test cố định phục vụ benchmark G3
│   └── review_pool.csv          # 105 cặp ứng viên pooled (Chroma + BM25 + Draft) chờ kiểm duyệt G3
├── smoke_documents.json         # Tập 16 tài liệu thử nghiệm G0 (giữ nguyên độc lập)
└── README.md                    # Tài liệu đặc tả cấu trúc và quy ước dữ liệu
```

---

## 2. Đặc tả Schema

### 2.1 Registry nguồn: `data/corpus/sources.jsonl`
Mỗi dòng là một đối tượng JSON mô tả một nguồn tài liệu tham khảo:
```json
{
  "source_id": "src_ml_01",
  "name": "Scikit-learn User Guide (Supervised & Unsupervised Learning)",
  "author_or_org": "Scikit-learn Consortium",
  "source_url": "https://scikit-learn.org/stable/user_guide.html",
  "license": "BSD-3-Clause",
  "provenance": "adapted",
  "notes": "Diễn giải các nguyên lý cơ bản về giải thuật học có giám sát và không giám sát"
}
```

### 2.2 Tập tài liệu: `data/corpus/documents.jsonl`
Gồm 114 đoạn văn bản (passages), cân bằng 19 đoạn cho mỗi nhóm trong 6 nhóm chủ đề:
- `Machine Learning`
- `Deep Learning`
- `Computer Vision`
- `Database`
- `Algorithms`
- `Computer Networks / Security`

Mỗi bản ghi tuân thủ cấu trúc:
```json
{
  "id": "cv_001",
  "source_id": "src_cv_01",
  "title": "Tầng Tích chập Convolutional Layer và Bộ lọc Kernel",
  "text": "Tầng tích chập (Convolutional Layer) là khối kiến trúc trung tâm của mạng CNN...",
  "topic": "Computer Vision",
  "language": "vi",
  "chunk_index": 0,
  "source_url": "https://cs231n.github.io/",
  "provenance": "adapted"
}
```

### 2.3 Tập câu truy vấn: `data/eval/queries.jsonl`
Gồm 32 câu hỏi tiếng Việt được phân bổ theo 5 thể loại nhằm kiểm tra các khía cạnh khác nhau của hệ thống tìm kiếm:
- `semantic_paraphrase` (11 câu): Diễn đạt tự nhiên bằng từ vựng khác biệt, kiểm tra khả năng bắt ý ngữ nghĩa.
- `direct_technical` (7 câu): Truy vấn trực tiếp các khái niệm kỹ thuật cốt lõi.
- `ambiguous_confusable` (6 câu): So sánh hoặc phân biệt các khái niệm dễ gây nhầm lẫn (như L1 vs L2, Semantic vs Instance Segmentation, TCP vs UDP).
- `exact_match` (4 câu): Chứa thuật ngữ chuyên sâu, từ khóa viết tắt mà phương pháp từ khóa (BM25) có thế mạnh.
- `unanswerable` (4 câu): Nằm ngoài phạm vi kiến thức của corpus, có cờ `"is_unanswerable": true`.

Phân chia tập:
- `dev`: 9 câu hỏi (phục vụ tinh chỉnh và thử nghiệm sớm).
- `test`: 23 câu hỏi (phục vụ đánh giá benchmark chính thức).

```json
{
  "query_id": "q_001",
  "query": "làm thế nào để máy tính tự động tìm ra các vật thể và vị trí của chúng trong một bức ảnh?",
  "category": "semantic_paraphrase",
  "split": "dev",
  "is_unanswerable": false
}
```

### 2.4 Nhãn tương quan dự thảo: `data/eval/qrels_draft.jsonl`
Gồm 86 nhãn tương quan do AI đề xuất ban đầu (Draft Labels):
```json
{
  "query_id": "q_001",
  "doc_id": "cv_010",
  "relevance": 2,
  "label_status": "proposed",
  "note": "Faster R-CNN định vị hộp bao và phân loại vật thể qua mạng đề xuất vùng RPN"
}
```

**Thang điểm Relevance:**
- `2` (Directly relevant): Đoạn văn trả lời trực tiếp hoặc giải thích trúng trọng tâm câu hỏi.
- `1` (Partially relevant): Đoạn văn đề cập đến chủ đề liên quan hoặc ngữ cảnh nền tảng, nhưng không giải đáp trực tiếp.
- `0` (Irrelevant / Distractor): Đoạn văn gây nhiễu, dùng kiểm chứng khả năng phân biệt của mô hình.

*Lưu ý quan trọng:* Toàn bộ các cặp không xuất hiện trong file này được hiểu là **unjudged** (chưa đánh giá), tuyệt đối không tự động coi là 0 (irrelevant).

---

## 3. Quy trình Kiểm duyệt Nhãn (Human Review)

Để hỗ trợ người kiểm duyệt (Reviewer) xem xét và xác nhận nhãn mà không cần viết mã hay dùng công cụ phức tạp, tệp `data/eval/label_review.csv` được tạo sẵn (mã hóa UTF-8 with BOM, mở trực tiếp bằng Microsoft Excel hoặc Google Sheets).

Tệp chứa đầy đủ các trường: `query_id`, `query_text`, `category`, `split`, `doc_id`, `doc_title`, `doc_text`, `source_url`, `relevance`, `label_status`, `note`.

Quy trình kiểm duyệt:
1. Mở `data/eval/label_review.csv`.
2. Đọc nội dung câu hỏi `query_text` và toàn văn đoạn trích `doc_text` (kèm `source_url` tham chiếu nếu cần).
3. Kiểm tra cột `relevance` (0, 1, 2) và `note`.
4. Nếu đồng ý, đổi `label_status` từ `proposed` thành `verified`. Nếu không đồng ý, sửa lại giá trị `relevance` phù hợp.
5. **Lưu ý quan trọng về tính đồng bộ:** Việc chỉnh sửa trong tệp CSV **không tự động cập nhật** ngược lại tệp `data/eval/qrels_draft.jsonl`. Trạng thái nhãn chính thức (`verified`) chỉ được đồng bộ và phê duyệt thông qua bước đối soát có kiểm soát trong giai đoạn đánh giá thực nghiệm (G2/G3). Trong phạm vi G1, toàn bộ nhãn trong `qrels_draft.jsonl` được giữ ở trạng thái dự thảo (`proposed`) và không xây dựng annotation UI hay công cụ đồng bộ phức tạp.

---

## 4. Hướng dẫn Kiểm tra Tính Toàn vẹn Dữ liệu (Validation)

Script `scripts/validate_dataset.py` được viết hoàn toàn bằng thư viện tiêu chuẩn của Python (Python standard library), không phụ thuộc vào bất kỳ thư viện bên ngoài nào:

```powershell
python scripts/validate_dataset.py
```

Bộ validator kiểm tra toàn diện:
- Cú pháp và định dạng JSONL hợp lệ.
- Tính đầy đủ của các trường bắt buộc theo schema.
- Tính duy nhất của các mã định danh (`id`, `source_id`, `query_id`).
- Tính hợp lệ của liên kết khóa ngoại (`source_id` trong documents phải tồn tại trong sources; `doc_id` và `query_id` trong qrels phải tồn tại trong corpus).
- Không có đoạn văn hoặc câu hỏi rỗng hoặc trùng lặp nguyên văn.
- Tính nhất quán của cờ `is_unanswerable` (không gán nhãn dương tính cho câu hỏi out-of-scope).

---

## 5. Tập Đánh Giá Cố Định & Candidate Pool G3

### 5.1 Tập truy vấn đánh giá cố định: `data/eval/evaluation_subset.json`
Được cố định trước khi thực hiện đánh giá xếp hạng nhằm đảm bảo tính khách quan (tránh thiên kiến chọn câu hỏi):
- Gồm đúng **12 test queries** đại diện cho 5 thể loại:
  - 4 câu `semantic_paraphrase`: `q_002`, `q_003`, `q_006`, `q_010`
  - 2 câu `direct_technical`: `q_013`, `q_016`
  - 3 câu `ambiguous_confusable`: `q_020`, `q_021`, `q_024`
  - 2 câu `exact_match`: `q_026`, `q_028`
  - 1 câu `unanswerable`: `q_030`
- Toàn bộ đều thuộc `split: test`. Các câu `dev` được giữ riêng để phục vụ debug.

### 5.2 Bảng Candidate Review Pool: `data/eval/review_pool.csv`
Tạo bởi lệnh `python scripts/prepare_eval_pool.py`, hợp nhất:
1. Top-5 ứng viên của Chroma Semantic Search.
2. Top-5 ứng viên của Vietnamese BM25.
3. Các ứng viên đã có trong nhãn dự thảo `qrels_draft.jsonl`.

Tổng cộng có **105 cặp query-document** độc nhất cần kiểm duyệt.
Cấu trúc bảng (UTF-8 with BOM):
- `query_id`, `query_text`, `category`, `split`, `is_unanswerable`
- `doc_id`, `doc_title`, `doc_text`
- `chroma_rank`: Thứ hạng Top-5 từ Chroma (nếu có)
- `bm25_rank`: Thứ hạng Top-5 từ BM25 (nếu có)
- `proposed_relevance`: Nhãn dự thảo ban đầu (0/1/2 hoặc trống nếu chưa đánh giá)
- `proposed_note`: Ghi chú từ nhãn dự thảo
- `verified_relevance`: Điểm số tương quan do người kiểm duyệt thẩm định và xác nhận {0, 1, 2}
- `review_status`: Trạng thái kiểm duyệt (đã hoàn thành thẩm định 105/105 cặp sang `verified`)

### 5.3 Cơ chế Verified-Label Gate & Benchmark Chính Thức (Task G6)
Để xuất báo cáo benchmark chính thức (`python scripts/run_evaluation.py --official`), hệ thống yêu cầu:
- Toàn bộ các dòng trong pool của các câu hỏi được đánh giá phải có `verified_relevance` thuộc `{0, 1, 2}` và `review_status == 'verified'`.
- Kiểm tra mã băm SHA-256 đối soát đồng thời cả `corpus_sha256` và `review_pool_sha256` để loại trừ hoàn toàn dữ liệu lỗi thời.
- Câu hỏi unanswerable (`q_030`) được phân tích định tính riêng biệt, không đưa vào trung bình của các câu hỏi answerable.
- **Trạng thái hiện tại (Task G6):** Toàn bộ 105 cặp ứng viên đã được người kiểm duyệt hoàn tất thẩm định (22 nhãn 2, 18 nhãn 1, 65 nhãn 0), cổng gate mở thành công và lưu kết quả tại `results/g3/official_metrics.json`.

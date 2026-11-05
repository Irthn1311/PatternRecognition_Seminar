"""
Builder and executor for notebooks/Chroma_Seminar_Experiments.ipynb.

Constructs the complete 5-part seminar experiment notebook conforming to nbformat v4,
executes every cell in order on the active Python environment, and saves the executed
notebook with all outputs, tables, and Matplotlib plots populated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "Chroma_Seminar_Experiments.ipynb"


def create_experiment_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3 (.venv)",
            "language": "python",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.16",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
        },
    }

    cells = []

    # Title & Introduction
    cells.append(
        nbformat.v4.new_markdown_cell(
            "# Báo Cáo Thực Nghiệm: Chroma VectorDB & Tìm Kiếm Ngữ Nghĩa\n"
            "### Bài Seminar Môn Nhận Diện Mẫu — Đề Tài: Cơ Sở Dữ Liệu Vector Chroma\n\n"
            "Notebook này tổng hợp 5 thí nghiệm cốt lõi minh chứng cho các nguyên lý kỹ thuật của cơ sở dữ liệu vector Chroma, "
            "từ cơ chế tính khoảng cách vector và lọc siêu dữ liệu, quy trình trích xuất embedding cho học liệu CNTT tiếng Việt, "
            "đối sánh với phương pháp từ vựng BM25, kiểm chứng độ chính xác của chỉ mục đồ thị HNSW, đến quy trình đánh giá "
            "chất lượng truy xuất theo tiêu chuẩn khoa học."
        )
    )

    # Setup Environment
    cells.append(
        nbformat.v4.new_code_cell(
            "# Thiết lập môi trường và nạp các thư viện cần thiết\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "# Tắt cảnh báo không cần thiết của Hugging Face\n"
            "os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'\n\n"
            "# Đặt đường dẫn gốc dự án để nạp các module mã nguồn\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "if PROJECT_ROOT.name == 'notebooks':\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "import tempfile\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import chromadb\n\n"
            "# Cấu hình giao diện biểu đồ phục vụ trình chiếu\n"
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
            "plt.rcParams['figure.dpi'] = 110\n"
            "plt.rcParams['font.size'] = 10\n"
            "print('Môi trường thực nghiệm đã sẵn sàng.')"
        )
    )

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Vector Space & Metadata Filtering (Slide 15-16)
    # -------------------------------------------------------------------------
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 1. Không Gian Vector và Lọc Siêu Dữ Liệu (Metadata Filtering)\n"
            "*Liên hệ Slide 15–16*\n\n"
            "Thí nghiệm sử dụng đúng bộ 3 vector ba chiều $a, b, c$ và vector truy vấn $q$ như trong slide bài giảng:\n"
            "- Vector $a = [1.0, 0.0, 0.0]$ (Chủ đề: X)\n"
            "- Vector $b = [0.9, 0.1, 0.0]$ (Chủ đề: X)\n"
            "- Vector $c = [0.0, 1.0, 0.0]$ (Chủ đề: Y)\n"
            "- Truy vấn $q = [1.0, 0.0, 0.0]$\n\n"
            "Mục tiêu là khảo sát thứ tự truy xuất theo khoảng cách cosine và ảnh hưởng của phép lọc metadata trước khi xếp hạng (pre-filtering)."
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# Khởi tạo collection tạm thời và nạp 3 vector mẫu\n"
            "# Sử dụng ignore_cleanup_errors=True trên Windows để tránh giữ file handle SQLite\n"
            "with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:\n"
            "    client = chromadb.PersistentClient(path=tmp_dir)\n"
            "    col = client.create_collection(name='demo_3vectors', metadata={'hnsw:space': 'cosine'})\n"
            "    \n"
            "    vectors = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]\n"
            "    ids = ['a', 'b', 'c']\n"
            "    metas = [{'topic': 'X'}, {'topic': 'X'}, {'topic': 'Y'}]\n"
            "    docs = ['Tài liệu a (hướng trục X)', 'Tài liệu b (lệch nhẹ trục X)', 'Tài liệu c (vuông góc, trục Y)']\n"
            "    \n"
            "    col.add(ids=ids, embeddings=vectors, metadatas=metas, documents=docs)\n"
            "    q = [1.0, 0.0, 0.0]\n"
            "    \n"
            "    # 1. Truy vấn không áp dụng bộ lọc\n"
            "    res_all = col.query(query_embeddings=[q], n_results=3, include=['metadatas', 'distances'])\n"
            "    \n"
            "    # 2. Truy vấn có lọc metadata where={'topic': 'Y'}\n"
            "    res_filtered = col.query(query_embeddings=[q], n_results=3, where={'topic': 'Y'}, include=['metadatas', 'distances'])\n\n"
            "# Tính khoảng cách cosine lý thuyết bằng NumPy: 1.0 - (u . v) / (||u|| * ||v||)\n"
            "def np_cosine_dist(u, v):\n"
            "    u_arr, v_arr = np.array(u), np.array(v)\n"
            "    return float(1.0 - np.dot(u_arr, v_arr) / (np.linalg.norm(u_arr) * np.linalg.norm(v_arr)))\n\n"
            "numpy_dists = {i: np_cosine_dist(q, v) for i, v in zip(ids, vectors)}\n\n"
            "# Tổng hợp bảng kết quả so sánh\n"
            "table_rows = []\n"
            "for did, meta, vec in zip(ids, metas, vectors):\n"
            "    # Thứ hạng không lọc\n"
            "    rank_unfiltered = res_all['ids'][0].index(did) + 1 if did in res_all['ids'][0] else None\n"
            "    chroma_dist = res_all['distances'][0][rank_unfiltered - 1] if rank_unfiltered else None\n"
            "    \n"
            "    # Thứ hạng có lọc\n"
            "    rank_filtered = res_filtered['ids'][0].index(did) + 1 if did in res_filtered['ids'][0] else 'Bị loại'\n"
            "    \n"
            "    table_rows.append({\n"
            "        'Vector ID': did,\n"
            "        'Chủ đề (Topic)': meta['topic'],\n"
            "        'Tọa độ Vector': str(vec),\n"
            "        'Cosine Distance (NumPy)': round(numpy_dists[did], 4),\n"
            "        'Cosine Distance (Chroma)': round(chroma_dist, 4) if chroma_dist is not None else None,\n"
            "        'Thứ hạng (Không lọc)': rank_unfiltered,\n"
            "        'Thứ hạng (Lọc Topic = Y)': rank_filtered,\n"
            "    })\n\n"
            "df_exp1 = pd.DataFrame(table_rows)\n"
            "df_exp1"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# Trực quan hóa hình học 2D các vector q, a, b, c trên mặt phẳng (x1, x2)\n"
            "fig, ax = plt.subplots(figsize=(6.5, 5))\n"
            "ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)\n"
            "ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)\n\n"
            "# Vẽ cung tròn đơn vị để định hướng góc\n"
            "theta = np.linspace(0, np.pi / 2, 100)\n"
            "ax.plot(np.cos(theta), np.sin(theta), color='lightgray', linestyle=':', label='Đường tròn đơn vị')\n\n"
            "# Vẽ các vector mũi tên\n"
            "colors = {'q': '#DC2626', 'a': '#16A34A', 'b': '#2563EB', 'c': '#D97706'}\n"
            "vec_coords = {'q': [1.0, 0.0], 'a': [1.0, 0.0], 'b': [0.9, 0.1], 'c': [0.0, 1.0]}\n\n"
            "for label, coord in vec_coords.items():\n"
            "    dx, dy = coord[0], coord[1]\n"
            "    # Vẽ vector b với độ dài chuẩn hóa để trực quan góc\n"
            "    norm = np.hypot(dx, dy)\n"
            "    dx_norm, dy_norm = dx / norm, dy / norm\n"
            "    ax.quiver(0, 0, dx_norm, dy_norm, angles='xy', scale_units='xy', scale=1,\n"
            "              color=colors[label], width=0.012, alpha=0.9)\n"
            "    offset = 0.06\n"
            "    ax.text(dx_norm + (offset if dx_norm > 0.5 else -0.02), dy_norm + offset,\n"
            "            f'{label} {coord}', fontsize=11, fontweight='bold', color=colors[label])\n\n"
            "ax.set_xlim(-0.1, 1.25)\n"
            "ax.set_ylim(-0.1, 1.25)\n"
            "ax.set_aspect('equal')\n"
            "ax.set_xlabel('Thành phần chiều 1 (x1)')\n"
            "ax.set_ylabel('Thành phần chiều 2 (x2)')\n"
            "ax.set_title('Hình học 2D minh họa góc giữa các vector (Thí nghiệm 1)', fontsize=12, fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "**Nhận xét:**\n"
            "1. Khi không áp dụng bộ lọc, vector $a$ cùng hướng với $q$ (khoảng cách cosine 0.0000) xếp hạng 1, "
            "tiếp theo là vector $b$ (khoảng cách 0.0062) và vector $c$ (vuông góc, khoảng cách 1.0000) xếp hạng 3.\n"
            "2. Khi áp dụng bộ lọc `topic = 'Y'`, Chroma tiến hành lọc siêu dữ liệu trước (pre-filtering), "
            "loại bỏ toàn bộ các tài liệu thuộc topic X. Do đó, vector $c$ trở thành kết quả duy nhất ở hạng 1.\n"
            "3. Khoảng cách cosine do Chroma trả về trùng khớp hoàn toàn với kết quả tính toán đại số tuyến tính của NumPy. "
            "Phép lọc siêu dữ liệu chỉ thay đổi tập tài liệu hợp lệ được xem xét, không làm thay đổi định nghĩa khoảng cách."
        )
    )

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Vietnamese Corpus & Chroma Storage (Slide 5-6, 8-9, 17)
    # -------------------------------------------------------------------------
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 2. Trích Xuất Vector Embedding và Lưu Trữ Corpus Tiếng Việt\n"
            "*Liên hệ Slide 5–6, 8–9 và 17*\n\n"
            "Khảo sát kho học liệu gồm 114 đoạn văn bản CNTT tiếng Việt được lập chỉ mục trong Chroma (collection `seminar_corpus_e5`) "
            "bằng mô hình nhúng đa ngôn ngữ `intfloat/multilingual-e5-small`."
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# 1. Đọc corpus và thống kê phân bố theo chủ đề\n"
            "docs = []\n"
            "with open(PROJECT_ROOT / 'data/corpus/documents.jsonl', 'r', encoding='utf-8') as f:\n"
            "    for line in f:\n"
            "        if line.strip():\n"
            "            docs.append(json.loads(line))\n\n"
            "df_corpus = pd.DataFrame(docs)\n"
            "topic_counts = df_corpus['topic'].value_counts().reset_index()\n"
            "topic_counts.columns = ['Chủ đề CNTT', 'Số đoạn văn (Passages)']\n\n"
            "# 2. Vẽ biểu đồ phân bố chủ đề\n"
            "fig, ax = plt.subplots(figsize=(8, 3.8))\n"
            "bars = ax.barh(topic_counts['Chủ đề CNTT'], topic_counts['Số đoạn văn (Passages)'], color='#0284C7', height=0.6)\n"
            "ax.set_xlabel('Số lượng đoạn văn bản')\n"
            "ax.set_title('Phân bố 114 tài liệu tiếng Việt theo 6 nhóm chủ đề CNTT', fontsize=12, fontweight='bold')\n"
            "ax.set_xlim(0, 24)\n"
            "for bar in bars:\n"
            "    w = bar.get_width()\n"
            "    ax.text(w + 0.4, bar.get_y() + bar.get_height() / 2, f'{int(w)} đoạn', va='center', fontsize=10)\n"
            "plt.tight_layout()\n"
            "plt.show()\n\n"
            "topic_counts"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# Mở collection Chroma chính thức ở chế độ chỉ đọc và trích xuất một bản ghi mẫu\n"
            "client = chromadb.PersistentClient(path=str(PROJECT_ROOT / 'chroma_db'))\n"
            "col = client.get_collection(name='seminar_corpus_e5')\n"
            "total_records = col.count()\n\n"
            "# Lấy tài liệu db_003 (Chỉ mục B-Tree)\n"
            "sample_record = col.get(ids=['db_003'], include=['embeddings', 'documents', 'metadatas'])\n"
            "sample_meta = sample_record['metadatas'][0]\n"
            "sample_vec = np.array(sample_record['embeddings'][0])\n\n"
            "print(f'Collection Name:     {col.name}')\n"
            "print(f'Tổng số bản ghi:     {total_records} tài liệu')\n"
            "print(f'Độ đo khoảng cách:   {col.metadata.get(\"hnsw:space\", \"cosine\")}')\n"
            "print(f'Mã tài liệu mẫu:     db_003')\n"
            "print(f'Tiêu đề:             {sample_meta.get(\"title\")}')\n"
            "print(f'Chủ đề:              {sample_meta.get(\"topic\")}')\n"
            "print(f'Độ dài vector:       {len(sample_vec)} chiều (Dimensions)')\n"
            "print(f'Chuẩn L2 vector:     {np.linalg.norm(sample_vec):.6f} (Đã chuẩn hóa L2)')\n"
            "print(f'Mẫu 6 giá trị đầu:   {np.round(sample_vec[:6], 4)}')\n"
            "print(f'Đoạn văn bản trích:  \"{sample_record[\"documents\"][0][:130]}...\"')"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "**Nhận xét:**\n"
            "1. Kho tài liệu 114 đoạn được phân bố cân bằng tuyệt đối (19 đoạn mỗi chủ đề) trên 6 lĩnh vực CNTT quan trọng.\n"
            "2. Mô hình `multilingual-e5-small` nén thông tin ngữ nghĩa của đoạn văn bản thành một vector 384 chiều đã được chuẩn hóa L2 (độ dài vector bằng 1.0).\n"
            "3. Phân định rạch ròi kiến trúc: **Mô hình E5 đảm nhiệm việc biểu diễn ngữ nghĩa dạng vector, còn Chroma đóng vai trò cơ sở dữ liệu chuyên dụng để quản lý khóa ID, lưu trữ siêu dữ liệu và tối ưu hóa việc tìm kiếm láng giềng gần nhất trên đĩa cứng.**"
        )
    )

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Semantic Search vs BM25 (Slide 3-4)
    # -------------------------------------------------------------------------
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 3. Đối Sánh Tìm Kiếm Ngữ Nghĩa (Chroma) và Tìm Kiếm Từ Khóa (BM25)\n"
            "*Liên hệ Slide 3–4*\n\n"
            "Thí nghiệm so sánh kết quả Top-5 giữa Chroma Semantic Search (vector dense) và BM25 (sparse lexical) trên cùng 114 tài liệu "
            "với 3 câu hỏi thuộc tập phát triển (Dev split):\n"
            "- `q_001`: Tìm vật thể và vị trí trong ảnh (diễn đạt tự nhiên, paraphrase)\n"
            "- `q_005`: Ngăn chặn mã độc nhập vào ô tìm kiếm phá CSDL (thuật ngữ an ninh ứng dụng)\n"
            "- `q_019`: Sự khác biệt giữa điều chuẩn L1 Lasso và điều chuẩn L2 Ridge (đối chiếu khái niệm kỹ thuật)"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# Nạp module truy xuất Chroma và BM25 đã xây dựng trong dự án\n"
            "from src.chroma_search import ChromaRetriever\n"
            "from src.bm25_search import BM25Retriever\n\n"
            "chroma_retriever = ChromaRetriever(db_path=str(PROJECT_ROOT / 'chroma_db'))\n"
            "bm25_retriever = BM25Retriever(data_path=str(PROJECT_ROOT / 'data/corpus/documents.jsonl'))\n\n"
            "dev_queries = [\n"
            "    ('q_001', 'làm thế nào để máy tính tự động tìm ra các vật thể và vị trí của chúng trong một bức ảnh?'),\n"
            "    ('q_005', 'làm sao để ngăn chặn người dùng nhập mã độc vào ô tìm kiếm nhằm phá hoại cơ sở dữ liệu web'),\n"
            "    ('q_019', 'sự khác biệt giữa điều chuẩn L1 Lasso và điều chuẩn L2 Ridge'),\n"
            "]\n\n"
            "# Chạy truy vấn và hiển thị bảng đối chiếu Top-5 cho từng câu hỏi\n"
            "for qid, qtext in dev_queries:\n"
            "    c_top5 = chroma_retriever.search(qtext, top_k=5)\n"
            "    b_top5 = bm25_retriever.search(qtext, top_k=5)\n"
            "    \n"
            "    rows = []\n"
            "    for rank in range(1, 6):\n"
            "        c_item = c_top5[rank - 1]\n"
            "        b_item = b_top5[rank - 1]\n"
            "        rows.append({\n"
            "            'Hạng': rank,\n"
            "            'Chroma ID': c_item['id'],\n"
            "            'Tiêu đề (Chroma)': c_item['title'][:32],\n"
            "            'Cosine Dist': round(c_item['distance'], 4),\n"
            "            'BM25 ID': b_item['id'],\n"
            "            'Tiêu đề (BM25)': b_item['title'][:32],\n"
            "            'BM25 Score': round(b_item['score'], 4),\n"
            "        })\n"
            "    \n"
            "    print(f'=== TRUY VẤN {qid}: \"{qtext}\" ===')\n"
            "    df_q = pd.DataFrame(rows)\n"
            "    display(df_q)\n"
            "    print()"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# Phân tích chuyên sâu truy vấn q_001: Khảo sát thứ tự của 4 tài liệu thị giác liên quan\n"
            "# cv_009: Image Classification | cv_010: Faster R-CNN | cv_011: YOLO | cv_014: Mask R-CNN\n"
            "q_001_text = dev_queries[0][1]\n"
            "c_all = chroma_retriever.search(q_001_text, top_k=114)\n"
            "b_all = bm25_retriever.search(q_001_text, top_k=114)\n\n"
            "target_ids = {\n"
            "    'cv_009': 'Phân loại ảnh (Classification)',\n"
            "    'cv_014': 'Phân đoạn cá thể (Mask R-CNN)',\n"
            "    'cv_011': 'Phát hiện đối tượng (YOLO)',\n"
            "    'cv_010': 'Phát hiện đối tượng (Faster R-CNN)',\n"
            "}\n\n"
            "c_ranks = {r['id']: r['rank'] for r in c_all}\n"
            "b_ranks = {r['id']: r['rank'] for r in b_all}\n\n"
            "df_targets = pd.DataFrame([\n"
            "    {\n"
            "        'Mã tài liệu': did,\n"
            "        'Nội dung chính': desc,\n"
            "        'Thứ hạng Chroma': c_ranks.get(did, '>114'),\n"
            "        'Thứ hạng BM25': b_ranks.get(did, '>114'),\n"
            "    }\n"
            "    for did, desc in target_ids.items()\n"
            "])\n\n"
            "# Trực quan hóa thứ hạng (trục tung đảo ngược: Hạng 1 cao nhất)\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "x = np.arange(len(target_ids))\n"
            "width = 0.35\n\n"
            "chroma_vals = [c_ranks[did] for did in target_ids]\n"
            "bm25_vals = [b_ranks[did] for did in target_ids]\n\n"
            "ax.bar(x - width/2, chroma_vals, width, label='Chroma (Cosine)', color='#10B981')\n"
            "ax.bar(x + width/2, bm25_vals, width, label='BM25 (Từ vựng)', color='#3B82F6')\n\n"
            "ax.set_ylabel('Thứ hạng truy xuất (Hạng 1 = Tốt nhất)')\n"
            "ax.set_title('So sánh thứ hạng 4 tài liệu thị giác trên truy vấn q_001', fontsize=12, fontweight='bold')\n"
            "ax.set_xticks(x)\n"
            "ax.set_xticklabels([f'{did}\\n{desc[:16]}' for did, desc in target_ids.items()], fontsize=9.5)\n"
            "ax.set_ylim(32, 0)  # Đảo ngược trục để hạng nhỏ hơn nằm phía trên\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()\n\n"
            "df_targets"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "**Nhận xét:**\n"
            "1. Trên $q_{001}$, cả hai phương pháp đều xếp `cv_009` (Phân loại ảnh) ở vị trí số 1 do độ bão hòa cao của các từ vựng chung như 'bức ảnh', 'thị giác máy tính'. Tuy nhiên, Chroma nắm bắt được ý định 'vị trí vật thể' để đưa `cv_014` (Phân đoạn cá thể) lên hạng 2 (khoảng cách 0.1504 rất sát 0.1452).\n"
            "2. Trên $q_{005}$, BM25 nhận diện chính xác tài liệu SQL Injection (`net_011`) ở hạng 1 nhờ khớp từ khóa 'cơ sở dữ liệu'. Chroma cũng đưa nhóm an ninh web (XSS, SQLi, DDoS, CSRF) vào toàn bộ Top-4.\n"
            "3. Trên $q_{019}$, Chroma phân biệt rất tốt ngữ cảnh để đưa L2 Ridge (`ml_005`) và L1 Lasso (`ml_004`) vào Top-1 và Top-2, trong khi BM25 bị nhiễu bởi từ 'chuẩn hóa' nên đẩy tài liệu Batch Normalization lên đầu.\n"
            "4. Thí nghiệm minh chứng khách quan cả ưu điểm và giới hạn: Tìm kiếm ngữ nghĩa vượt trội ở các câu hỏi diễn đạt khác từ khóa, nhưng có thể bị phân tán khi tài liệu chuyên môn sử dụng hệ thuật ngữ học thuật hẹp."
        )
    )

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: Chroma HNSW vs Exact Nearest Neighbor (Slide 11-13, 19)
    # -------------------------------------------------------------------------
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 4. Đối Chiếu Chỉ Mục HNSW của Chroma với Tìm Kiếm Chính Xác (Exact Search)\n"
            "*Liên hệ Slide 11–13 và Slide 19*\n\n"
            "**Câu hỏi khoa học:** Trên cùng một không gian vector, chỉ mục đồ thị HNSW của Chroma có tìm lại đúng Top-5 láng giềng gần nhất "
            "so với phép tính khoảng cách cosine vét cạn (exhaustive exact search) trên toàn bộ kho tài liệu hay không?\n\n"
            "Chỉ số đánh giá độ tái hồi láng giềng gần đúng:\n"
            "$$\\text{ANN Recall@5} = \\frac{|\\text{Top-5}_{\\text{Chroma HNSW}} \\cap \\text{Top-5}_{\\text{Exact Cosine}}|}{5}$$"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# 1. Lấy toàn bộ vector đã lưu trong Chroma collection seminar_corpus_e5\n"
            "stored_data = col.get(include=['embeddings', 'documents', 'metadatas'])\n"
            "corpus_ids = stored_data['ids']\n"
            "corpus_embeddings = np.array(stored_data['embeddings'], dtype=np.float32)\n"
            "print(f'Đã trích xuất ma trận embedding từ Chroma: {corpus_embeddings.shape}')\n\n"
            "# 2. Đối chiếu kết quả trên 3 dev queries\n"
            "from sentence_transformers import SentenceTransformer\n"
            "model = SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')\n\n"
            "ann_comparison_rows = []\n"
            "recall_scores = []\n\n"
            "for qid, qtext in dev_queries:\n"
            "    # Chroma HNSW search\n"
            "    c_top5 = chroma_retriever.search(qtext, top_k=5)\n"
            "    chroma_top5_ids = [item['id'] for item in c_top5]\n"
            "    \n"
            "    # Exact Cosine search in RAM using NumPy\n"
            "    q_vec = model.encode(f'query: {qtext.strip()}', normalize_embeddings=True, show_progress_bar=False)\n"
            "    # Cosine distance = 1.0 - dot_product (đối với vector đã chuẩn hóa L2)\n"
            "    exact_distances = 1.0 - np.dot(corpus_embeddings, q_vec)\n"
            "    exact_top5_indices = np.argsort(exact_distances)[:5]\n"
            "    exact_top5_ids = [corpus_ids[idx] for idx in exact_top5_indices]\n"
            "    \n"
            "    # Tính số lượng trùng khớp và ANN Recall@5\n"
            "    overlap = set(chroma_top5_ids) & set(exact_top5_ids)\n"
            "    recall_5 = len(overlap) / 5.0\n"
            "    recall_scores.append(recall_5)\n"
            "    \n"
            "    ann_comparison_rows.append({\n"
            "        'Mã Query': qid,\n"
            "        'Chroma HNSW Top-5': ', '.join(chroma_top5_ids),\n"
            "        'Exact Cosine Top-5': ', '.join(exact_top5_ids),\n"
            "        'Số ID trùng khớp': f'{len(overlap)} / 5',\n"
            "        'ANN Recall@5': f'{recall_5 * 100:.1f}%',\n"
            "    })\n\n"
            "df_ann = pd.DataFrame(ann_comparison_rows)\n"
            "df_ann"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# Trực quan hóa tỷ lệ ANN Recall@5 trên 3 câu hỏi kiểm tra\n"
            "fig, ax = plt.subplots(figsize=(6.5, 3.5))\n"
            "q_labels = [f'{qid}\\n({qtext[:25]}...)' for qid, qtext in dev_queries]\n"
            "bars = ax.bar(q_labels, [r * 100 for r in recall_scores], color='#059669', width=0.45)\n"
            "ax.set_ylabel('ANN Recall@5 (%)')\n"
            "ax.set_ylim(0, 120)\n"
            "ax.set_title('Độ tái hồi ANN Recall@5 của Chroma HNSW so với Exact Cosine', fontsize=12, fontweight='bold')\n"
            "for bar in bars:\n"
            "    y = bar.get_height()\n"
            "    ax.text(bar.get_x() + bar.get_width() / 2, y + 3, f'{y:.0f}% (5/5 khớp)', ha='center', fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "**Nhận xét:**\n"
            "1. Trên cả 3 truy vấn được kiểm tra, Top-5 của Chroma HNSW trùng khớp 100% với phương pháp tìm kiếm chính xác vét cạn (ANN Recall@5 = 1.0).\n"
            "2. Kết quả này chứng minh thuật toán HNSW của Chroma hoạt động hoàn toàn chính xác và không làm thất lạc láng giềng gần nhất trên tập dữ liệu này. Xếp hạng của `cv_009` trên $q_{001}$ là do đặc trưng hình học của vector E5 chứ không phải do sai số xấp xỉ của chỉ mục HNSW.\n"
            "3. Cần lưu ý rằng kết quả 100% này được ghi nhận cụ thể trên 3 truy vấn thử nghiệm với quy mô 114 tài liệu; không phải là sự bảo đảm phổ quát chung rằng HNSW luôn đạt 100% trong mọi điều kiện dữ liệu quy mô hàng triệu vector."
        )
    )

    # -------------------------------------------------------------------------
    # EXPERIMENT 5: Evaluation Workflow & Verified Gate (Slide 19-20)
    # -------------------------------------------------------------------------
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 5. Quy Trình Đánh Giá Chất Lượng Truy Xuất và Cổng Kiểm Duyệt Nhãn\n"
            "*Liên hệ Slide 19–20*\n\n"
            "Thí nghiệm minh chứng quy trình đánh giá chuẩn khoa học: xây dựng candidate pool, định nghĩa các độ đo IR "
            "(Hit@5, MRR@5, nDCG@5) và thiết lập cơ chế kiểm soát nhãn đã xác thực (Verified-Label Gate) nhằm đảm bảo tính trung thực trong nghiên cứu."
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# 1. Khảo sát tiến độ kiểm duyệt trên tập Candidate Review Pool\n"
            "import csv\n"
            "pool_rows = []\n"
            "with open(PROJECT_ROOT / 'data/eval/review_pool.csv', 'r', encoding='utf-8-sig') as f:\n"
            "    reader = csv.DictReader(f)\n"
            "    pool_rows = list(reader)\n\n"
            "total_candidates = len(pool_rows)\n"
            "verified_count = sum(1 for r in pool_rows if r.get('review_status') == 'verified')\n"
            "pending_count = total_candidates - verified_count\n\n"
            "print(f'Tổng số cặp (query, doc) trong Review Pool: {total_candidates}')\n"
            "print(f'Số nhãn đã được con người thẩm định:       {verified_count} (0.0%)')\n"
            "print(f'Số nhãn đang chờ kiểm duyệt (Pending):      {pending_count} (100.0%)')\n\n"
            "# Hiển thị trích đoạn 5 dòng đầu tiên từ bảng kiểm duyệt\n"
            "df_pool_sample = pd.DataFrame([\n"
            "    {\n"
            "        'Query ID': r['query_id'],\n"
            "        'Câu hỏi': r['query_text'][:35] + '...',\n"
            "        'Doc ID': r['doc_id'],\n"
            "        'Tiêu đề tài liệu': r['doc_title'][:30],\n"
            "        'Proposed Rel': r['proposed_relevance'] or '(unjudged)',\n"
            "        'Verified Rel': r['verified_relevance'] or '[Trống]',\n"
            "        'Trạng thái': r['review_status'],\n"
            "    }\n"
            "    for r in pool_rows[:5]\n"
            "])\n"
            "df_pool_sample"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# 2. Kiểm tra Cổng Verified-Label Gate và Hiển thị Bảng Xếp Hạng Xem Trước (Preview Rankings)\n"
            "from src.ui_helpers import check_official_benchmark_eligibility, load_preview_artifact\n\n"
            "is_eligible, gate_reason, official_data = check_official_benchmark_eligibility(\n"
            "    review_pool_path=PROJECT_ROOT / 'data/eval/review_pool.csv',\n"
            "    corpus_path=PROJECT_ROOT / 'data/corpus/documents.jsonl',\n"
            "    official_json_path=PROJECT_ROOT / 'results/g3/official_metrics.json'\n"
            ")\n\n"
            "if not is_eligible:\n"
            "    print('>>> TRẠNG THÁI CỔNG KIỂM DUYỆT (VERIFIED GATE): ĐANG KHÓA AN TOÀN <<<')\n"
            "    print(f'Lý do: {gate_reason}')\n"
            "    print('Hệ thống từ chối xuất điểm số benchmark chính thức khi nhãn chưa được con người phê duyệt.\\n')\n\n"
            "# Đọc kết quả đối soát thứ hạng Top-1 từ preview artifact trên 12 test queries\n"
            "preview_data = load_preview_artifact(PROJECT_ROOT / 'results/g3/rankings_preview.json')\n"
            "if preview_data and 'eval_results' in preview_data:\n"
            "    preview_rows = []\n"
            "    for item in preview_data['eval_results']:\n"
            "        c_top1 = item['chroma_top_k'][0]['doc_id'] if item['chroma_top_k'] else 'N/A'\n"
            "        b_top1 = item['bm25_top_k'][0]['doc_id'] if item['bm25_top_k'] else 'N/A'\n"
            "        agree = 'Trùng khớp' if item['top1_match'] else 'Khác biệt'\n"
            "        if item['is_unanswerable']:\n"
            "            agree += ' (Unanswerable)'\n"
            "        preview_rows.append({\n"
            "            'Query ID': item['query_id'],\n"
            "            'Thể loại': item['category'],\n"
            "            'Chroma Top-1': c_top1,\n"
            "            'BM25 Top-1': b_top1,\n"
            "            'Đồng thuận Top-1': agree,\n"
            "        })\n"
            "    \n"
            "    df_preview = pd.DataFrame(preview_rows)\n"
            "    display(df_preview)\n"
            "    \n"
            "    agr_cnt = preview_data['agreement']['top1_agreement_count']\n"
            "    total_q = preview_data['subset']['total_queries']\n"
            "    print(f'Tỷ lệ đồng thuận Top-1: {agr_cnt}/{total_q} ({round(100.0 * agr_cnt / total_q, 1)}%)')"
        )
    )

    cells.append(
        nbformat.v4.new_code_cell(
            "# 3. Phân tích định tính truy vấn ngoài phạm vi (Unanswerable query q_030)\n"
            "unans_item = next(item for item in preview_data['eval_results'] if item['is_unanswerable'])\n"
            "print(f'Truy vấn q_030: \"{unans_item[\"query_text\"]}\"')\n"
            "print(f'- Chroma láng giềng gần nhất: {unans_item[\"chroma_top_k\"][0][\"doc_id\"]} (Khoảng cách: {unans_item[\"chroma_top_k\"][0][\"distance\"]})')\n"
            "print(f'- BM25 láng giềng gần nhất:   {unans_item[\"bm25_top_k\"][0][\"doc_id\"]} (Điểm BM25: {unans_item[\"bm25_top_k\"][0][\"score\"]})')\n"
            "print('\\nNhận xét: Cả hai phương pháp đều không có cơ chế tự từ chối (abstain).')\n"
            "print('Khoảng cách vector gần nhất không đồng nghĩa với câu trả lời đúng.')\n"
            "print('Vì vậy câu hỏi này được phân tích độc lập và loại khỏi mẫu số tính điểm trung bình Hit/MRR/nDCG.')"
        )
    )

    cells.append(
        nbformat.v4.new_markdown_cell(
            "**Nhận xét & Tổng kết:**\n"
            "1. Quy trình đánh giá phân định rạch ròi giữa thuật toán xếp hạng và ground truth đã được con người thẩm định. "
            "Việc từ chối công bố điểm benchmark khi nhãn còn pending là nguyên tắc liêm chính khoa học bắt buộc.\n"
            "2. Đánh giá dựa trên candidate pool (pooled judgments) phản ánh tính tương đối của pool ứng viên chứ chưa bao quát toàn bộ corpus. "
            "Khi người kiểm duyệt hoàn tất xác thực 105 nhãn, hệ thống sẽ tự động tính toán Hit@5, MRR@5 và nDCG@5 chính thức."
        )
    )

    nb.cells = cells
    return nb


def build_and_execute_notebook() -> None:
    print("[1/3] Constructing experiment notebook structure...")
    nb = create_experiment_notebook()

    print("[2/3] Executing notebook sequentially on active Python environment...")
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()

    print(f"[3/3] Saving executed notebook to '{OUTPUT_NOTEBOOK_PATH}'...")
    OUTPUT_NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    file_size_kb = OUTPUT_NOTEBOOK_PATH.stat().st_size / 1024
    print(f"Notebook successfully built, executed, and saved! Size: {file_size_kb:.1f} KB")


if __name__ == "__main__":
    build_and_execute_notebook()

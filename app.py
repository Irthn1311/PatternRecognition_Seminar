"""
Chroma Semantic Search Lab - Streamlit Web Application (Tasks G4 & G5).

An interactive demo application for the Pattern Recognition Seminar:
  - Tab 1: Semantic Search (Chroma vs Vietnamese BM25 comparison with topic filtering)
  - Tab 2: Collection Explorer (Read-only official corpus + Mutable CRUD demo collection)
  - Tab 3: Experiment Results (12 test queries preview, disagreement analysis, verified gate)
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import ReviewPoolGate
from src.indexer import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_NAME,
    CorpusIndexer,
)
from src.ui_helpers import (
    DEFAULT_MUTABLE_COLLECTION,
    DEFAULT_OFFICIAL_JSON_PATH,
    DEFAULT_PREVIEW_JSON_PATH,
    DEFAULT_REVIEW_POOL_PATH,
    STANDARD_TOPICS,
    add_mutable_document,
    check_official_benchmark_eligibility,
    delete_mutable_document,
    get_cached_bm25_retriever,
    get_cached_chroma_client,
    get_cached_chroma_retriever,
    get_cached_embedding_model,
    get_or_create_mutable_collection,
    list_mutable_documents,
    load_preview_artifact,
    search_mutable_collection,
    update_mutable_document,
)


def safe_html(text: Any) -> str:
    """Escapes HTML special characters in dynamic strings to prevent injection."""
    return html.escape(str(text) if text is not None else "", quote=True)


# Set page configuration
st.set_page_config(
    page_title="Chroma Semantic Search Lab",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS focused on readability, high contrast, and dark theme presentation
st.markdown(
    """
    <style>
    /* Main title and header styling */
    .app-main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    .app-main-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
        font-weight: 400;
    }
    /* Badges with explicit high-contrast colors */
    .badge-rank-chroma {
        display: inline-block;
        background-color: #065F46;
        color: #A7F3D0;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .badge-rank-bm25 {
        display: inline-block;
        background-color: #1E3A8A;
        color: #BFDBFE;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .badge-topic-tag {
        display: inline-block;
        background-color: #334155;
        color: #F1F5F9;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .metric-score-chroma {
        color: #34D399;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .metric-score-bm25 {
        color: #60A5FA;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .result-title-text {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
        margin-bottom: 6px;
    }
    .passage-preview-text {
        font-size: 0.95rem;
        color: #CBD5E1;
        line-height: 1.5;
        margin-top: 6px;
    }
    .disclaimer-box {
        background-color: #1E293B;
        border-left: 4px solid #F59E0B;
        border-radius: 4px;
        padding: 12px 16px;
        font-size: 0.9rem;
        color: #FDE68A;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar_info() -> None:
    """Renders compact sidebar with essential status and technical expander."""
    with st.sidebar:
        st.markdown("### 🔍 Semantic Search Lab")
        st.markdown("**Seminar:** Nhận diện Mẫu (Pattern Recognition)")

        # Index Status
        try:
            indexer = CorpusIndexer()
            is_consistent, msg, manifest = indexer.check_index_consistency()
            if is_consistent:
                st.success("✅ **Chroma Index:** Đã đồng bộ (114 records)")
            else:
                st.error("⚠️ **Chroma Index:** Chưa đồng bộ / lỗi")
                st.caption(msg)
                st.code("python scripts/build_index.py", language="powershell")
        except Exception as e:
            st.error(f"Lỗi kiểm tra index: {e}")

        st.divider()

        # Technical specifications in compact expander
        with st.expander("🛠️ Chi tiết cấu hình kỹ thuật"):
            st.markdown(
                "- **Embedding Model:** `intfloat/multilingual-e5-small`\n"
                "- **Vector Dim:** 384 dimensions\n"
                "- **Normalization:** L2 Normalization\n"
                "- **Prefixes:** `passage: ` vs `query: `\n"
                "- **Distance Metric:** Cosine Distance\n"
                "- **Lexical Baseline:** BM25Okapi (`rank_bm25`)\n"
                "- **Tokenizer:** `pyvi.ViTokenizer` (từ ghép tiếng Việt)\n"
                "- **Corpus:** 114 passages (6 chủ đề)\n"
                "- **Inference:** CPU (local)"
            )

        st.caption("Trường Đại học Sài Gòn (SGU) — Khoa CNTT")


def init_session_state() -> None:
    """Initializes session state keys for persistent search context."""
    if "has_searched" not in st.session_state:
        st.session_state["has_searched"] = False
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""
    if "last_mode" not in st.session_state:
        st.session_state["last_mode"] = "Both (Chroma vs BM25)"
    if "last_top_k" not in st.session_state:
        st.session_state["last_top_k"] = 5
    if "last_topic" not in st.session_state:
        st.session_state["last_topic"] = "All Topics"
    if "last_chroma_results" not in st.session_state:
        st.session_state["last_chroma_results"] = []
    if "last_bm25_results" not in st.session_state:
        st.session_state["last_bm25_results"] = []
    if "last_chroma_error" not in st.session_state:
        st.session_state["last_chroma_error"] = None
    if "last_bm25_error" not in st.session_state:
        st.session_state["last_bm25_error"] = None
    if "query_input_val" not in st.session_state:
        st.session_state["query_input_val"] = ""


def render_tab_semantic_search() -> None:
    """Tab 1: Interactive semantic search and BM25 comparison with deliberate submission."""
    init_session_state()

    st.markdown("### 🔍 Tìm Kiếm Ngữ Nghĩa & Đối Sánh BM25")
    st.markdown(
        "Nhập câu hỏi bằng tiếng Việt để đối sánh kết quả giữa **Chroma VectorDB (Dense Semantic)** "
        "và **BM25 Tiếng Việt (Sparse Lexical)** trên cùng 114 đoạn tài liệu CNTT."
    )

    # 1. Preset sample queries selector (only populates text, does not auto-submit)
    sample_queries = [
        "-- Chọn câu hỏi mẫu để điền nhanh --",
        "làm thế nào để máy tính tự động tìm ra các vật thể và vị trí của chúng trong một bức ảnh?",
        "cách giải quyết khi mô hình học máy thuộc lòng dữ liệu huấn luyện và dự đoán kém trên thực tế",
        "giải pháp tăng tốc độ tìm kiếm bản ghi trong bảng cơ sở dữ liệu hàng triệu dòng",
        "thuật toán tìm đường đi ngắn nhất giữa hai địa điểm trên bản đồ giao thông",
        "sự khác biệt giữa điều chuẩn L1 Lasso và điều chuẩn L2 Ridge",
        "thuật toán đồng thuận Proof of Stake và xử lý phân nhánh blockchain Ethereum",
    ]

    def on_sample_change():
        selected = st.session_state.get("sample_selector_val", "")
        if selected and selected != sample_queries[0]:
            st.session_state["query_input_val"] = selected

    st.selectbox(
        "📌 Câu hỏi mẫu gợi ý (chọn để điền nội dung vào ô tìm kiếm):",
        sample_queries,
        key="sample_selector_val",
        on_change=on_sample_change,
    )

    # 2. Search Form (deliberate submission - prevents accidental reruns)
    with st.form("search_form", clear_on_submit=False):
        query_input = st.text_input(
            "Nhập câu hỏi truy vấn tiếng Việt:",
            value=st.session_state.get("query_input_val", ""),
            placeholder="Ví dụ: kiến trúc mạng nơ-ron tích chập CNN trích xuất đặc trưng hình ảnh...",
        )

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 2])
        with ctrl_col1:
            mode_choice = st.selectbox(
                "Phương pháp truy xuất:",
                ["Both (Chroma vs BM25)", "Chroma Semantic Search", "Vietnamese BM25"],
                index=0,
            )
        with ctrl_col2:
            top_k = st.slider("Top-K kết quả:", min_value=1, max_value=10, value=5)
        with ctrl_col3:
            topic_filter = st.selectbox("Lọc theo chủ đề (Metadata filter):", STANDARD_TOPICS, index=0)

        submit_search = st.form_submit_button("🚀 Tìm kiếm", type="primary", use_container_width=True)

    # 3. Handle deliberate form submission
    if submit_search:
        q_text = query_input.strip()
        if not q_text:
            st.warning("⚠️ Vui lòng nhập câu hỏi truy vấn trước khi nhấn tìm kiếm.")
        else:
            filter_val = None if topic_filter == "All Topics" else topic_filter

            # Check Chroma index consistency
            indexer = CorpusIndexer()
            is_consistent, msg, _ = indexer.check_index_consistency()

            chroma_results: List[Dict[str, Any]] = []
            bm25_results: List[Dict[str, Any]] = []
            chroma_error: Optional[str] = None
            bm25_error: Optional[str] = None

            # Execute Chroma
            if mode_choice in ["Both (Chroma vs BM25)", "Chroma Semantic Search"]:
                if not is_consistent:
                    chroma_error = f"Chroma index chưa đồng bộ hoặc bị thiếu ({msg}). Chạy 'python scripts/build_index.py' để sửa."
                else:
                    try:
                        retriever = get_cached_chroma_retriever()
                        chroma_results = retriever.search(q_text, top_k=top_k, topic=filter_val)
                    except Exception as e:
                        chroma_error = str(e)

            # Execute BM25
            if mode_choice in ["Both (Chroma vs BM25)", "Vietnamese BM25"]:
                try:
                    retriever = get_cached_bm25_retriever()
                    bm25_results = retriever.search(q_text, top_k=top_k, topic=filter_val)
                except Exception as e:
                    bm25_error = str(e)

            # Update session state with last search results and parameters
            st.session_state["has_searched"] = True
            st.session_state["last_query"] = q_text
            st.session_state["last_mode"] = mode_choice
            st.session_state["last_top_k"] = top_k
            st.session_state["last_topic"] = topic_filter
            st.session_state["last_chroma_results"] = chroma_results
            st.session_state["last_bm25_results"] = bm25_results
            st.session_state["last_chroma_error"] = chroma_error
            st.session_state["last_bm25_error"] = bm25_error

    # 4. Display Results OR Empty State
    st.divider()

    if not st.session_state.get("has_searched", False):
        # Clean informative empty state
        with st.container(border=True):
            st.markdown("#### 👋 Chào mừng đến với Chroma Semantic Search Lab!")
            st.markdown(
                "Để bắt đầu trải nghiệm, bạn có thể:\n\n"
                "1. Chọn một trong các **câu hỏi mẫu gợi ý** ở trên để tự động điền câu hỏi.\n"
                "2. Hoặc tự nhập câu hỏi kỹ thuật CNTT bằng tiếng Việt vào ô tìm kiếm.\n"
                "3. Tùy chỉnh chế độ đối sánh (**Both**, **Chroma**, hoặc **BM25**), số lượng Top-K và bộ lọc chủ đề.\n"
                "4. Nhấn **'🚀 Tìm kiếm'** để kích hoạt suy luận vector và truy hồi từ khóa."
            )
            st.caption("Hệ thống sẽ chạy đối sánh song song trên 114 đoạn tài liệu học liệu CNTT.")
        return

    # Render persisted search results
    last_query = st.session_state.get("last_query", "")
    last_mode = st.session_state.get("last_mode", "Both (Chroma vs BM25)")
    last_top_k = st.session_state.get("last_top_k", 5)
    last_topic = st.session_state.get("last_topic", "All Topics")
    chroma_results = st.session_state.get("last_chroma_results", [])
    bm25_results = st.session_state.get("last_bm25_results", [])
    chroma_error = st.session_state.get("last_chroma_error", None)
    bm25_error = st.session_state.get("last_bm25_error", None)

    st.markdown(
        f"**Kết quả tìm kiếm cho:** *\"{last_query}\"* &nbsp;|&nbsp; "
        f"**Top-K:** `{last_top_k}` &nbsp;|&nbsp; "
        f"**Chủ đề:** `{last_topic}` &nbsp;|&nbsp; "
        f"**Chế độ:** `{last_mode}`"
    )

    if last_mode == "Both (Chroma vs BM25)":
        col_left, col_right = st.columns(2)

        # Left Column: Chroma
        with col_left:
            st.markdown("#### 🧠 Chroma Semantic Search (E5 Small)")
            st.caption("Thang đo: **Cosine Distance** (càng nhỏ càng gần ngữ nghĩa, khoảng [0..2])")
            if chroma_error:
                st.error(f"❌ Lỗi Chroma: {chroma_error}")
            elif not chroma_results:
                st.info("Không tìm thấy tài liệu Chroma nào khớp với bộ lọc.")
            else:
                for item in chroma_results:
                    with st.container(border=True):
                        # Title is prominent with safe escaping
                        st.markdown(f'<div class="result-title-text">{safe_html(item["title"])}</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<span class="badge-rank-chroma">Hạng {item["rank"]}</span> '
                            f'<span class="badge-topic-tag">🏷️ {safe_html(item["topic"])}</span> &nbsp; '
                            f'<span class="metric-score-chroma">📐 Cosine Distance: {item["distance"]:.4f}</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Mã ID: `{item['id']}`")
                        # Preview text with safe escaping
                        preview = item["document"][:200] + ("..." if len(item["document"]) > 200 else "")
                        st.markdown(f'<div class="passage-preview-text">{safe_html(preview)}</div>', unsafe_allow_html=True)
                        with st.expander("📖 Xem toàn văn tài liệu & nguồn"):
                            st.write(item["document"])
                            st.caption(f"Nguồn: {item.get('source_url') or 'Nội bộ dự án'}")

        # Right Column: BM25
        with col_right:
            st.markdown("#### 📖 Vietnamese BM25 Baseline (PyVi)")
            st.caption("Thang đo: **BM25 Score** (càng lớn càng khớp từ khóa, không giới hạn trên)")
            if bm25_error:
                st.error(f"❌ Lỗi BM25: {bm25_error}")
            elif not bm25_results:
                st.info("Không tìm thấy tài liệu BM25 nào khớp với bộ lọc.")
            else:
                for item in bm25_results:
                    with st.container(border=True):
                        # Title is prominent with safe escaping
                        st.markdown(f'<div class="result-title-text">{safe_html(item["title"])}</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<span class="badge-rank-bm25">Hạng {item["rank"]}</span> '
                            f'<span class="badge-topic-tag">🏷️ {safe_html(item["topic"])}</span> &nbsp; '
                            f'<span class="metric-score-bm25">📊 BM25 Score: {item["score"]:.4f}</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Mã ID: `{item['id']}`")
                        # Preview text with safe escaping
                        preview = item["document"][:200] + ("..." if len(item["document"]) > 200 else "")
                        st.markdown(f'<div class="passage-preview-text">{safe_html(preview)}</div>', unsafe_allow_html=True)
                        with st.expander("📖 Xem toàn văn tài liệu & nguồn"):
                            st.write(item["document"])
                            st.caption(f"Nguồn: {item.get('source_url') or 'Nội bộ dự án'}")

        st.markdown(
            """
            <div class="disclaimer-box">
                💡 <b>Lưu ý khoa học về độ đo:</b> Khoảng cách Cosine của Chroma và Điểm số BM25 có bản chất toán học 
                hoàn toàn khác nhau. Tuyệt đối không quy đổi trực tiếp hoặc gộp chung hai thang đo này thành phần trăm độ tin cậy (% confidence).
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif last_mode == "Chroma Semantic Search":
        st.markdown("#### 🧠 Kết Quả Chroma Semantic Search")
        st.caption("Thang đo: Cosine Distance (nhỏ hơn = gần ngữ nghĩa hơn)")
        if chroma_error:
            st.error(f"❌ Lỗi Chroma: {chroma_error}")
        elif not chroma_results:
            st.info("Không tìm thấy tài liệu nào khớp với bộ lọc.")
        else:
            for item in chroma_results:
                with st.container(border=True):
                    st.markdown(f'<div class="result-title-text">{safe_html(item["title"])}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<span class="badge-rank-chroma">Hạng {item["rank"]}</span> '
                        f'<span class="badge-topic-tag">🏷️ {safe_html(item["topic"])}</span> &nbsp; '
                        f'<span class="metric-score-chroma">📐 Cosine Distance: {item["distance"]:.4f}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Mã ID: `{item['id']}`")
                    st.write(item["document"])
                    st.caption(f"Nguồn: {item.get('source_url') or 'Nội bộ dự án'}")

    elif last_mode == "Vietnamese BM25":
        st.markdown("#### 📖 Kết Quả Vietnamese BM25 Baseline")
        st.caption("Thang đo: BM25 Score (lớn hơn = khớp từ khóa hơn)")
        if bm25_error:
            st.error(f"❌ Lỗi BM25: {bm25_error}")
        elif not bm25_results:
            st.info("Không tìm thấy tài liệu nào khớp với bộ lọc.")
        else:
            for item in bm25_results:
                with st.container(border=True):
                    st.markdown(f'<div class="result-title-text">{safe_html(item["title"])}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<span class="badge-rank-bm25">Hạng {item["rank"]}</span> '
                        f'<span class="badge-topic-tag">🏷️ {safe_html(item["topic"])}</span> &nbsp; '
                        f'<span class="metric-score-bm25">📊 BM25 Score: {item["score"]:.4f}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Mã ID: `{item['id']}`")
                    st.write(item["document"])
                    st.caption(f"Nguồn: {item.get('source_url') or 'Nội bộ dự án'}")


def render_tab_collection_explorer() -> None:
    """Tab 2: Read-only corpus browser reading directly from Chroma + isolated mutable demo collection for live CRUD."""
    st.markdown("### 🗄️ Khám Phá & Thao Tác Collection")

    tab_official, tab_mutable = st.tabs([
        "📖 Official Corpus (seminar_corpus_e5) [Chỉ Đọc]",
        "✍️ Demo Collection (seminar_demo_mutable) [CRUD Trực Tiếp]"
    ])

    # -------------------------------------------------------------------------
    # Sub-tab 1: Official Corpus Browser (READ DIRECTLY FROM CHROMA)
    # -------------------------------------------------------------------------
    with tab_official:
        st.markdown("#### Kho Học Liệu Chính Thức (`seminar_corpus_e5`)")
        st.caption("Duyệt trực tiếp các bản ghi và metadata được lưu trữ trong Chroma VectorDB (bất biến, chế độ chỉ đọc).")

        client = get_cached_chroma_client()
        try:
            indexer = CorpusIndexer()
            is_consistent, msg, manifest = indexer.check_index_consistency()
            if not is_consistent:
                st.error(f"❌ Collection '{DEFAULT_COLLECTION_NAME}' chưa đồng bộ hoặc bị lỗi: {msg}")
                st.code("python scripts/build_index.py", language="powershell")
            else:
                col = client.get_collection(DEFAULT_COLLECTION_NAME)
                real_count = col.count()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tổng Tài Liệu (Chroma)", f"{real_count} đoạn")
                m2.metric("Số Chiều Vector", f"{manifest.get('vector_dim', 384)} dims")
                m3.metric("Độ Đo Khoảng Cách", "Cosine")
                m4.metric("Trạng Thái", "🔒 Bất biến (Read-Only)")

                # Read records directly from Chroma without embeddings
                chroma_data = col.get(include=["documents", "metadatas"])
                chroma_records = []
                ids = chroma_data.get("ids", [])
                docs = chroma_data.get("documents", [])
                metas = chroma_data.get("metadatas", [])

                for d_id, doc_text, meta in zip(ids, docs, metas):
                    meta_dict = meta if meta is not None else {}
                    chroma_records.append({
                        "id": d_id,
                        "topic": meta_dict.get("topic", "General"),
                        "title": meta_dict.get("title", ""),
                        "text": doc_text,
                        "source_id": meta_dict.get("source_id", ""),
                        "source_url": meta_dict.get("source_url", ""),
                    })

                # Sort by ID
                chroma_records.sort(key=lambda x: x["id"])
                df_docs = pd.DataFrame(chroma_records)

                # Topic filter for official explorer
                filter_col1, filter_col2 = st.columns([1, 2])
                with filter_col1:
                    topic_sel = st.selectbox("Lọc xem theo chủ đề:", STANDARD_TOPICS, key="official_topic_filter")
                with filter_col2:
                    search_kw = st.text_input("Tìm nhanh trong tiêu đề/nội dung:", "", key="official_kw_search")

                filtered_df = df_docs
                if topic_sel != "All Topics":
                    filtered_df = filtered_df[filtered_df["topic"] == topic_sel]
                if search_kw.strip():
                    kw = search_kw.strip().lower()
                    filtered_df = filtered_df[
                        filtered_df["title"].str.lower().str.contains(kw)
                        | filtered_df["text"].str.lower().str.contains(kw)
                    ]

                st.write(f"Hiển thị **{len(filtered_df)}** / {real_count} đoạn tài liệu từ Chroma DB:")
                st.dataframe(
                    filtered_df[["id", "topic", "title", "text", "source_id"]],
                    use_container_width=True,
                    height=350,
                )
        except Exception as e:
            st.error(f"Lỗi khi đọc official collection từ Chroma: {e}")

    # -------------------------------------------------------------------------
    # Sub-tab 2: Isolated Mutable Demo Collection (CRUD)
    # -------------------------------------------------------------------------
    with tab_mutable:
        st.markdown("#### Collection Thao Tác Trực Tiếp (`seminar_demo_mutable`)")
        st.info(
            "🛡️ **An toàn dữ liệu:** Collection này hoàn toàn độc lập với kho 114 tài liệu chính thức. "
            "Người dùng có thể tự do Thêm (Add), Sửa (Update), Xóa (Delete) và Thử nghiệm Tìm kiếm vector trực tiếp."
        )

        client = get_cached_chroma_client()
        demo_col = get_or_create_mutable_collection(client, DEFAULT_MUTABLE_COLLECTION)
        model = get_cached_embedding_model()

        # Display current count
        current_demo_docs = list_mutable_documents(demo_col)
        st.metric("Số bản ghi hiện có trong Demo Collection:", f"{len(current_demo_docs)} records")

        # Action Selector
        crud_action = st.radio(
            "Chọn thao tác:",
            ["📋 Xem Danh Sách & Tìm Kiếm", "➕ Thêm Tài Liệu Mới", "✏️ Sửa Tài Liệu", "🗑️ Xóa Tài Liệu"],
            horizontal=True,
        )

        # ACTION: LIST & SEARCH
        if crud_action == "📋 Xem Danh Sách & Tìm Kiếm":
            if not current_demo_docs:
                st.warning("Demo Collection hiện đang trống. Hãy chọn '➕ Thêm Tài Liệu Mới' để thêm vài đoạn văn bản thử nghiệm.")
            else:
                st.markdown("##### Danh sách tài liệu hiện có:")
                df_demo = pd.DataFrame(current_demo_docs)
                st.dataframe(df_demo[["id", "topic", "title", "text"]], use_container_width=True)

                st.divider()
                st.markdown("##### 🔎 Thử nghiệm tìm kiếm vector trên Demo Collection:")
                demo_q = st.text_input("Nhập câu hỏi tìm kiếm trên demo collection:", "")
                if st.button("Tìm trên Demo Collection") and demo_q.strip():
                    results = search_mutable_collection(demo_col, model, demo_q.strip(), top_k=3)
                    if results:
                        st.write("Kết quả tìm kiếm láng giềng gần nhất:")
                        for r in results:
                            st.success(f"Hạng {r['rank']}: [{r['id']}] {r['title']} (Cosine Distance: {r['distance']:.4f})")
                            st.write(r["document"])
                    else:
                        st.info("Không tìm thấy kết quả nào.")

        # ACTION: ADD (Refuses existing IDs)
        elif crud_action == "➕ Thêm Tài Liệu Mới":
            st.markdown("##### Thêm tài liệu vào demo collection:")
            with st.form("add_demo_form", clear_on_submit=True):
                new_id = st.text_input("Mã định danh ID (ví dụ: demo_001):")
                new_title = st.text_input("Tiêu đề tài liệu:")
                new_topic = st.selectbox("Chủ đề:", STANDARD_TOPICS[1:])
                new_text = st.text_area("Nội dung đoạn văn bản tiếng Việt:", height=100)
                submit_add = st.form_submit_button("➕ Lưu Vào Collection", type="primary")

                if submit_add:
                    if not new_id.strip() or not new_text.strip():
                        st.error("Vui lòng điền đầy đủ ID và nội dung văn bản.")
                    else:
                        try:
                            add_mutable_document(
                                demo_col, model, new_id.strip(), new_title.strip(), new_topic, new_text.strip()
                            )
                            st.success(f"✅ Đã thêm tài liệu `{new_id}` và tự động tính vector embedding E5 thành công!")
                            st.rerun()
                        except ValueError as ve:
                            st.error(f"❌ {ve}")
                        except Exception as e:
                            st.error(f"Lỗi thêm tài liệu: {e}")

        # ACTION: UPDATE (Requires existing ID)
        elif crud_action == "✏️ Sửa Tài Liệu":
            st.markdown("##### Sửa tài liệu trong demo collection:")
            if not current_demo_docs:
                st.info("Chưa có tài liệu nào trong demo collection để sửa.")
            else:
                doc_options = {d["id"]: d for d in current_demo_docs}
                edit_id = st.selectbox("Chọn ID tài liệu cần sửa:", list(doc_options.keys()))
                doc_to_edit = doc_options[edit_id]

                with st.form("edit_demo_form"):
                    edit_title = st.text_input("Tiêu đề:", value=doc_to_edit["title"])
                    edit_topic = st.text_input("Chủ đề:", value=doc_to_edit["topic"])
                    edit_text = st.text_area("Nội dung văn bản:", value=doc_to_edit["text"], height=100)
                    submit_update = st.form_submit_button("💾 Cập Nhật Tài Liệu", type="primary")

                    if submit_update:
                        try:
                            update_mutable_document(
                                demo_col, model, edit_id, edit_title.strip(), edit_topic.strip(), edit_text.strip()
                            )
                            st.success(f"✅ Đã cập nhật tài liệu `{edit_id}` và tái tính vector embedding tương ứng!")
                            st.rerun()
                        except ValueError as ve:
                            st.error(f"❌ {ve}")
                        except Exception as e:
                            st.error(f"Lỗi cập nhật: {e}")

        # ACTION: DELETE
        elif crud_action == "🗑️ Xóa Tài Liệu":
            st.markdown("##### Xóa tài liệu khỏi demo collection:")
            if not current_demo_docs:
                st.info("Chưa có tài liệu nào trong demo collection để xóa.")
            else:
                del_ids = [d["id"] for d in current_demo_docs]
                target_del_id = st.selectbox("Chọn ID tài liệu cần xóa:", del_ids)

                confirm_del = st.checkbox(f"Tôi xác nhận muốn xóa tài liệu `{target_del_id}` khỏi Demo Collection.")
                if st.button("🗑️ Xác Nhận Xóa", type="primary"):
                    if confirm_del:
                        try:
                            delete_mutable_document(demo_col, target_del_id)
                            st.success(f"✅ Đã xóa tài liệu `{target_del_id}` thành công!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lỗi xóa tài liệu: {e}")
                    else:
                        st.warning("Vui lòng tích chọn ô xác nhận trước khi xóa.")


def render_tab_experiments() -> None:
    """Tab 3: Experiment results, review pool coverage, and official benchmark status."""
    st.markdown("### 📊 Kết Quả Thực Nghiệm & Đánh Giá Truy Hồi")
    st.markdown(
        "Bảng điều khiển theo dõi tiến độ thẩm định nhãn và kết quả đánh giá trên **tập 12 test queries cố định** (G3)."
    )

    # 1. Official Benchmark Gate Status Banner & Verification
    is_official_eligible, gate_msg, official_data = check_official_benchmark_eligibility(
        review_pool_path=DEFAULT_REVIEW_POOL_PATH,
        corpus_path=DEFAULT_DATA_PATH,
        official_json_path=DEFAULT_OFFICIAL_JSON_PATH,
    )

    # 2. Dynamic coverage statistics from review pool
    gate = ReviewPoolGate(DEFAULT_REVIEW_POOL_PATH)
    coverage = gate.get_coverage_statistics()
    total_pairs = coverage.get("total_pairs", 105)
    verified_pairs = coverage.get("verified_pairs", 0)
    completion_pct = coverage.get("completion_percentage", 0.0)

    # 3. Dynamic metadata from preview artifact
    preview_artifact = load_preview_artifact(DEFAULT_PREVIEW_JSON_PATH)
    total_queries = 12
    agreement_str = "N/A"
    if preview_artifact and "agreement" in preview_artifact:
        agr_cnt = preview_artifact["agreement"].get("top1_agreement_count", 0)
        agr_pct = preview_artifact["agreement"].get("top1_agreement_percentage", 0.0)
        total_queries = preview_artifact.get("subset", {}).get("total_queries", 12)
        agreement_str = f"{agr_cnt} / {total_queries} ({agr_pct}%)"

    st.markdown("#### 📈 Thống Kê Tổng Quan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số Câu Hỏi Đánh Giá", f"{total_queries} queries (Test split)")
    c2.metric("Số Cặp Ứng Viên Cần Duyệt", f"{total_pairs} candidate pairs")
    c3.metric("Tỷ Lệ Đồng Thuận Top-1", agreement_str)
    c4.metric("Tiến Độ Thẩm Định Nhãn", f"{verified_pairs} / {total_pairs} ({completion_pct}%)")

    st.divider()

    # 4. Official Benchmark Metrics Table (when Gate PASS) OR Pending Banner
    if is_official_eligible and official_data:
        st.success("✅ **Official Benchmark Đã Thẩm Định Hoàn Tất:** Kết quả đạt chuẩn khoa học với 100% nhãn verified!")

        # Macro-averages comparison table
        st.markdown("#### 🏆 Bảng Điểm Benchmark Chính Thức (Macro-Averages, n = 11 answerable)")
        st.caption("Các chỉ số được tính toán trên 100% nhãn đã được kiểm duyệt bởi con người (loại trừ q_030 unanswerable).")
        macro = official_data.get("macro_averages", {})
        c_m = macro.get("chroma", {})
        b_m = macro.get("bm25", {})
        df_macro = pd.DataFrame([
            {"Chỉ số (Metric)": "Hit@5", "Chroma (E5)": f"{c_m.get('hit@5', 0.0):.4f}", "BM25 (PyVi)": f"{b_m.get('hit@5', 0.0):.4f}", "Chênh lệch": f"{c_m.get('hit@5', 0.0) - b_m.get('hit@5', 0.0):+.4f}"},
            {"Chỉ số (Metric)": "MRR@5", "Chroma (E5)": f"{c_m.get('mrr@5', 0.0):.4f}", "BM25 (PyVi)": f"{b_m.get('mrr@5', 0.0):.4f}", "Chênh lệch": f"{c_m.get('mrr@5', 0.0) - b_m.get('mrr@5', 0.0):+.4f}"},
            {"Chỉ số (Metric)": "nDCG@5", "Chroma (E5)": f"{c_m.get('ndcg@5', 0.0):.4f}", "BM25 (PyVi)": f"{b_m.get('ndcg@5', 0.0):.4f}", "Chênh lệch": f"{c_m.get('ndcg@5', 0.0) - b_m.get('ndcg@5', 0.0):+.4f}"},
        ])
        st.table(df_macro)

        # Per-query official metrics table
        st.markdown("#### 📋 Chi Tiết Điểm Số Theo Từng Câu Hỏi (Per-Query Metrics)")
        per_query_rows = []
        for item in official_data.get("per_query_metrics", []):
            cm = item.get("chroma", {}).get("metrics", {})
            bm = item.get("bm25", {}).get("metrics", {})
            per_query_rows.append({
                "Query ID": item.get("query_id"),
                "Thể loại": item.get("category"),
                "Chroma Hit@5": cm.get("hit@5"),
                "Chroma MRR@5": cm.get("mrr@5"),
                "Chroma nDCG@5": cm.get("ndcg@5"),
                "BM25 Hit@5": bm.get("hit@5"),
                "BM25 MRR@5": bm.get("mrr@5"),
                "BM25 nDCG@5": bm.get("ndcg@5"),
            })
        st.dataframe(pd.DataFrame(per_query_rows), use_container_width=True)
        st.divider()
    else:
        st.warning(
            f"⏳ **Official benchmark is pending human verification.**\n\n"
            f"Chi tiết: {gate_msg}\n\n"
            "Để bảo đảm tính liêm chính khoa học, điểm số chính thức (Hit@5, MRR@5, nDCG@5) chỉ được hiển thị "
            "sau khi người kiểm duyệt thẩm định đầy đủ 105 cặp ứng viên trong `data/eval/review_pool.csv`. "
            "Dưới đây là kết quả đối sánh thứ hạng xem trước (Preview Mode)."
        )

    # 5. Per-query ranking disagreement preview
    if preview_artifact and "eval_results" in preview_artifact:
        st.markdown("#### 🔍 Đối Soát Chi Tiết Từng Câu Hỏi Trong Tập Đánh Giá (n = 12)")

        eval_items = preview_artifact["eval_results"]
        query_dict = {f"{item['query_id']} — {item['query_text'][:45]}...": item for item in eval_items}

        selected_q_label = st.selectbox("Chọn câu hỏi để đối chiếu chi tiết Top-5:", list(query_dict.keys()))
        curr_item = query_dict[selected_q_label]

        st.markdown(f"**Câu hỏi:** *\"{curr_item['query_text']}\"*")
        col_meta1, col_meta2 = st.columns(2)
        col_meta1.markdown(f"**Thể loại:** `{curr_item['category']}`")
        col_meta2.markdown(
            f"**Đồng thuận Top-1:** {'✅ Trùng khớp Top-1' if curr_item['top1_match'] else '⚠️ Khác biệt Top-1'}"
        )

        # Show Top-5 Chroma vs BM25 side-by-side
        q_col_left, q_col_right = st.columns(2)
        with q_col_left:
            st.markdown("##### 🧠 Chroma Top-5")
            df_c = pd.DataFrame(curr_item["chroma_top_k"])
            st.dataframe(df_c, use_container_width=True)

        with q_col_right:
            st.markdown("##### 📖 BM25 Top-5")
            df_b = pd.DataFrame(curr_item["bm25_top_k"])
            st.dataframe(df_b, use_container_width=True)

    # 6. Qualitative Analysis on Unanswerable Query (q_030)
    st.divider()
    st.markdown("#### ❓ Phân Tích Định Tính Câu Hỏi Unanswerable (`q_030`)")
    st.markdown(
        "**Nội dung câu hỏi:** *\"thuật toán đồng thuận Proof of Stake và xử lý phân nhánh blockchain Ethereum\"*\n\n"
        "- **Kết quả Chroma:** Trả về `algo_014` (Thuật toán Prim) với Cosine Distance `0.1513`\n"
        "- **Kết quả BM25:** Trả về `ml_001` (Học có giám sát và không giám sát) với BM25 Score `9.0677`\n\n"
        "**Quan sát & Kết luận khoa học:**\n"
        "Cả hai hệ thống tìm kiếm đều **không có cơ chế từ chối (abstain)**; chúng bắt buộc phải trả về các đoạn "
        "văn bản gần nhất trong kho dữ liệu. Việc vector láng giềng gần nhất trả về kết quả không đồng nghĩa với việc tìm thấy "
        "câu trả lời đúng. Do đó, câu hỏi này được phân tích định tính độc lập và **loại bỏ khỏi mẫu số tính điểm trung bình** "
        "của các chỉ số Hit/MRR/nDCG."
    )


def main() -> None:
    render_sidebar_info()

    st.markdown('<div class="app-main-title">Chroma Semantic Search Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-main-subtitle">Phòng thí nghiệm tìm kiếm ngữ nghĩa tiếng Việt với Chroma VectorDB và Multilingual-E5-Small</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "🔍 Semantic Search",
        "🗄️ Collection Explorer",
        "📊 Experiment Results"
    ])

    with tab1:
        render_tab_semantic_search()

    with tab2:
        render_tab_collection_explorer()

    with tab3:
        render_tab_experiments()


if __name__ == "__main__":
    main()

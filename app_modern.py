"""
Aplikasi Streamlit: Sistem Pencarian Two-Stage Retrieval + Perbandingan terhadap Baseline VSM.
Judul: Optimasi Pencarian Komentar Ketahanan Pangan berbasis Bi-Encoder IndoBERT
       dengan Cosine Similarity dan Reranking Cross-Encoder terhadap Baseline VSM.
"""
import streamlit as st
import pandas as pd
import numpy as np

# Modul Baseline (TF-IDF VSM)
from src.engine import (
    build_inverted_index, calculate_idf, calculate_tfidf_matrix,
    vectorize_query, compute_cosine_similarity
)

# Modul Modern (Two-Stage Retrieval)
from src.semantic_engine import BiEncoderRetriever
from src.reranker import CrossEncoderReranker

# Modul Evaluasi (Klasik + Modern)
from src.evaluation import (
    calculate_all_metrics, calculate_modern_metrics,
    calculate_mrr, calculate_average_ndcg,
    calculate_rr, calculate_ndcg,
    calculate_precision, calculate_recall
)


# ============================================================
# 1. KONFIGURASI HALAMAN & ESTETIKA
# ============================================================
st.set_page_config(
    page_title="IR Modern Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    /* === Variabel Warna (Mendukung Dark/Light Mode Streamlit) === */
    :root {
        --primary-light: #4CAF50;
        --primary-bg: rgba(76, 175, 80, 0.15);
        --accent-blue-light: #42A5F5;
        --accent-blue-bg: rgba(33, 150, 243, 0.15);
        --card-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* === Header === */
    .title-box {
        padding: 30px;
        border: 2px solid var(--primary-light);
        border-radius: 15px;
        background-color: var(--secondary-background-color);
        text-align: center;
        font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif;
        box-shadow: var(--card-shadow);
        margin-bottom: 30px;
    }
    .title-box h1 { color: var(--primary-color, var(--primary-light)); margin-bottom: 10px; font-weight: 800; font-size: 2.2rem; }
    .title-box p  { font-size: 1.1rem; opacity: 0.8; }

    /* === Kartu Dokumen === */
    .doc-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid var(--primary-light);
        box-shadow: var(--card-shadow);
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .doc-card:hover { transform: translateY(-3px); box-shadow: 0 6px 18px rgba(76,175,80,0.2); }

    .doc-card-modern {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid var(--accent-blue-light);
        box-shadow: var(--card-shadow);
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .doc-card-modern:hover { transform: translateY(-3px); box-shadow: 0 6px 18px rgba(33,150,243,0.2); }

    /* === Badge Skor === */
    .score-badge {
        background-color: var(--primary-bg);
        color: var(--primary-light);
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
        border: 1px solid var(--primary-light);
    }
    .score-badge-blue {
        background-color: var(--accent-blue-bg);
        color: var(--accent-blue-light);
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
        border: 1px solid var(--accent-blue-light);
    }

    /* === Blok Laporan === */
    .report-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 25px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 20px;
    }

    /* === Indikator Stage === */
    .stage-label {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8em;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .stage-1 { background: rgba(255, 152, 0, 0.15); color: #FF9800; border: 1px solid rgba(255,152,0,0.3); }
    .stage-2 { background: var(--accent-blue-bg); color: var(--accent-blue-light); border: 1px solid rgba(33,150,243,0.3); }
</style>

<div class="title-box">
    <h1>Mesin Pencari Modern — Ketahanan Pangan 🌾</h1>
    <p><b>Two-Stage Retrieval:</b> Bi-Encoder IndoBERT + Cross-Encoder Reranking</p>
    <p style="font-size:0.9rem; opacity:0.7;">Dikembangkan oleh <b>Andri Darmawan</b> (301210004) & <b>Muhammad Fakhrudin</b> (3012310043)</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 2. DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """Memuat data dari file Excel lokal."""
    try:
        df = pd.read_excel('hasil_vsm_ketahanan_pangan.xlsx')
        if len(df.columns) >= 2:
            df.columns = ['Komentar', 'Sumber'] + list(df.columns[2:])
        else:
            df.columns = ['Komentar']
        df = df.dropna(subset=['Komentar']).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Gagal memuat file Excel: {e}")
        return pd.DataFrame({'Komentar': []})


# ============================================================
# 3. ENGINE INITIALIZATION
# ============================================================

# --- Baseline Engine (TF-IDF VSM) ---
@st.cache_resource
def prepare_baseline_engine(documents):
    """Membangun Inverted Index, IDF, dan Matriks TF-IDF."""
    if not documents:
        return {}, {}, {}, np.array([]), []
    total_docs = len(documents)
    inverted_index, df_dict, processed_docs = build_inverted_index(documents)
    idf_weights = calculate_idf(df_dict, total_docs)
    doc_matrix, vocab = calculate_tfidf_matrix(inverted_index, idf_weights, total_docs)
    return inverted_index, df_dict, idf_weights, doc_matrix, vocab

# --- Modern Engine (Bi-Encoder + Cross-Encoder) ---
@st.cache_resource
def prepare_modern_engine(documents):
    """Menginisialisasi Bi-Encoder IndoBERT dan Cross-Encoder Reranker."""
    retriever = BiEncoderRetriever()
    retriever.encode_documents(documents)
    reranker = CrossEncoderReranker()
    return retriever, reranker


# Load data
df = load_data()
documents = df['Komentar'].astype(str).tolist()

# Initialize engines
with st.spinner('⏳ Menyiapkan Baseline Engine (TF-IDF, Inverted Index)...'):
    inverted_index, df_dict, idf_weights, doc_matrix, vocab = prepare_baseline_engine(documents)

with st.spinner('🧠 Menyiapkan Modern Engine (Bi-Encoder IndoBERT + Cross-Encoder)...'):
    retriever, reranker = prepare_modern_engine(documents)


# ============================================================
# 4. FUNGSI HELPER PENCARIAN
# ============================================================

def search_baseline(query_text: str, top_k: int = 5) -> tuple[list[int], list[float]]:
    """Jalankan pencarian Baseline VSM (TF-IDF + Cosine Similarity)."""
    q_vector = vectorize_query(query_text, vocab, idf_weights)
    scores = compute_cosine_similarity(doc_matrix, q_vector)
    ranked_indices = scores.argsort()[::-1][:top_k]
    ranked_scores = [float(scores[i]) for i in ranked_indices]
    # Filter skor > 0 (pastikan ID dan skor tetap sinkron)
    filtered = [(int(idx), s) for idx, s in zip(ranked_indices, ranked_scores) if s > 0]
    if filtered:
        result_ids, result_scores = zip(*filtered)
        return list(result_ids), list(result_scores)
    return [], []

def search_modern(query_text: str, top_k_stage1: int = 15, top_k_final: int = 5) -> dict:
    """
    Jalankan pencarian Two-Stage (Bi-Encoder → Cross-Encoder).
    Mengembalikan hasil Tahap 1 dan Tahap 2 secara terpisah untuk transparansi.
    """
    # Tahap 1: Bi-Encoder Retrieval
    bi_ids, bi_scores = retriever.retrieve(query_text, top_k=top_k_stage1)
    
    # Tahap 2: Cross-Encoder Reranking
    candidate_docs = [documents[i] for i in bi_ids]
    final_ids, final_scores = reranker.rerank(
        query_text, candidate_docs, bi_ids, top_k=top_k_final
    )
    
    return {
        'stage1_ids': bi_ids,
        'stage1_scores': bi_scores,
        'final_ids': final_ids,
        'final_scores': final_scores,
    }


# ============================================================
# 5. ANTARMUKA PENGGUNA (3 TAB)
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🔍 Pencarian Modern",
    "⚖️ Perbandingan Sistem",
    "📊 Evaluasi Komparatif"
])


# ----------------------------------------------------------
# TAB 1: PENCARIAN MODERN (Two-Stage Retrieval)
# ----------------------------------------------------------
with tab1:
    st.markdown("### Pencarian Two-Stage Retrieval")
    st.caption("Tahap 1: Bi-Encoder IndoBERT (Cosine Similarity) → Tahap 2: Cross-Encoder Reranking")

    col_q, col_k = st.columns([3, 1])
    with col_q:
        query_modern = st.text_input(
            "Masukkan Kueri:", 
            placeholder="Contoh: harga beras stabil murah",
            key="q_modern"
        )
    with col_k:
        top_k_modern = st.slider("Jumlah Hasil Final:", 1, 10, 5, key="k_modern")

    if query_modern:
        with st.spinner('🔄 Menjalankan Two-Stage Retrieval...'):
            results = search_modern(query_modern, top_k_stage1=15, top_k_final=top_k_modern)

        # --- Tampilkan Hasil Tahap 1 (Bi-Encoder) ---
        with st.expander("📋 Tahap 1: Kandidat Bi-Encoder (Top-15)", expanded=False):
            st.markdown('<span class="stage-label stage-1">STAGE 1 — Bi-Encoder</span>', unsafe_allow_html=True)
            for rank, (idx, score) in enumerate(zip(results['stage1_ids'], results['stage1_scores']), 1):
                st.markdown(f"**#{rank}** | Doc ID: {idx} | Skor: `{score:.4f}` — _{df['Komentar'].iloc[idx][:80]}..._")

        # --- Tampilkan Hasil Tahap 2 (Cross-Encoder) ---
        st.markdown("---")
        st.markdown("### Hasil Pencarian Final (Setelah Reranking)")

        if not results['final_ids']:
            st.warning("Tidak ditemukan dokumen relevan.")
        else:
            chart_data = {"Dokumen": [], "Skor": []}
            for rank, (idx, score) in enumerate(zip(results['final_ids'], results['final_scores']), 1):
                teks_asli = df['Komentar'].iloc[idx]
                sumber = df.get('Sumber', pd.Series(['-'] * len(df))).iloc[idx]

                chart_data["Dokumen"].append(f"Doc {idx}")
                chart_data["Skor"].append(score)

                st.markdown(f"""
                    <div class="doc-card-modern">
                        <span class="stage-label stage-2">FINAL — Cross-Encoder Reranked</span>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin: 8px 0;">
                            <h4 style="margin: 0; color: var(--accent-blue-light);">Peringkat #{rank} | Dokumen ID: {idx}</h4>
                            <span class="score-badge-blue">Skor Relevansi: {score:.4f}</span>
                        </div>
                        <p style="font-size: 1.05em; line-height: 1.5; color: var(--text-color);">"{teks_asli}"</p>
                        <small style="opacity: 0.7;"><b>Sumber:</b> {sumber}</small>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("### Visualisasi Skor")
            df_chart = pd.DataFrame(chart_data).set_index("Dokumen")
            st.bar_chart(df_chart, use_container_width=True, color="#42A5F5")


# ----------------------------------------------------------
# TAB 2: PERBANDINGAN BASELINE vs MODERN
# ----------------------------------------------------------
with tab2:
    st.markdown("### Perbandingan: Baseline VSM vs Two-Stage Retrieval")
    st.caption("Masukkan kueri yang sama untuk melihat perbedaan hasil kedua sistem secara berdampingan.")

    query_compare = st.text_input(
        "Masukkan Kueri untuk Perbandingan:",
        placeholder="Contoh: stabilitas pasokan pangan",
        key="q_compare"
    )
    top_k_compare = st.slider("Jumlah Hasil per Sistem:", 1, 10, 5, key="k_compare")

    pilihan_tampilan = st.radio(
        "Pilih Sistem yang Ditampilkan:",
        options=["Keduanya (Berdampingan)", "Hanya Baseline VSM (TF-IDF)", "Hanya Modern Two-Stage"],
        horizontal=True
    )

    if query_compare:
        with st.spinner('🔄 Menjalankan sistem...'):
            baseline_ids, baseline_scores = search_baseline(query_compare, top_k=top_k_compare)
            modern_results = search_modern(query_compare, top_k_stage1=15, top_k_final=top_k_compare)

        if pilihan_tampilan == "Keduanya (Berdampingan)":
            col_baseline, col_modern = st.columns(2)
            
            # --- Kolom Kiri: Baseline VSM ---
            with col_baseline:
                st.markdown("#### 🟢 Baseline VSM (TF-IDF)")
                if not baseline_ids:
                    st.warning("Tidak ditemukan dokumen relevan.")
                else:
                    for rank, (idx, score) in enumerate(zip(baseline_ids, baseline_scores), 1):
                        teks = df['Komentar'].iloc[idx]
                        st.markdown(f"""
                            <div class="doc-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <b style="color: var(--primary-light);">#{rank} | Doc {idx}</b>
                                    <span class="score-badge">{score:.4f}</span>
                                </div>
                                <p style="font-size: 0.95em; color: var(--text-color);">"{teks}"</p>
                            </div>
                        """, unsafe_allow_html=True)

            # --- Kolom Kanan: Modern Two-Stage ---
            with col_modern:
                st.markdown("#### 🔵 Modern Two-Stage (IndoBERT + Cross-Encoder)")
                if not modern_results['final_ids']:
                    st.warning("Tidak ditemukan dokumen relevan.")
                else:
                    for rank, (idx, score) in enumerate(zip(modern_results['final_ids'], modern_results['final_scores']), 1):
                        teks = df['Komentar'].iloc[idx]
                        st.markdown(f"""
                            <div class="doc-card-modern">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <b style="color: var(--accent-blue-light);">#{rank} | Doc {idx}</b>
                                    <span class="score-badge-blue">{score:.4f}</span>
                                </div>
                                <p style="font-size: 0.95em; color: var(--text-color);">"{teks}"</p>
                            </div>
                        """, unsafe_allow_html=True)

        elif pilihan_tampilan == "Hanya Baseline VSM (TF-IDF)":
            st.markdown("#### 🟢 Baseline VSM (TF-IDF)")
            if not baseline_ids:
                st.warning("Tidak ditemukan dokumen relevan.")
            else:
                for rank, (idx, score) in enumerate(zip(baseline_ids, baseline_scores), 1):
                    teks = df['Komentar'].iloc[idx]
                    st.markdown(f"""
                        <div class="doc-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <b style="color: var(--primary-light);">#{rank} | Doc {idx}</b>
                                <span class="score-badge">{score:.4f}</span>
                            </div>
                            <p style="font-size: 0.95em; color: var(--text-color);">"{teks}"</p>
                        </div>
                    """, unsafe_allow_html=True)

        elif pilihan_tampilan == "Hanya Modern Two-Stage":
            st.markdown("#### 🔵 Modern Two-Stage (IndoBERT + Cross-Encoder)")
            if not modern_results['final_ids']:
                st.warning("Tidak ditemukan dokumen relevan.")
            else:
                for rank, (idx, score) in enumerate(zip(modern_results['final_ids'], modern_results['final_scores']), 1):
                    teks = df['Komentar'].iloc[idx]
                    st.markdown(f"""
                        <div class="doc-card-modern">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <b style="color: var(--accent-blue-light);">#{rank} | Doc {idx}</b>
                                <span class="score-badge-blue">{score:.4f}</span>
                            </div>
                            <p style="font-size: 0.95em; color: var(--text-color);">"{teks}"</p>
                        </div>
                    """, unsafe_allow_html=True)

        # --- Tabel Perbandingan Peringkat ---
        if pilihan_tampilan == "Keduanya (Berdampingan)":
            st.markdown("---")
            st.markdown("#### Tabel Perbandingan Urutan Peringkat")
            max_rows = max(len(baseline_ids), len(modern_results['final_ids']))
            comparison_data = []
            for i in range(max_rows):
                row = {"Peringkat": i + 1}
                if i < len(baseline_ids):
                    row["Baseline Doc ID"] = baseline_ids[i]
                    row["Baseline Skor"] = f"{baseline_scores[i]:.4f}"
                else:
                    row["Baseline Doc ID"] = "-"
                    row["Baseline Skor"] = "-"
                if i < len(modern_results['final_ids']):
                    row["Modern Doc ID"] = modern_results['final_ids'][i]
                    row["Modern Skor"] = f"{modern_results['final_scores'][i]:.4f}"
                else:
                    row["Modern Doc ID"] = "-"
                    row["Modern Skor"] = "-"
                comparison_data.append(row)
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)


# ----------------------------------------------------------
# TAB 3: EVALUASI KOMPARATIF (MRR & NDCG)
# ----------------------------------------------------------
with tab3:
    st.markdown("### Evaluasi Komparatif: Baseline VSM vs Two-Stage Retrieval")

    eval_mode = st.radio(
        "Pilih Mode Evaluasi:",
        ["📋 Evaluasi 5 Kueri Riset (Otomatis)", "🧪 Uji Kueri Kustom (Interaktif)"],
        horizontal=True,
        key="eval_mode_radio"
    )

    st.markdown("---")

    # ==========================================
    # MODE 1: EVALUASI 5 KUERI RISET (TETAP)
    # ==========================================
    if eval_mode == "📋 Evaluasi 5 Kueri Riset (Otomatis)":
        st.markdown("""
        <div class="report-card">
            <b>Evaluasi Otomatis — 5 Skenario Kueri Riset</b><br>
            Evaluasi ini menggunakan 5 kueri uji beserta <i>Ground Truth</i> yang telah ditentukan
            secara manual dalam metodologi penelitian. Ground Truth bersifat tetap dan tidak dapat diubah
            karena merupakan bagian dari desain eksperimen.<br><br>
            Klik tombol di bawah untuk menjalankan evaluasi otomatis.
        </div>
        """, unsafe_allow_html=True)

        EVAL_SCENARIOS = [
            {"query": "harga beras murah", "gt": [26, 41]},
            {"query": "stabilitas pasokan pangan", "gt": [3, 4]},
            {"query": "kemenko pangan kebijakan", "gt": [3, 7, 8, 47]},
            {"query": "stok beras aman", "gt": [2, 5, 8, 9]},
            {"query": "petani kesejahteraan rakyat", "gt": [0, 6]},
        ]

        st.markdown("#### Skenario Kueri & Ground Truth")
        scenario_display = []
        for i, s in enumerate(EVAL_SCENARIOS, 1):
            scenario_display.append({
                "No": f"Q{i}",
                "Kueri": s["query"],
                "Ground Truth (ID Dokumen)": str(s["gt"]),
                "Jumlah Dok. Relevan": len(s["gt"]),
            })
        st.dataframe(pd.DataFrame(scenario_display), use_container_width=True, hide_index=True)

        if st.button("🚀 Jalankan Evaluasi 5 Kueri Riset", use_container_width=True, key="btn_static_eval"):
            with st.spinner("⏳ Menjalankan evaluasi pada kedua sistem..."):
                all_baseline_retrieved = []
                all_modern_retrieved = []
                all_ground_truths = []
                detail_rows = []

                for i, scenario in enumerate(EVAL_SCENARIOS, 1):
                    q = scenario["query"]
                    gt_ids = scenario["gt"]
                    all_ground_truths.append(gt_ids)

                    b_ids, b_scores = search_baseline(q, top_k=5)
                    all_baseline_retrieved.append(b_ids)

                    m_results = search_modern(q, top_k_stage1=15, top_k_final=5)
                    m_ids = m_results['final_ids']
                    all_modern_retrieved.append(m_ids)

                    b_metrics = calculate_modern_metrics(b_ids, gt_ids, k=5)
                    m_metrics = calculate_modern_metrics(m_ids, gt_ids, k=5)

                    detail_rows.append({
                        "Skenario": f"Q{i}",
                        "Kueri": q,
                        "Ground Truth": str(gt_ids),
                        "Baseline RR": f"{b_metrics['rr']:.4f}",
                        "Baseline NDCG@5": f"{b_metrics['ndcg']:.4f}",
                        "Modern RR": f"{m_metrics['rr']:.4f}",
                        "Modern NDCG@5": f"{m_metrics['ndcg']:.4f}",
                    })

                baseline_mrr = calculate_mrr(all_baseline_retrieved, all_ground_truths)
                modern_mrr = calculate_mrr(all_modern_retrieved, all_ground_truths)
                baseline_ndcg = calculate_average_ndcg(all_baseline_retrieved, all_ground_truths, k=5)
                modern_ndcg = calculate_average_ndcg(all_modern_retrieved, all_ground_truths, k=5)

            st.session_state['eval_static'] = {
                'detail_rows': detail_rows,
                'baseline_mrr': baseline_mrr,
                'modern_mrr': modern_mrr,
                'baseline_ndcg': baseline_ndcg,
                'modern_ndcg': modern_ndcg,
            }

        if 'eval_static' in st.session_state:
            r = st.session_state['eval_static']

            st.markdown("#### Detail Evaluasi Per Kueri")
            st.dataframe(pd.DataFrame(r['detail_rows']), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### Ringkasan Metrik Keseluruhan")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🟢 Baseline MRR", f"{r['baseline_mrr']:.4f}")
                st.metric("🟢 Baseline Avg NDCG@5", f"{r['baseline_ndcg']:.4f}")
            with col2:
                mrr_d = r['modern_mrr'] - r['baseline_mrr']
                ndcg_d = r['modern_ndcg'] - r['baseline_ndcg']
                st.metric("🔵 Modern MRR", f"{r['modern_mrr']:.4f}", delta=f"{mrr_d:+.4f}")
                st.metric("🔵 Modern Avg NDCG@5", f"{r['modern_ndcg']:.4f}", delta=f"{ndcg_d:+.4f}")

            st.markdown("---")
            st.markdown("#### Kesimpulan Evaluasi")
            mrr_d = r['modern_mrr'] - r['baseline_mrr']
            ndcg_d = r['modern_ndcg'] - r['baseline_ndcg']
            if r['modern_mrr'] > r['baseline_mrr'] and r['modern_ndcg'] > r['baseline_ndcg']:
                st.success(
                    f"✅ **Sistem Modern (Two-Stage Retrieval) mengungguli Baseline VSM** pada kedua metrik. "
                    f"MRR meningkat **{mrr_d:+.4f}** dan NDCG@5 meningkat **{ndcg_d:+.4f}**. "
                    f"Hal ini membuktikan bahwa pendekatan Semantic Search dengan Bi-Encoder IndoBERT "
                    f"dan Reranking Cross-Encoder berhasil mengatasi masalah *vocabulary mismatch* "
                    f"pada metode TF-IDF tradisional."
                )
            elif r['modern_mrr'] > r['baseline_mrr'] or r['modern_ndcg'] > r['baseline_ndcg']:
                st.info(
                    f"⚠️ **Sistem Modern mengungguli Baseline pada sebagian metrik.** "
                    f"MRR: {r['baseline_mrr']:.4f} → {r['modern_mrr']:.4f} ({mrr_d:+.4f}). "
                    f"NDCG@5: {r['baseline_ndcg']:.4f} → {r['modern_ndcg']:.4f} ({ndcg_d:+.4f}). "
                    f"Perlu analisis lebih lanjut terhadap skenario kueri dan kualitas ground truth."
                )
            else:
                st.warning(
                    f"⚠️ **Baseline VSM masih setara atau lebih baik pada metrik ini.** "
                    f"Kemungkinan penyebab: ground truth kurang representatif, "
                    f"atau dataset terlalu kecil (50 dokumen) sehingga TF-IDF masih cukup efektif. "
                    f"Disarankan untuk memperbanyak kueri uji dan memperbaiki anotasi ground truth."
                )

    # ==========================================
    # MODE 2: UJI KUERI KUSTOM (INTERAKTIF)
    # ==========================================
    else:
        st.markdown("""
        <div class="report-card">
            <b>Mode Interaktif — Uji Kueri Kustom (Adjudication System)</b><br>
            Masukkan kueri pencarian baru. Sistem akan menjalankan kedua metode pencarian,
            lalu menampilkan dokumen-dokumen yang ditemukan.<br><br>
            <b>Cara Penggunaan:</b><br>
            1. Ketik kueri pencarian pada kolom di bawah.<br>
            2. Baca setiap dokumen hasil pencarian yang muncul.<br>
            3. <b>Centang</b> dokumen yang menurut Anda relevan dengan kueri tersebut.<br>
            4. Klik tombol <b>"Hitung Evaluasi"</b> untuk melihat perbandingan metrik.<br><br>
            <i>Mode ini memungkinkan penguji untuk menilai relevansi dokumen secara langsung
            tanpa perlu menghafal seluruh isi dataset.</i>
        </div>
        """, unsafe_allow_html=True)

        custom_query = st.text_input(
            "Masukkan Kueri Baru untuk Diuji:",
            placeholder="Contoh: kebijakan bantuan untuk petani",
            key="custom_eval_query"
        )

        if custom_query:
            # Cache search results to avoid re-running model on every interaction
            cache_key = '_custom_search_cache'
            if (cache_key not in st.session_state or
                st.session_state[cache_key].get('query') != custom_query):
                with st.spinner("🔄 Menjalankan pencarian pada kedua sistem..."):
                    _b_ids, _b_scores = search_baseline(custom_query, top_k=5)
                    _m_results = search_modern(custom_query, top_k_stage1=15, top_k_final=5)
                    st.session_state[cache_key] = {
                        'query': custom_query,
                        'b_ids': _b_ids,
                        'b_scores': _b_scores,
                        'm_ids': _m_results['final_ids'],
                        'm_scores': _m_results['final_scores'],
                    }

            cached = st.session_state[cache_key]
            b_ids = cached['b_ids']
            b_scores = cached['b_scores']
            m_ids = cached['m_ids']
            m_scores = cached['m_scores']

            # Collect unique document IDs from both systems (preserve order)
            all_candidate_ids = list(dict.fromkeys(b_ids + m_ids))

            st.markdown("---")
            st.markdown("#### 📝 Langkah 1: Tentukan Ground Truth")

            gt_input_method = st.radio(
                "Pilih Metode Penentuan Ground Truth:",
                ["☑️ Centang dari Hasil Pencarian (Rekomendasi)", "✍️ Masukkan ID Dokumen secara Manual"],
                horizontal=True,
                key="gt_input_method_radio"
            )

            if gt_input_method == "☑️ Centang dari Hasil Pencarian (Rekomendasi)":
                st.caption(
                    f"Ditemukan {len(all_candidate_ids)} dokumen unik dari kedua sistem. "
                    f"Baca setiap dokumen, lalu centang jika relevan dengan kueri \"{custom_query}\"."
                )

                # Form-based adjudication (prevents page rerun on every checkbox toggle)
                with st.form("adjudication_form"):
                    for doc_id in all_candidate_ids:
                        teks = df['Komentar'].iloc[doc_id]
                        source_tags = []
                        if doc_id in b_ids:
                            source_tags.append("🟢 Ditemukan oleh Baseline")
                        if doc_id in m_ids:
                            source_tags.append("🔵 Ditemukan oleh Modern")
                        source_str = " | ".join(source_tags)

                        st.markdown(f"""
                        <div style="background: var(--secondary-background-color); padding: 15px; border-radius: 10px;
                                    border-left: 5px solid rgba(128,128,128,0.3); margin-bottom: 4px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <b>Dokumen ID: {doc_id}</b>
                                <span style="font-size: 0.8em; opacity: 0.6;">{source_str}</span>
                            </div>
                            <p style="margin: 8px 0; font-size: 1.0em; line-height: 1.5;">"{teks}"</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.checkbox(
                            f"✅ Tandai Dokumen {doc_id} sebagai RELEVAN",
                            key=f"adj_{custom_query}_{doc_id}",
                        )

                    submitted = st.form_submit_button(
                        "📊 Hitung Evaluasi Berdasarkan Penilaian Anda",
                        use_container_width=True
                    )

                if submitted:
                    gt_ids_custom = [
                        doc_id for doc_id in all_candidate_ids
                        if st.session_state.get(f"adj_{custom_query}_{doc_id}", False)
                    ]
                    if not gt_ids_custom:
                        st.error("⚠️ Anda belum mencentang dokumen relevan manapun. Centang minimal 1 dokumen.")
                    else:
                        st.session_state['custom_eval_done'] = {
                            'query': custom_query,
                            'gt_ids': gt_ids_custom,
                            'b_ids': b_ids,
                            'b_scores': b_scores,
                            'm_ids': m_ids,
                            'm_scores': m_scores,
                        }
            else:
                # Manual entry method
                st.caption(
                    "Masukkan ID dokumen yang menurut Anda relevan secara manual. "
                    f"Dataset Anda berisi dokumen dengan ID 0 sampai {len(df)-1}."
                )

                with st.form("manual_gt_form"):
                    gt_manual_input = st.text_input(
                        "Masukkan ID Ground Truth (pisahkan dengan koma, contoh: 26, 41):",
                        placeholder="Contoh: 26, 41",
                        key="gt_manual_input_val"
                    )

                    submitted_manual = st.form_submit_button(
                        "📊 Hitung Evaluasi Berdasarkan ID Manual",
                        use_container_width=True
                    )

                if submitted_manual:
                    if not gt_manual_input.strip():
                        st.error("⚠️ Silakan masukkan minimal satu ID dokumen.")
                    else:
                        parsed_ids = []
                        invalid_tokens = []
                        out_of_bounds = []

                        tokens = [x.strip() for x in gt_manual_input.split(',') if x.strip()]
                        for t in tokens:
                            if t.isdigit():
                                val = int(t)
                                if 0 <= val < len(df):
                                    parsed_ids.append(val)
                                else:
                                    out_of_bounds.append(val)
                            else:
                                invalid_tokens.append(t)

                        if invalid_tokens:
                            st.error(f"❌ Input tidak valid: {', '.join(invalid_tokens)}. Harap masukkan angka saja.")
                        elif out_of_bounds:
                            st.error(f"❌ ID dokumen di luar jangkauan: {', '.join(map(str, out_of_bounds))}. Dataset hanya berisi ID 0 sampai {len(df)-1}.")
                        elif not parsed_ids:
                            st.error("⚠️ Tidak ada ID dokumen valid yang terdeteksi.")
                        else:
                            st.session_state['custom_eval_done'] = {
                                'query': custom_query,
                                'gt_ids': parsed_ids,
                                'b_ids': b_ids,
                                'b_scores': b_scores,
                                'm_ids': m_ids,
                                'm_scores': m_scores,
                            }

            # Display results if evaluation has been performed for this query
            if ('custom_eval_done' in st.session_state and
                st.session_state['custom_eval_done']['query'] == custom_query):
                res = st.session_state['custom_eval_done']
                gt = res['gt_ids']

                st.markdown("---")
                st.markdown("#### 📊 Langkah 2: Hasil Perbandingan Evaluasi")
                st.success(
                    f"**Ground Truth Anda:** Dokumen ID **{gt}** — "
                    f"Total {len(gt)} dokumen relevan"
                )

                col_b, col_m = st.columns(2)

                with col_b:
                    st.markdown("##### 🟢 Baseline VSM (TF-IDF)")
                    if not res['b_ids']:
                        st.warning("Tidak ada hasil pencarian.")
                    else:
                        for rank, (idx, score) in enumerate(zip(res['b_ids'], res['b_scores']), 1):
                            teks = df['Komentar'].iloc[idx]
                            is_rel = idx in gt
                            rel_icon = "✅ RELEVAN" if is_rel else "❌ TIDAK RELEVAN"
                            border_color = "#4CAF50" if is_rel else "#f44336"
                            st.markdown(f"""
                                <div style="background: var(--secondary-background-color); padding: 12px;
                                            border-radius: 10px; border-left: 5px solid {border_color}; margin-bottom: 8px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <b style="color: var(--primary-light);">#{rank} | Doc {idx}</b>
                                        <span class="score-badge">{score:.4f}</span>
                                    </div>
                                    <p style="font-size: 0.9em; margin: 6px 0; line-height: 1.4;">"{teks}"</p>
                                    <b style="font-size: 0.85em;">{rel_icon}</b>
                                </div>
                            """, unsafe_allow_html=True)

                with col_m:
                    st.markdown("##### 🔵 Modern Two-Stage")
                    if not res['m_ids']:
                        st.warning("Tidak ada hasil pencarian.")
                    else:
                        for rank, (idx, score) in enumerate(zip(res['m_ids'], res['m_scores']), 1):
                            teks = df['Komentar'].iloc[idx]
                            is_rel = idx in gt
                            rel_icon = "✅ RELEVAN" if is_rel else "❌ TIDAK RELEVAN"
                            border_color = "#42A5F5" if is_rel else "#f44336"
                            st.markdown(f"""
                                <div style="background: var(--secondary-background-color); padding: 12px;
                                            border-radius: 10px; border-left: 5px solid {border_color}; margin-bottom: 8px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <b style="color: var(--accent-blue-light);">#{rank} | Doc {idx}</b>
                                        <span class="score-badge-blue">{score:.4f}</span>
                                    </div>
                                    <p style="font-size: 0.9em; margin: 6px 0; line-height: 1.4;">"{teks}"</p>
                                    <b style="font-size: 0.85em;">{rel_icon}</b>
                                </div>
                            """, unsafe_allow_html=True)

                # Metrics comparison
                st.markdown("---")
                st.markdown("#### 📈 Metrik Evaluasi untuk Kueri Ini")

                b_prec = calculate_precision(res['b_ids'], gt)
                b_rec = calculate_recall(res['b_ids'], gt)
                b_met = calculate_modern_metrics(res['b_ids'], gt, k=5)
                m_prec = calculate_precision(res['m_ids'], gt)
                m_rec = calculate_recall(res['m_ids'], gt)
                m_met = calculate_modern_metrics(res['m_ids'], gt, k=5)

                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.markdown("**🟢 Baseline VSM**")
                    st.metric("Precision", f"{b_prec:.2%}")
                    st.metric("Recall", f"{b_rec:.2%}")
                    st.metric("Reciprocal Rank", f"{b_met['rr']:.4f}")
                    st.metric("NDCG@5", f"{b_met['ndcg']:.4f}")
                with col_met2:
                    st.markdown("**🔵 Modern Two-Stage**")
                    st.metric("Precision", f"{m_prec:.2%}", delta=f"{m_prec - b_prec:+.2%}")
                    st.metric("Recall", f"{m_rec:.2%}", delta=f"{m_rec - b_rec:+.2%}")
                    st.metric("Reciprocal Rank", f"{m_met['rr']:.4f}", delta=f"{m_met['rr'] - b_met['rr']:+.4f}")
                    st.metric("NDCG@5", f"{m_met['ndcg']:.4f}", delta=f"{m_met['ndcg'] - b_met['ndcg']:+.4f}")

                # Conclusion for this query
                st.markdown("---")
                if m_met['ndcg'] > b_met['ndcg'] and m_prec >= b_prec:
                    st.success(
                        f"✅ **Sistem Modern lebih unggul** untuk kueri \"{custom_query}\". "
                        f"Modern berhasil menempatkan lebih banyak dokumen relevan di peringkat atas."
                    )
                elif m_met['ndcg'] < b_met['ndcg']:
                    st.warning(
                        f"⚠️ **Baseline VSM lebih unggul** untuk kueri \"{custom_query}\". "
                        f"Kemungkinan kata kunci kueri ini muncul secara eksplisit di dokumen relevan, "
                        f"sehingga TF-IDF sudah cukup efektif."
                    )
                else:
                    st.info(
                        f"ℹ️ **Kedua sistem menunjukkan performa setara** untuk kueri \"{custom_query}\"."
                    )


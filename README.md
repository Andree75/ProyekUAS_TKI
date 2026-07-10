# Mesin Pencari Modern — Ketahanan Pangan 🌾

**Judul Proyek:** Optimasi Pencarian Komentar Ketahanan Pangan berbasis Bi-Encoder IndoBERT dengan Cosine Similarity dan Reranking Cross-Encoder terhadap Baseline VSM

---

## Tim Pengembang

| Nama               | NIM        |
| ------------------ | ---------- |
| Andri Darmawan     | 301210004  |
| Muhammad Fakhrudin | 3012310043 |

---

## Prasyarat

- Python 3.9 atau lebih baru
- pip (package manager Python)
- Koneksi internet (untuk download model pertama kali)
- RAM minimal 4GB

Cek versi Python:

```bash
python --version
```

---

## Inisialisasi Awal

### 1. Buat Virtual Environment (Disarankan)

Virtual environment menjaga sistem Python utama tetap bersih.

```bash
# Buat environment baru
python -m venv venv

# Aktifkan environment
# Linux / Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

> Setelah aktif, terminal akan menampilkan `(venv)` di awal baris.

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

> **Catatan:** Proses ini akan mengunduh PyTorch (~2GB) dan library lainnya. Pastikan koneksi internet stabil.

### 3. Jalankan Aplikasi

```bash
# Sistem Modern (Two-Stage Retrieval + Perbandingan + Evaluasi)
streamlit run app_modern.py

# Sistem Baseline lama (TF-IDF VSM saja)
streamlit run app.py
```

> **Pertama kali menjalankan `app_modern.py`**, sistem akan otomatis mengunduh model IndoBERT dan Cross-Encoder dari HuggingFace (~500MB). Proses ini hanya terjadi sekali — selanjutnya model di-cache secara lokal.

Setelah berhasil, browser akan terbuka otomatis di `http://localhost:8501`.

---

## Struktur Proyek

```
├── app.py                 # Aplikasi Baseline (TF-IDF VSM)
├── app_modern.py           # Aplikasi Modern (Two-Stage Retrieval)
├── requirements.txt        # Daftar dependensi Python
├── hasil_vsm_ketahanan_pangan.xlsx  # Dataset (50 komentar)
├── README.md               # File ini
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Text cleaning (Sastrawi stemmer)
│   ├── engine.py           # Baseline: TF-IDF, Inverted Index, Cosine Similarity
│   ├── evaluation.py       # Metrik: Precision, Recall, F1, MRR, NDCG
│   ├── semantic_engine.py  # Tahap 1: Bi-Encoder IndoBERT + Cosine Similarity
│   └── reranker.py         # Tahap 2: Cross-Encoder Reranking
```

---

## Fitur Aplikasi Modern (`app_modern.py`)

| Tab                     | Fungsi                                            |
| ----------------------- | ------------------------------------------------- |
| **Pencarian Modern**    | Pencarian Two-Stage: Bi-Encoder → Cross-Encoder   |
| **Perbandingan Sistem** | Hasil Baseline vs Modern ditampilkan berdampingan |
| **Evaluasi Komparatif** | Hitung MRR dan NDCG untuk kedua sistem            |

---

## Membersihkan Setelah Selesai

```bash
# Nonaktifkan virtual environment
deactivate

# Hapus seluruh environment (opsional)
# Linux / Mac:
rm -rf venv

# Windows:
rmdir /s /q venv
```

"""
Module: reranker
Description: Implementasi Cross-Encoder untuk Reranking hasil Bi-Encoder (Tahap 2).
             Cross-Encoder memproses pasangan [Kueri, Dokumen] secara bersamaan
             untuk menghasilkan skor relevansi yang lebih presisi.
"""
import numpy as np
from sentence_transformers import CrossEncoder


# Model Cross-Encoder default (Multilingual, mendukung Bahasa Indonesia)
DEFAULT_CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


class CrossEncoderReranker:
    """
    Reranker Tahap 2: Menggunakan Cross-Encoder untuk mengurutkan ulang
    kandidat dokumen dari Tahap 1 (Bi-Encoder) secara lebih presisi.
    
    Mekanisme:
        Input  → [CLS] Teks Kueri [SEP] Teks Dokumen Kandidat [SEP]
        Output → Skor relevansi (float)
    """
    
    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        """
        Inisialisasi model Cross-Encoder.
        
        Args:
            model_name (str): Nama atau path model Cross-Encoder.
        """
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
    
    def rerank(
        self,
        query: str,
        candidate_docs: list[str],
        candidate_ids: list[int],
        top_k: int = None
    ) -> tuple[list[int], list[float]]:
        """
        Mengurutkan ulang (Rerank) dokumen kandidat berdasarkan skor Cross-Encoder.
        
        Pipeline:
            1. Buat pasangan [Kueri, Dokumen] untuk setiap kandidat
            2. Cross-Encoder prediksi skor relevansi tiap pasangan
            3. Urutkan ulang berdasarkan skor tertinggi
        
        Args:
            query (str): Teks kueri pengguna.
            candidate_docs (list[str]): Daftar teks dokumen kandidat dari Tahap 1.
            candidate_ids (list[int]): Daftar ID dokumen kandidat.
            top_k (int, optional): Jumlah hasil akhir. Jika None, kembalikan semua.
            
        Returns:
            tuple:
                - reranked_ids (list[int]): ID dokumen setelah di-rerank.
                - reranked_scores (list[float]): Skor relevansi Cross-Encoder.
        """
        if not candidate_docs:
            return [], []
        
        # 1. Buat pasangan [Kueri, Dokumen Kandidat]
        pairs = [[query, doc] for doc in candidate_docs]
        
        # 2. Prediksi skor relevansi menggunakan Cross-Encoder
        scores = self.model.predict(pairs)
        
        # 3. Urutkan berdasarkan skor tertinggi
        sorted_indices = np.argsort(scores)[::-1]
        
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
        
        reranked_ids = [candidate_ids[i] for i in sorted_indices]
        reranked_scores = [float(scores[i]) for i in sorted_indices]
        
        return reranked_ids, reranked_scores

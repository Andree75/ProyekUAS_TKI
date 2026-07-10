"""
Module: semantic_engine
Description: Implementasi Bi-Encoder IndoBERT untuk Dense Retrieval
             dengan Cosine Similarity dari Scikit-Learn (Tahap 1).
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Model Bi-Encoder default (IndoBERT untuk Sentence Similarity)
DEFAULT_BI_ENCODER_MODEL = 'firqaaa/indo-sentence-bert-base'


class BiEncoderRetriever:
    """
    Retriever Tahap 1: Menggunakan Bi-Encoder (IndoBERT) untuk mengubah
    teks menjadi Dense Vector, lalu menghitung kedekatan menggunakan
    Cosine Similarity dari Scikit-Learn.
    """
    
    def __init__(self, model_name: str = DEFAULT_BI_ENCODER_MODEL):
        """
        Inisialisasi model Bi-Encoder.
        
        Args:
            model_name (str): Nama atau path model Sentence Transformer.
        """
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.model_name = model_name
    
    def encode_documents(self, documents: list[str]) -> np.ndarray:
        """
        Mengubah seluruh dokumen menjadi Dense Vectors (Proses Offline/Indexing).
        Vektor disimpan di memori sebagai atribut instance.
        
        Args:
            documents (list[str]): Daftar teks dokumen mentah.
            
        Returns:
            np.ndarray: Matriks embedding dokumen berukuran (N x dimensi_embedding).
        """
        self.doc_embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return self.doc_embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Mengubah kueri pengguna menjadi Dense Vector (Proses Online).
        
        Args:
            query (str): Teks kueri pengguna.
            
        Returns:
            np.ndarray: Vektor embedding kueri berukuran (1 x dimensi_embedding).
        """
        return self.model.encode([query], convert_to_numpy=True)
    
    def retrieve(self, query: str, top_k: int = 15) -> tuple[list[int], list[float]]:
        """
        Mencari Top-K dokumen terdekat menggunakan Cosine Similarity (Proses Online).
        
        Pipeline:
            1. Encode kueri → Dense Vector
            2. Cosine Similarity (Scikit-Learn) kueri vs seluruh dokumen
            3. Urutkan dari skor tertinggi, ambil Top-K
        
        Args:
            query (str): Teks kueri pengguna.
            top_k (int): Jumlah kandidat dokumen yang dikembalikan.
            
        Returns:
            tuple:
                - ranked_indices (list[int]): ID dokumen terurut dari paling relevan.
                - ranked_scores (list[float]): Skor Cosine Similarity per dokumen.
        """
        if self.doc_embeddings is None:
            raise ValueError(
                "Dokumen belum di-encode. Panggil encode_documents() terlebih dahulu."
            )
        
        # 1. Encode kueri menjadi Dense Vector
        query_embedding = self.encode_query(query)
        
        # 2. Hitung Cosine Similarity (Scikit-Learn) antara kueri vs seluruh dokumen
        scores = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        # 3. Urutkan dari skor tertinggi, ambil Top-K
        ranked_indices = scores.argsort()[::-1][:top_k]
        ranked_scores = scores[ranked_indices]
        
        return ranked_indices.tolist(), ranked_scores.tolist()
    
    def get_embedding_dim(self) -> int:
        """Mengembalikan dimensi embedding model (misal 768)."""
        return self.model.get_sentence_embedding_dimension()

"""
Module: evaluation
Description: Menghitung metrik performa Information Retrieval.
             - Metrik Klasik: Precision, Recall, F-Measure (untuk Baseline)
             - Metrik Modern: RR, MRR, DCG, NDCG (untuk Sistem Baru)
"""
import math


# ============================================================
# METRIK KLASIK (Digunakan oleh Baseline VSM)
# ============================================================

def calculate_precision(retrieved_docs: list[int], ground_truth_docs: list[int]) -> float:
    """
    Menghitung Precision.
    Precision = (Jumlah dokumen relevan yang terambil) / (Total dokumen yang terambil)
    """
    if not retrieved_docs:
        return 0.0
    
    # Mencari irisan (dokumen yang terambil DAN memang relevan/ada di ground truth)
    relevant_retrieved = set(retrieved_docs).intersection(set(ground_truth_docs))
    
    precision = len(relevant_retrieved) / len(retrieved_docs)
    return precision


def calculate_recall(retrieved_docs: list[int], ground_truth_docs: list[int]) -> float:
    """
    Menghitung Recall.
    Recall = (Jumlah dokumen relevan yang terambil) / (Total semua dokumen relevan)
    """
    if not ground_truth_docs:
        return 0.0
        
    # Mencari irisan
    relevant_retrieved = set(retrieved_docs).intersection(set(ground_truth_docs))
    
    recall = len(relevant_retrieved) / len(ground_truth_docs)
    return recall


def calculate_f_measure(precision: float, recall: float) -> float:
    """
    Menghitung F-Measure (F1-Score).
    Ini adalah rata-rata harmonik dari Precision dan Recall.
    Rumus: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Menangani error pembagian dengan nol (Division by Zero)
    if (precision + recall) == 0:
        return 0.0
        
    f_measure = 2 * (precision * recall) / (precision + recall)
    return f_measure


def calculate_all_metrics(retrieved_docs: list[int], ground_truth_docs: list[int]) -> dict:
    """
    Fungsi pembantu (helper) untuk menghitung ketiga metrik sekaligus dalam format persentase (%).
    """
    p = calculate_precision(retrieved_docs, ground_truth_docs)
    r = calculate_recall(retrieved_docs, ground_truth_docs)
    f1 = calculate_f_measure(p, r)
    
    return {
        'precision': p * 100,
        'recall': r * 100,
        'f_measure': f1 * 100
    }


# ============================================================
# METRIK MODERN (Digunakan oleh Sistem Two-Stage Retrieval)
# ============================================================

def calculate_rr(retrieved_docs: list[int], ground_truth_docs: list[int]) -> float:
    """
    Menghitung Reciprocal Rank (RR) untuk satu kueri.
    RR = 1 / peringkat dokumen relevan pertama yang ditemukan.
    
    Contoh:
        retrieved = [5, 3, 8, 1]
        ground_truth = [8, 1]
        Dokumen relevan pertama (ID=8) ada di peringkat 3.
        RR = 1/3 = 0.333
    
    Args:
        retrieved_docs (list[int]): Daftar ID dokumen hasil retrieval (terurut peringkat).
        ground_truth_docs (list[int]): Daftar ID dokumen yang benar-benar relevan.
        
    Returns:
        float: Nilai RR (0.0 jika tidak ada dokumen relevan ditemukan).
    """
    gt_set = set(ground_truth_docs)
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0


def calculate_mrr(all_retrieved: list[list[int]], all_ground_truths: list[list[int]]) -> float:
    """
    Menghitung Mean Reciprocal Rank (MRR) dari beberapa kueri.
    MRR = rata-rata RR dari seluruh skenario kueri.
    
    Args:
        all_retrieved (list[list[int]]): Daftar hasil retrieval untuk setiap kueri.
        all_ground_truths (list[list[int]]): Daftar ground truth untuk setiap kueri.
        
    Returns:
        float: Nilai MRR (0.0 - 1.0).
    """
    if not all_retrieved:
        return 0.0
    
    total_rr = sum(
        calculate_rr(retrieved, gt)
        for retrieved, gt in zip(all_retrieved, all_ground_truths)
    )
    return total_rr / len(all_retrieved)


def calculate_dcg(retrieved_docs: list[int], ground_truth_docs: list[int], k: int = None) -> float:
    """
    Menghitung Discounted Cumulative Gain (DCG@k) untuk satu kueri.
    Rumus: DCG@k = Σ (rel_i / log2(i + 1)) untuk i = 1 sampai k.
    
    Relevansi bersifat biner (1 jika relevan, 0 jika tidak).
    
    Args:
        retrieved_docs (list[int]): Daftar ID dokumen hasil retrieval (terurut peringkat).
        ground_truth_docs (list[int]): Daftar ID dokumen yang benar-benar relevan.
        k (int, optional): Cutoff peringkat. Jika None, gunakan seluruh hasil.
        
    Returns:
        float: Nilai DCG.
    """
    gt_set = set(ground_truth_docs)
    if k is None:
        k = len(retrieved_docs)
    
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        rel = 1.0 if doc_id in gt_set else 0.0
        # i+2 karena indeks mulai dari 0, dan log2(1) = 0 (menghindari division by zero)
        dcg += rel / math.log2(i + 2)
    
    return dcg


def calculate_ndcg(retrieved_docs: list[int], ground_truth_docs: list[int], k: int = None) -> float:
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG@k) untuk satu kueri.
    Rumus: NDCG@k = DCG@k / IDCG@k
    
    IDCG adalah DCG ideal (semua dokumen relevan di peringkat paling atas).
    
    Contoh:
        retrieved = [5, 3, 8, 1], ground_truth = [8, 1]
        DCG  = 0/log2(2) + 0/log2(3) + 1/log2(4) + 1/log2(5) = 0 + 0 + 0.5 + 0.431 = 0.931
        IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
        NDCG = 0.931 / 1.631 = 0.571
    
    Args:
        retrieved_docs (list[int]): Daftar ID dokumen hasil retrieval (terurut peringkat).
        ground_truth_docs (list[int]): Daftar ID dokumen yang benar-benar relevan.
        k (int, optional): Cutoff peringkat. Jika None, gunakan seluruh hasil.
        
    Returns:
        float: Nilai NDCG (0.0 - 1.0).
    """
    if k is None:
        k = len(retrieved_docs)
    
    # DCG aktual dari hasil retrieval
    dcg = calculate_dcg(retrieved_docs, ground_truth_docs, k)
    
    # IDCG (Ideal DCG) — skenario sempurna: semua dokumen relevan di peringkat teratas
    n_relevant = min(len(ground_truth_docs), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_average_ndcg(
    all_retrieved: list[list[int]], 
    all_ground_truths: list[list[int]], 
    k: int = None
) -> float:
    """
    Menghitung rata-rata NDCG dari beberapa kueri.
    
    Args:
        all_retrieved (list[list[int]]): Daftar hasil retrieval untuk setiap kueri.
        all_ground_truths (list[list[int]]): Daftar ground truth untuk setiap kueri.
        k (int, optional): Cutoff peringkat.
        
    Returns:
        float: Nilai rata-rata NDCG (0.0 - 1.0).
    """
    if not all_retrieved:
        return 0.0
    
    total_ndcg = sum(
        calculate_ndcg(retrieved, gt, k)
        for retrieved, gt in zip(all_retrieved, all_ground_truths)
    )
    return total_ndcg / len(all_retrieved)


def calculate_modern_metrics(retrieved_docs: list[int], ground_truth_docs: list[int], k: int = None) -> dict:
    """
    Fungsi pembantu untuk menghitung metrik modern (RR dan NDCG) untuk satu kueri.
    
    Args:
        retrieved_docs (list[int]): Daftar ID dokumen hasil retrieval (terurut peringkat).
        ground_truth_docs (list[int]): Daftar ID dokumen yang benar-benar relevan.
        k (int, optional): Cutoff peringkat untuk NDCG.
        
    Returns:
        dict: Dictionary berisi nilai RR dan NDCG.
    """
    rr = calculate_rr(retrieved_docs, ground_truth_docs)
    ndcg = calculate_ndcg(retrieved_docs, ground_truth_docs, k)
    
    return {
        'rr': rr,
        'ndcg': ndcg,
    }

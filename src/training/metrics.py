"""
Evaluation Metrics for Neural RAMA (TensorFlow/Keras)

Provides recommendation quality metrics from TREC 2014:
  - Precision@K (P@5 target: 0.50)
  - Mean Reciprocal Rank (MRR target: 0.71)
  - Time-Biased Gain (TBG target: 0.70)

Phase 4 — Training Infrastructure

TensorFlow Version: 2.15+
"""

import numpy as np
from typing import List, Optional


def precision_at_k(
    predicted_scores: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 5,
    threshold: float = 0.0,
) -> float:
    """
    Precision@K: fraction of top-K predicted items that are relevant.

    Args:
        predicted_scores: (num_candidates,) — model scores for each candidate
        ground_truth:     (num_candidates,) — relevance labels (1=relevant, 0=not)
                          or continuous relevance; items with value > threshold
                          are treated as relevant
        k: Number of top items to evaluate
        threshold: Relevance threshold

    Returns:
        P@K as a float in [0, 1]
    """
    if len(predicted_scores) == 0:
        return 0.0

    k = min(k, len(predicted_scores))
    top_k_indices = np.argsort(predicted_scores)[::-1][:k]
    relevant = (ground_truth[top_k_indices] > threshold).astype(float)
    return float(np.mean(relevant))


def mean_reciprocal_rank(
    predicted_scores: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Mean Reciprocal Rank: 1 / rank of the first relevant item.

    Args:
        predicted_scores: (num_candidates,) — model scores
        ground_truth:     (num_candidates,) — relevance labels
        threshold: Relevance threshold

    Returns:
        MRR as a float in [0, 1]
    """
    if len(predicted_scores) == 0:
        return 0.0

    ranked_indices = np.argsort(predicted_scores)[::-1]
    for rank, idx in enumerate(ranked_indices, start=1):
        if ground_truth[idx] > threshold:
            return 1.0 / rank
    return 0.0


def time_biased_gain(
    predicted_scores: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.0,
    half_life: float = 224.0,
) -> float:
    """
    Time-Biased Gain (TBG): models the probability that a user reaches
    rank k, assuming exponential decay in user patience.

    TBG = sum_k gain(k) * decay(k)
    decay(k) = 2^(-T(k) / half_life)
    T(k) = sum of time spent on items up to rank k

    Simplified: assume unit time per item → T(k) = k

    Args:
        predicted_scores: (num_candidates,) — model scores
        ground_truth:     (num_candidates,) — relevance values
        threshold: Relevance threshold for binary gain
        half_life: Half-life parameter controlling patience decay (default 224s from TREC)

    Returns:
        TBG score as a float
    """
    if len(predicted_scores) == 0:
        return 0.0

    ranked_indices = np.argsort(predicted_scores)[::-1]
    tbg = 0.0

    for rank, idx in enumerate(ranked_indices):
        gain = 1.0 if ground_truth[idx] > threshold else 0.0
        # decay = 2^(-T(k) / half_life), T(k) = rank (0-indexed)
        decay = 2.0 ** (-rank / half_life)
        tbg += gain * decay

    return tbg


def evaluate_ranking(
    predicted_scores: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 5,
    threshold: float = 0.0,
) -> dict:
    """
    Compute all ranking metrics at once.

    Args:
        predicted_scores: (num_candidates,) — model scores
        ground_truth:     (num_candidates,) — relevance labels
        k: K for P@K
        threshold: Relevance threshold

    Returns:
        Dict with 'p_at_k', 'mrr', 'tbg'
    """
    return {
        "p_at_k": precision_at_k(predicted_scores, ground_truth, k, threshold),
        "mrr": mean_reciprocal_rank(predicted_scores, ground_truth, threshold),
        "tbg": time_biased_gain(predicted_scores, ground_truth, threshold),
    }


def evaluate_all_users(
    all_predicted: List[np.ndarray],
    all_ground_truth: List[np.ndarray],
    k: int = 5,
    threshold: float = 0.0,
) -> dict:
    """
    Compute mean metrics across all users.

    Args:
        all_predicted:    List of (num_candidates,) score arrays, one per user
        all_ground_truth: List of (num_candidates,) relevance arrays, one per user
        k: K for P@K
        threshold: Relevance threshold

    Returns:
        Dict with mean 'p_at_k', 'mrr', 'tbg'
    """
    metrics = [
        evaluate_ranking(pred, gt, k, threshold)
        for pred, gt in zip(all_predicted, all_ground_truth)
    ]
    return {
        "p_at_k": float(np.mean([m["p_at_k"] for m in metrics])),
        "mrr": float(np.mean([m["mrr"] for m in metrics])),
        "tbg": float(np.mean([m["tbg"] for m in metrics])),
    }

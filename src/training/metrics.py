"""
Evaluation Metrics for Neural RAMA (TensorFlow/Keras)

Provides recommendation quality metrics from TREC 2014:
  - Precision@K (P@5 target: 0.50)
  - Mean Reciprocal Rank (MRR target: 0.71)
  - Time-Biased Gain (TBG target: 0.70)

Note: time_biased_gain() returns a raw, unnormalized sum by default (this
is NOT on the paper's [0,1] scale). Pass normalize=True, or read
'tbg_normalized' from evaluate_ranking()/evaluate_all_users(), for a
bounded score comparable to the paper's 0.70 target. See
time_biased_gain()'s docstring for details on the normalization used.

Phase 4 — Training Infrastructure
Phase 5 — Added normalized TBG

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


def _raw_tbg(
    ranked_relevance: np.ndarray,
    half_life: float = 224.0,
) -> float:
    """
    Unnormalized Time-Biased Gain for a sequence of gains already sorted
    into rank order (rank 0 = top of the list).

    TBG = sum_k gain(k) * decay(k), decay(k) = 2^(-T(k) / half_life),
    with unit time per item so T(k) = k (0-indexed).
    """
    if len(ranked_relevance) == 0:
        return 0.0
    ranks = np.arange(len(ranked_relevance))
    decay = 2.0 ** (-ranks / half_life)
    return float(np.sum(ranked_relevance * decay))


def time_biased_gain(
    predicted_scores: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.0,
    half_life: float = 224.0,
    normalize: bool = False,
) -> float:
    """
    Time-Biased Gain (TBG): models the probability that a user reaches
    rank k, assuming exponential decay in user patience.

    TBG = sum_k gain(k) * decay(k)
    decay(k) = 2^(-T(k) / half_life)
    T(k) = sum of time spent on items up to rank k

    Simplified: assume unit time per item → T(k) = k

    Note on scale: the raw (unnormalized) sum above grows with the number
    of relevant candidates and is NOT on the same [0,1] scale as the TBG
    values reported in the RAMA/TREC 2014 paper (e.g. target ~0.70). To get
    a comparable, bounded score, set normalize=True, which divides the raw
    TBG by the TBG of the *ideal* ranking (ground truth sorted descending)
    — the same nDCG-style normalization used for normalized DCG (raw DCG /
    ideal DCG). This is our own interpretation for making the metric
    interpretable on a bounded scale; it is not a reproduction of the
    original TREC contextual-suggestion TBG formula, which was not
    available to verify directly.

    Args:
        predicted_scores: (num_candidates,) — model scores
        ground_truth:     (num_candidates,) — relevance values
        threshold: Relevance threshold for binary gain
        half_life: Half-life parameter controlling patience decay (default 224s from TREC)
        normalize: If True, return raw_tbg / ideal_tbg, bounded in [0, 1].
                   If False (default), return the raw unnormalized sum
                   (preserves prior behavior).

    Returns:
        TBG score as a float. Unbounded (raw) unless normalize=True.
    """
    if len(predicted_scores) == 0:
        return 0.0

    gains = (ground_truth > threshold).astype(float)

    ranked_indices = np.argsort(predicted_scores)[::-1]
    predicted_gains = gains[ranked_indices]
    raw_tbg = _raw_tbg(predicted_gains, half_life=half_life)

    if not normalize:
        return raw_tbg

    # Ideal ranking: sort ground truth gains descending (best possible order)
    ideal_gains = np.sort(gains)[::-1]
    ideal_tbg = _raw_tbg(ideal_gains, half_life=half_life)

    if ideal_tbg == 0.0:
        # No relevant items at all — nothing to normalize against.
        return 0.0

    return raw_tbg / ideal_tbg


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
        Dict with 'p_at_k', 'mrr', 'tbg' (raw, unnormalized — see
        time_biased_gain() docstring), and 'tbg_normalized' (bounded [0,1],
        nDCG-style normalization against the ideal ranking).
    """
    return {
        "p_at_k": precision_at_k(predicted_scores, ground_truth, k, threshold),
        "mrr": mean_reciprocal_rank(predicted_scores, ground_truth, threshold),
        "tbg": time_biased_gain(predicted_scores, ground_truth, threshold),
        "tbg_normalized": time_biased_gain(
            predicted_scores, ground_truth, threshold, normalize=True
        ),
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
        Dict with mean 'p_at_k', 'mrr', 'tbg', 'tbg_normalized'
    """
    metrics = [
        evaluate_ranking(pred, gt, k, threshold)
        for pred, gt in zip(all_predicted, all_ground_truth)
    ]
    return {
        "p_at_k": float(np.mean([m["p_at_k"] for m in metrics])),
        "mrr": float(np.mean([m["mrr"] for m in metrics])),
        "tbg": float(np.mean([m["tbg"] for m in metrics])),
        "tbg_normalized": float(np.mean([m["tbg_normalized"] for m in metrics])),
    }

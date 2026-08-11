"""
Phase 5 — End-to-end equivalence check: classical RAMA vs. trained NeuralRAMA.

Loads the same synthetic users/candidates, scores them with both the
classical RAMAPipeline and a trained NeuralRAMA checkpoint, and reports
rank correlation (Spearman's rho) between the two rankings per user.

This is NOT a test that neural must exactly reproduce classical output --
the two use different feature representations (discrete cosine similarity
vs. learned embeddings) by design. The goal is to confirm the neural model
has learned a *directionally consistent* notion of relevance: users whose
classical model prefers museums should also be scored by the neural model
as preferring candidates in categories the user rated positively.

Run:
    python src/training/equivalence_check.py --checkpoint checkpoints/<file>.weights.h5
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

from models.classical.pipeline import RAMAPipeline, load_synthetic_data
from models.neural.embeddings import TermEmbedding
from models.neural.neural_rama import NeuralRAMA
from training.data_pipeline import RAMADataPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Classical vs Neural RAMA equivalence check")
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a .weights.h5 checkpoint; if omitted, uses the latest in --checkpoint-dir")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--fusion-mode", type=str, default="fixed", choices=["fixed", "learned"])
    parser.add_argument("--fusion-scheme", type=str, default="RUN2", choices=["RUN1", "RUN2"])
    parser.add_argument("--num-users", type=int, default=15,
                         help="Number of users to compare (subset for speed)")
    return parser.parse_args()


def latest_checkpoint(checkpoint_dir: str) -> str:
    candidates = sorted(glob.glob(str(Path(checkpoint_dir) / "*.weights.h5")))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}. Run src/training/run_training.py first."
        )
    return candidates[-1]


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Load data (classical side): all profiles + first context's candidates
    # ------------------------------------------------------------------
    profiles, candidates, context = load_synthetic_data(args.data_dir)
    profiles = profiles[: args.num_users]
    print(f"Loaded {len(profiles)} user profiles, {len(candidates)} candidates "
          f"for context {context.city}, {context.state}")

    classical_pipeline = RAMAPipeline()

    # ------------------------------------------------------------------
    # Load data (neural side): reuse RAMADataPipeline for category vocab +
    # pre-computed term embeddings, matching what the model was trained on.
    # ------------------------------------------------------------------
    term_embedder = TermEmbedding(model_name="all-MiniLM-L6-v2", trainable=True)
    neural_pipeline = RAMADataPipeline(data_dir=args.data_dir)
    neural_pipeline.prepare(term_embedder)

    model = NeuralRAMA(
        num_categories=neural_pipeline.num_categories,
        embed_dim=args.embed_dim,
        term_dim=neural_pipeline.term_dim,
        num_heads=args.num_heads,
        fusion_mode=args.fusion_mode,
        fusion_scheme=args.fusion_scheme,
    )
    # Build the model (create weights) via one forward pass before loading.
    warm_user = neural_pipeline.users[0]
    max_h = neural_pipeline.max_history
    n = min(len(warm_user.rating_weights), max_h)
    pad_n = max_h - n
    cat_ids = np.array(warm_user.category_ids[:n], dtype=np.int32)
    if pad_n > 0:
        cat_ids = np.pad(cat_ids, ((0, pad_n), (0, 0)), constant_values=0)
    term_embs = warm_user.term_embeddings[:n]
    if pad_n > 0:
        term_embs = np.pad(term_embs, ((0, pad_n), (0, 0)), constant_values=0.0)
    ratings = np.array(warm_user.rating_weights[:n], dtype=np.float32)
    if pad_n > 0:
        ratings = np.pad(ratings, (0, pad_n), constant_values=0.0)

    warm_batch = {
        "user_category_ids": tf.expand_dims(tf.constant(cat_ids), 0),
        "user_term_embeddings": tf.expand_dims(tf.constant(term_embs, dtype=tf.float32), 0),
        "user_rating_weights": tf.expand_dims(tf.constant(ratings), 0),
        "candidate_category_ids": tf.constant(neural_pipeline.all_candidates[0].category_ids, dtype=tf.int32)[tf.newaxis, :],
        "candidate_term_embedding": tf.constant(neural_pipeline.all_candidates[0].term_embedding, dtype=tf.float32)[tf.newaxis, :],
        "distance": tf.constant([neural_pipeline.all_candidates[0].distance], dtype=tf.float32),
        "yelp_rating": tf.constant([neural_pipeline.all_candidates[0].yelp_rating], dtype=tf.float32),
        "review_count": tf.constant([neural_pipeline.all_candidates[0].review_count], dtype=tf.float32),
    }
    model(warm_batch, training=False)
    model.load_weights(checkpoint_path)
    print("Checkpoint loaded.")

    # Candidate features for the context used by the classical side
    ctx_id = context.context_id
    cand_features = neural_pipeline.get_candidate_features(ctx_id)
    cand_cat_emb, cand_term_emb = model.embed_candidate(
        tf.constant(cand_features["category_ids"]),
        tf.constant(cand_features["term_embeddings"]),
        training=False,
    )

    # ------------------------------------------------------------------
    # Score each user with both pipelines, compare rankings
    # ------------------------------------------------------------------
    correlations = []
    for profile, proc_user in zip(profiles, neural_pipeline.users[: len(profiles)]):
        # Classical scores
        user_model = classical_pipeline.build_user_model(profile)
        classical_scores = np.array([
            classical_pipeline.scorer.score_candidate(user_model, c)["final_score"]
            for c in candidates
        ])

        # Neural scores (same candidate order as `candidates`, matched via id)
        cand_id_to_idx = {
            c.candidate_id: i
            for i, c in enumerate(neural_pipeline.candidates_by_context[ctx_id])
        }
        order = [cand_id_to_idx[c.candidate_id] for c in candidates]

        max_h = neural_pipeline.max_history
        n = min(len(proc_user.rating_weights), max_h)
        pad_n = max_h - n
        cat_ids = np.array(proc_user.category_ids[:n], dtype=np.int32)
        if pad_n > 0:
            cat_ids = np.pad(cat_ids, ((0, pad_n), (0, 0)), constant_values=0)
        term_embs = proc_user.term_embeddings[:n]
        if pad_n > 0:
            term_embs = np.pad(term_embs, ((0, pad_n), (0, 0)), constant_values=0.0)
        ratings = np.array(proc_user.rating_weights[:n], dtype=np.float32)
        if pad_n > 0:
            ratings = np.pad(ratings, (0, pad_n), constant_values=0.0)

        user_general, user_specific = model.encode_user(
            tf.expand_dims(tf.constant(cat_ids), 0),
            tf.expand_dims(tf.constant(term_embs, dtype=tf.float32), 0),
            tf.expand_dims(tf.constant(ratings), 0),
            training=False,
        )
        num_cands = cand_cat_emb.shape[0]
        ug = tf.tile(user_general, [num_cands, 1])
        us = tf.tile(user_specific, [num_cands, 1])
        scores = model.score(
            ug, us, cand_cat_emb, cand_term_emb,
            tf.constant(cand_features["distances"]),
            tf.constant(cand_features["yelp_ratings"]),
            tf.constant(cand_features["review_counts"]),
            training=False,
        )
        neural_scores_full = scores["final"].numpy()
        neural_scores = neural_scores_full[order]

        rho, _ = spearmanr(classical_scores, neural_scores)
        correlations.append(rho)
        print(f"  {profile.user_id}: Spearman rho = {rho:+.3f}")

    correlations = np.array(correlations)
    print("\n" + "=" * 70)
    print("Classical vs Neural RAMA — Ranking Equivalence Summary")
    print("=" * 70)
    print(f"  Users compared:      {len(correlations)}")
    print(f"  Mean Spearman rho:   {np.nanmean(correlations):+.3f}")
    print(f"  Median Spearman rho: {np.nanmedian(correlations):+.3f}")
    print(f"  Users with rho > 0:  {int(np.sum(correlations > 0))}/{len(correlations)}")
    print("=" * 70)
    print(
        "\nNote: perfect agreement is not expected -- classical RAMA uses exact\n"
        "cosine similarity over discrete category/term weights, while the neural\n"
        "model uses learned embeddings trained via BPR ranking loss. A positive\n"
        "mean correlation with most users > 0 indicates the neural model learned\n"
        "a directionally consistent notion of user preference."
    )


if __name__ == "__main__":
    main()

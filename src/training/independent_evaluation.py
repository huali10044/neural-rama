"""
Phase 5 — Independent ground-truth evaluation for the trained NeuralRAMA model.

Motivation: the P@5/MRR/TBG numbers reported by run_training.py use ground
truth built from category overlap with the user's own rating history — the
exact same signal used to construct the BPR training triples in
RAMADataPipeline._generate_triples(). That's a check that the model learned
its training objective, not a check against an independent notion of
relevance.

This script instead uses the CLASSICAL RAMAPipeline's final_score (general
cosine similarity + specific cosine similarity + context score, Formula 2)
as the relevance ground truth. The classical scorer never participates in
neural training at all, so agreement here is a genuinely independent signal
that the neural model is ranking candidates the way RAMA's own algorithm
would consider relevant -- not just recovering its own BPR training labels.

Ground truth definition: for each user, the top `--relevant-fraction` of
candidates by classical final_score are labeled relevant (1), the rest not
relevant (0). Quantile-based rather than a fixed score threshold so it's
well-defined regardless of each user's score distribution.

Also reports a random-ranking baseline for the same ground truth, so the
model's scores can be read as "how much better than chance" rather than in
isolation.

Run:
    python src/training/independent_evaluation.py --checkpoint-dir checkpoints
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

from models.classical.pipeline import RAMAPipeline, load_synthetic_data
from models.neural.embeddings import TermEmbedding
from models.neural.neural_rama import NeuralRAMA
from training.data_pipeline import RAMADataPipeline
from training.metrics import evaluate_ranking


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NeuralRAMA against classical-RAMA-derived ground truth "
                     "(independent of the BPR training signal)"
    )
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a .weights.h5 checkpoint; if omitted, uses the latest in --checkpoint-dir")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--fusion-mode", type=str, default="fixed", choices=["fixed", "learned"])
    parser.add_argument("--fusion-scheme", type=str, default="RUN2", choices=["RUN1", "RUN2"])
    parser.add_argument("--num-users", type=int, default=None,
                         help="Number of users to evaluate; default = all")
    parser.add_argument("--relevant-fraction", type=float, default=0.1,
                         help="Top fraction of candidates (by classical final_score) "
                              "labeled relevant in the ground truth (default 0.1 = top 10%%)")
    parser.add_argument("--k", type=int, default=5, help="K for P@K")
    parser.add_argument("--random-trials", type=int, default=20,
                         help="Number of random-ranking trials per user for the chance baseline")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def latest_checkpoint(checkpoint_dir: str) -> str:
    candidates = sorted(glob.glob(str(Path(checkpoint_dir) / "*.weights.h5")))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}. Run src/training/run_training.py first."
        )
    return candidates[-1]


def build_warm_batch(neural_pipeline):
    """Construct a single-example batch to force NeuralRAMA to build its weights."""
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

    return {
        "user_category_ids": tf.expand_dims(tf.constant(cat_ids), 0),
        "user_term_embeddings": tf.expand_dims(tf.constant(term_embs, dtype=tf.float32), 0),
        "user_rating_weights": tf.expand_dims(tf.constant(ratings), 0),
        "candidate_category_ids": tf.constant(
            neural_pipeline.all_candidates[0].category_ids, dtype=tf.int32
        )[tf.newaxis, :],
        "candidate_term_embedding": tf.constant(
            neural_pipeline.all_candidates[0].term_embedding, dtype=tf.float32
        )[tf.newaxis, :],
        "distance": tf.constant([neural_pipeline.all_candidates[0].distance], dtype=tf.float32),
        "yelp_rating": tf.constant([neural_pipeline.all_candidates[0].yelp_rating], dtype=tf.float32),
        "review_count": tf.constant([neural_pipeline.all_candidates[0].review_count], dtype=tf.float32),
    }


def encode_user_tensors(proc_user, max_history):
    n = min(len(proc_user.rating_weights), max_history)
    pad_n = max_history - n
    cat_ids = np.array(proc_user.category_ids[:n], dtype=np.int32)
    if pad_n > 0:
        cat_ids = np.pad(cat_ids, ((0, pad_n), (0, 0)), constant_values=0)
    term_embs = proc_user.term_embeddings[:n]
    if pad_n > 0:
        term_embs = np.pad(term_embs, ((0, pad_n), (0, 0)), constant_values=0.0)
    ratings = np.array(proc_user.rating_weights[:n], dtype=np.float32)
    if pad_n > 0:
        ratings = np.pad(ratings, (0, pad_n), constant_values=0.0)
    return cat_ids, term_embs, ratings


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")
    rng = np.random.RandomState(args.seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    profiles, candidates, context = load_synthetic_data(args.data_dir)
    if args.num_users is not None:
        profiles = profiles[: args.num_users]
    print(f"Loaded {len(profiles)} user profiles, {len(candidates)} candidates "
          f"for context {context.city}, {context.state}")

    classical_pipeline = RAMAPipeline()

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
    model(build_warm_batch(neural_pipeline), training=False)
    model.load_weights(checkpoint_path)
    print("Checkpoint loaded.")

    ctx_id = context.context_id
    cand_features = neural_pipeline.get_candidate_features(ctx_id)
    cand_cat_emb, cand_term_emb = model.embed_candidate(
        tf.constant(cand_features["category_ids"]),
        tf.constant(cand_features["term_embeddings"]),
        training=False,
    )
    num_cands = cand_cat_emb.shape[0]

    # Map classical `candidates` list order -> neural pipeline's candidate order
    cand_id_to_idx = {
        c.candidate_id: i
        for i, c in enumerate(neural_pipeline.candidates_by_context[ctx_id])
    }
    order = [cand_id_to_idx[c.candidate_id] for c in candidates]

    # ------------------------------------------------------------------
    # Per-user: build ground truth from classical RAMA, score with neural
    # AND with random baselines, evaluate P@K / MRR / TBG (normalized).
    # ------------------------------------------------------------------
    all_p_at_k, all_mrr, all_tbg_norm = [], [], []
    rand_p_at_k, rand_mrr, rand_tbg_norm = [], [], []

    for profile, proc_user in zip(profiles, neural_pipeline.users[: len(profiles)]):
        # --- Ground truth: classical RAMA's own final_score ---
        user_model = classical_pipeline.build_user_model(profile)
        classical_scores = np.array([
            classical_pipeline.scorer.score_candidate(user_model, c)["final_score"]
            for c in candidates
        ])
        n_relevant = max(1, int(len(classical_scores) * args.relevant_fraction))
        relevant_indices = np.argsort(classical_scores)[::-1][:n_relevant]
        ground_truth = np.zeros(len(candidates), dtype=np.float32)
        ground_truth[relevant_indices] = 1.0

        # --- Predicted: neural model's scores, reordered to match `candidates` ---
        cat_ids, term_embs, ratings = encode_user_tensors(proc_user, neural_pipeline.max_history)
        user_general, user_specific = model.encode_user(
            tf.expand_dims(tf.constant(cat_ids), 0),
            tf.expand_dims(tf.constant(term_embs, dtype=tf.float32), 0),
            tf.expand_dims(tf.constant(ratings), 0),
            training=False,
        )
        ug = tf.tile(user_general, [num_cands, 1])
        us = tf.tile(user_specific, [num_cands, 1])
        scores = model.score(
            ug, us, cand_cat_emb, cand_term_emb,
            tf.constant(cand_features["distances"]),
            tf.constant(cand_features["yelp_ratings"]),
            tf.constant(cand_features["review_counts"]),
            training=False,
        )
        neural_scores = scores["final"].numpy()[order]

        metrics = evaluate_ranking(neural_scores, ground_truth, k=args.k)
        all_p_at_k.append(metrics["p_at_k"])
        all_mrr.append(metrics["mrr"])
        all_tbg_norm.append(metrics["tbg_normalized"])

        # --- Random-ranking baseline for the same ground truth ---
        for _ in range(args.random_trials):
            random_scores = rng.rand(len(candidates))
            rmetrics = evaluate_ranking(random_scores, ground_truth, k=args.k)
            rand_p_at_k.append(rmetrics["p_at_k"])
            rand_mrr.append(rmetrics["mrr"])
            rand_tbg_norm.append(rmetrics["tbg_normalized"])

    print("\n" + "=" * 70)
    print("Independent Evaluation — NeuralRAMA vs. Classical-RAMA-Derived Ground Truth")
    print("=" * 70)
    print(f"  Users evaluated:        {len(all_p_at_k)}")
    print(f"  Relevance definition:   top {args.relevant_fraction:.0%} of candidates "
          f"by classical RAMAPipeline final_score")
    print(f"  {'Metric':20s} {'NeuralRAMA':>12s} {'Random baseline':>18s}")
    print(f"  {'-'*20} {'-'*12} {'-'*18}")
    print(f"  {'P@' + str(args.k):20s} {np.mean(all_p_at_k):12.4f} {np.mean(rand_p_at_k):18.4f}")
    print(f"  {'MRR':20s} {np.mean(all_mrr):12.4f} {np.mean(rand_mrr):18.4f}")
    print(f"  {'TBG (normalized)':20s} {np.mean(all_tbg_norm):12.4f} {np.mean(rand_tbg_norm):18.4f}")
    print("=" * 70)
    print(
        "\nUnlike run_training.py's P@5/MRR/TBG (which use ground truth built from\n"
        "the same category-overlap signal as BPR training), this evaluation's\n"
        "ground truth comes from the classical RAMAPipeline's cosine-similarity\n"
        "scorer -- a component the neural model never sees during training. The\n"
        "random baseline column shows chance-level performance against the same\n"
        "ground truth, so NeuralRAMA's numbers should be read relative to it, not\n"
        "against the TREC 2014 targets (those apply to the self-referential\n"
        "evaluation in run_training.py, not this one)."
    )


if __name__ == "__main__":
    main()

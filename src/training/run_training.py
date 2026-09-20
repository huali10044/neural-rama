"""
Phase 5 — Training & Validation entry point for Neural RAMA.

Generates (or reuses) a synthetic TREC-format dataset, builds BPR training
triples, trains NeuralRAMA via RAMATrainer, and reports P@5 / MRR / TBG
against the TREC 2014 targets (P@5 >= 0.50, MRR >= 0.71, TBG >= 0.70).

Run:
    python src/training/run_training.py
    python src/training/run_training.py --num-users 60 --num-epochs 30
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf

from data.synthetic_generator import SyntheticDataGenerator
from models.neural.embeddings import TermEmbedding
from models.neural.neural_rama import NeuralRAMA
from training.data_pipeline import RAMADataPipeline
from training.trainer import RAMATrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Neural RAMA (Phase 5)")
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--num-users", type=int, default=60)
    parser.add_argument("--num-examples-per-user", type=int, default=80)
    parser.add_argument("--regenerate-data", action="store_true",
                         help="Force regeneration of synthetic data even if it exists")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--fusion-mode", type=str, default="fixed", choices=["fixed", "learned"])
    parser.add_argument("--fusion-scheme", type=str, default="RUN2", choices=["RUN1", "RUN2"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_synthetic_data(args) -> None:
    """Generate synthetic data if missing or --regenerate-data was passed."""
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / "metadata.json"

    if metadata_path.exists() and not args.regenerate_data:
        print(f"Using existing synthetic data at {data_dir}")
        return

    print(f"Generating synthetic data: {args.num_users} users, "
          f"{args.num_examples_per_user} examples/user...")
    generator = SyntheticDataGenerator(seed=args.seed)
    # generate_user_profile's num_examples default (80) is used inside
    # generate_dataset; monkey-patch not needed since generate_dataset only
    # takes num_users — regenerate manually to honor num_examples_per_user.
    import json as _json

    output_path = data_dir
    output_path.mkdir(parents=True, exist_ok=True)

    profiles = []
    for i in range(args.num_users):
        user_id = f"user_{i:04d}"
        profile = generator.generate_user_profile(user_id, num_examples=args.num_examples_per_user)
        profiles.append(profile)
        generator._save_profile(profile, output_path / f"profiles/{user_id}.json")

    contexts = generator.generate_contexts()
    for context in contexts:
        candidates = generator.generate_candidates(context)
        generator._save_candidates(candidates, output_path / f"candidates/{context.context_id}.json")

    from data.synthetic_generator import YELP_CATEGORIES
    metadata = {
        "num_users": args.num_users,
        "num_contexts": len(contexts),
        "categories": YELP_CATEGORIES,
    }
    with open(metadata_path, "w") as f:
        _json.dump(metadata, f, indent=2)

    print(f"Dataset generated at {output_path}")


def main():
    args = parse_args()

    print("=" * 70)
    print("Neural RAMA — Phase 5: Training & Validation")
    print("=" * 70)

    # 1. Data
    ensure_synthetic_data(args)

    print("\n[1/4] Loading and preparing data pipeline...")
    pipeline = RAMADataPipeline(
        data_dir=args.data_dir,
        val_fraction=0.2,
        seed=args.seed,
    )
    term_embedder = TermEmbedding(model_name="all-MiniLM-L6-v2", trainable=True)
    pipeline.prepare(term_embedder)
    print(f"   Users: {len(pipeline.users)} | "
          f"Categories: {pipeline.num_categories} | "
          f"Term dim: {pipeline.term_dim}")

    train_ds, val_ds = pipeline.create_datasets(
        batch_size=args.batch_size, num_negatives=1
    )
    print(f"   Train batches: {sum(1 for _ in train_ds)} | "
          f"Val batches: {sum(1 for _ in val_ds)}")

    # 2. Model
    print("\n[2/4] Building NeuralRAMA model...")
    model = NeuralRAMA(
        num_categories=pipeline.num_categories,
        embed_dim=args.embed_dim,
        term_dim=pipeline.term_dim,
        num_heads=args.num_heads,
        fusion_mode=args.fusion_mode,
        fusion_scheme=args.fusion_scheme,
    )

    # 3. Train
    print("\n[3/4] Training...")
    # Warm up the model with one forward pass so all sublayers (e.g. the
    # Sequential context encoder) build their weights before RAMATrainer
    # tries to count trainable_variables for the startup log line.
    for warm_batch in train_ds.take(1):
        model.compute_bpr_scores(warm_batch, training=False)

    trainer = RAMATrainer(
        model,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
    )
    trainer.train(train_ds, val_ds, num_epochs=args.num_epochs)

    # 4. Evaluate ranking quality (P@5, MRR, TBG) against TREC 2014 targets
    print("\n[4/4] Evaluating ranking quality (P@5, MRR, TBG)...")
    # Rebuild val users list from the same split used by create_datasets()
    import random as _random
    rng = _random.Random(args.seed)
    indices = list(range(len(pipeline.users)))
    rng.shuffle(indices)
    val_size = max(1, int(len(pipeline.users) * pipeline.val_fraction))
    val_indices = set(indices[:val_size])
    val_users = [u for i, u in enumerate(pipeline.users) if i in val_indices]

    ranking_metrics = trainer.evaluate_ranking(val_users, pipeline, k=5)

    print("\n" + "=" * 70)
    print("TREC 2014 Target Comparison (validation users)")
    print("=" * 70)
    targets = {
        "p_at_k": ("P@5", 0.50),
        "mrr": ("MRR", 0.71),
        "tbg_normalized": ("TBG (normalized)", 0.70),
    }
    for key, (label, target) in targets.items():
        value = ranking_metrics.get(key, 0.0)
        status = "PASS" if value >= target else "below target"
        print(f"  {label:18s}: {value:.4f}  (target >= {target:.2f})  [{status}]")
    print(f"  {'TBG (raw)':18s}: {ranking_metrics.get('tbg', 0.0):.4f}  "
          f"(unnormalized, not comparable to a fixed target — see metrics.py docstring)")
    print("=" * 70)

    return ranking_metrics


if __name__ == "__main__":
    main()

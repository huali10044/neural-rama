"""
Quick helper: load a checkpoint and print the learned ScoreFusion weights
(Wg, Ws, Wc). Used for the Phase 5 ablation study writeup.

Run:
    python src/training/inspect_fusion_weights.py --checkpoint-dir checkpoints_learned_run2init --fusion-scheme RUN2
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

from models.neural.embeddings import TermEmbedding
from models.neural.neural_rama import NeuralRAMA
from training.data_pipeline import RAMADataPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--fusion-scheme", type=str, default="RUN2")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = sorted(glob.glob(str(Path(args.checkpoint_dir) / "*.weights.h5")))[-1]

    term_embedder = TermEmbedding(model_name="all-MiniLM-L6-v2", trainable=True)
    pipeline = RAMADataPipeline(data_dir=args.data_dir)
    pipeline.prepare(term_embedder)

    model = NeuralRAMA(
        num_categories=pipeline.num_categories,
        embed_dim=args.embed_dim,
        term_dim=pipeline.term_dim,
        num_heads=args.num_heads,
        fusion_mode="learned",
        fusion_scheme=args.fusion_scheme,
    )

    train_ds, _ = pipeline.create_datasets(batch_size=32, num_negatives=1)
    for warm_batch in train_ds.take(1):
        model.compute_bpr_scores(warm_batch, training=False)

    model.load_weights(checkpoint)
    print(f"Checkpoint: {checkpoint}")
    print(f"Learned fusion weights: {model.score_fusion.get_weights_summary()}")


if __name__ == "__main__":
    main()

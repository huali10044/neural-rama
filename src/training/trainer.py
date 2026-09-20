"""
Training Loop for Neural RAMA (TensorFlow/Keras)

RAMATrainer manages:
  - BPR pairwise ranking training with tf.GradientTape
  - Validation with P@5, MRR, TBG metrics
  - Checkpoint management
  - Early stopping
  - Learning rate scheduling
  - Training history and logging

Phase 4 — Training Infrastructure

TensorFlow Version: 2.15+
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from training.losses import RAMALoss
from training.metrics import evaluate_ranking


class RAMATrainer:
    """
    Manages the full training loop for Neural RAMA.

    Usage:
        model = NeuralRAMA(...)
        trainer = RAMATrainer(model, learning_rate=1e-3)
        history = trainer.train(train_ds, val_ds, num_epochs=50)

    Args:
        model: NeuralRAMA model instance
        learning_rate: Initial learning rate
        loss_margin: BPR margin (0.0 = standard log-sigmoid)
        lambda_range: Weight range regularization strength
        lambda_balance: Component balance regularization strength
        checkpoint_dir: Directory for saving model checkpoints
        patience: Early stopping patience (epochs without improvement)
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        loss_margin: float = 0.0,
        lambda_range: float = 0.01,
        lambda_balance: float = 0.01,
        checkpoint_dir: str = "checkpoints",
        patience: int = 10,
    ):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = RAMALoss(
            margin=loss_margin,
            lambda_range=lambda_range,
            lambda_balance=lambda_balance,
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience = patience

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_ranking_loss": [],
            "val_loss": [],
            "val_ranking_loss": [],
            "val_p_at_5": [],
            "val_mrr": [],
            "learning_rate": [],
        }

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    @tf.function
    def _train_step(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        One gradient update step on a BPR batch.

        Args:
            batch: dict from the data pipeline

        Returns:
            Dict with loss components
        """
        with tf.GradientTape() as tape:
            pos_scores, neg_scores, user_general, user_specific = (
                self.model.compute_bpr_scores(batch, training=True)
            )

            # Get fusion weights if learned
            fusion_weights = None
            if self.model.score_fusion.mode == "learned":
                fusion_weights = tf.nn.softmax(self.model.score_fusion.raw_weights)

            losses = self.loss_fn(
                score_positive=pos_scores["final"],
                score_negative=neg_scores["final"],
                user_general=user_general,
                user_specific=user_specific,
                fusion_weights=fusion_weights,
            )

        gradients = tape.gradient(losses["total"], self.model.trainable_variables)

        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        return losses

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    @tf.function
    def _val_step(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Compute validation loss without gradient updates."""
        pos_scores, neg_scores, user_general, user_specific = (
            self.model.compute_bpr_scores(batch, training=False)
        )

        fusion_weights = None
        if self.model.score_fusion.mode == "learned":
            fusion_weights = tf.nn.softmax(self.model.score_fusion.raw_weights)

        losses = self.loss_fn(
            score_positive=pos_scores["final"],
            score_negative=neg_scores["final"],
            user_general=user_general,
            user_specific=user_specific,
            fusion_weights=fusion_weights,
        )

        # Also return the score differences for ranking metrics
        return {
            **losses,
            "score_diff": pos_scores["final"] - neg_scores["final"],
        }

    # ------------------------------------------------------------------
    # Epoch routines
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        dataset: tf.data.Dataset,
        training: bool = True,
    ) -> Dict[str, float]:
        """Run one epoch of training or validation."""
        total_loss = 0.0
        total_ranking = 0.0
        num_batches = 0
        all_score_diffs = []

        for batch in dataset:
            if training:
                losses = self._train_step(batch)
            else:
                losses = self._val_step(batch)
                all_score_diffs.append(losses["score_diff"].numpy())

            total_loss += float(losses["total"])
            total_ranking += float(losses["ranking"])
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_ranking = total_ranking / max(num_batches, 1)

        result = {"loss": avg_loss, "ranking_loss": avg_ranking}

        # For validation, compute pairwise accuracy (fraction of triples
        # where positive scored higher than negative)
        if not training and all_score_diffs:
            diffs = np.concatenate(all_score_diffs)
            pairwise_accuracy = float(np.mean(diffs > 0))
            result["pairwise_accuracy"] = pairwise_accuracy

        return result

    # ------------------------------------------------------------------
    # Ranking evaluation
    # ------------------------------------------------------------------

    def evaluate_ranking(
        self,
        users: list,
        pipeline,
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate ranking quality on full candidate lists.

        For each user × context combination, scores all candidates and
        computes P@K, MRR, TBG against the user's rating preferences.

        Args:
            users: List of ProcessedUser objects
            pipeline: RAMADataPipeline with candidates loaded
            k: K for P@K

        Returns:
            Dict with mean 'p_at_k', 'mrr', 'tbg'
        """
        all_p_at_k = []
        all_mrr = []
        all_tbg = []
        all_tbg_norm = []

        for user in users:
            n_items = len(user.rating_weights)
            if n_items == 0:
                continue

            # Build user tensors (batch_size=1)
            max_h = pipeline.max_history
            n = min(n_items, max_h)
            pad_n = max_h - n

            cat_ids = np.array(user.category_ids[:n], dtype=np.int32)
            if pad_n > 0:
                cat_ids = np.pad(cat_ids, ((0, pad_n), (0, 0)), constant_values=0)
            user_cat_ids = tf.expand_dims(tf.constant(cat_ids), 0)

            term_embs = user.term_embeddings[:n]
            if pad_n > 0:
                term_embs = np.pad(term_embs, ((0, pad_n), (0, 0)), constant_values=0.0)
            user_term_embs = tf.expand_dims(tf.constant(term_embs, dtype=tf.float32), 0)

            ratings = np.array(user.rating_weights[:n], dtype=np.float32)
            if pad_n > 0:
                ratings = np.pad(ratings, (0, pad_n), constant_values=0.0)
            user_ratings = tf.expand_dims(tf.constant(ratings), 0)

            # Encode user once
            user_general, user_specific = self.model.encode_user(
                user_cat_ids, user_term_embs, user_ratings, training=False
            )

            # Score all candidates in each context
            for ctx_id in pipeline.candidates_by_context:
                cand_features = pipeline.get_candidate_features(ctx_id)
                num_cands = len(cand_features["distances"])

                # Tile user vectors for batch scoring
                ug = tf.tile(user_general, [num_cands, 1])
                us = tf.tile(user_specific, [num_cands, 1])

                cand_cat_emb, cand_term_emb = self.model.embed_candidate(
                    tf.constant(cand_features["category_ids"]),
                    tf.constant(cand_features["term_embeddings"]),
                    training=False,
                )

                scores = self.model.score(
                    ug, us,
                    cand_cat_emb, cand_term_emb,
                    tf.constant(cand_features["distances"]),
                    tf.constant(cand_features["yelp_ratings"]),
                    tf.constant(cand_features["review_counts"]),
                    training=False,
                )

                pred_scores = scores["final"].numpy()

                # Ground truth: candidates with categories matching user's
                # liked categories get relevance=1, others=0
                liked_cats = set()
                for i, r in enumerate(user.rating_weights):
                    if r > 0:
                        for cid in user.category_ids[i]:
                            if cid >= 2:  # skip PAD and UNK
                                liked_cats.add(cid)

                gt = np.zeros(num_cands, dtype=np.float32)
                for j, cand_cats in enumerate(cand_features["category_ids"]):
                    if any(c in liked_cats for c in cand_cats if c >= 2):
                        gt[j] = 1.0

                metrics = evaluate_ranking(pred_scores, gt, k=k)
                all_p_at_k.append(metrics["p_at_k"])
                all_mrr.append(metrics["mrr"])
                all_tbg.append(metrics["tbg"])
                all_tbg_norm.append(metrics["tbg_normalized"])

        return {
            "p_at_k": float(np.mean(all_p_at_k)) if all_p_at_k else 0.0,
            "mrr": float(np.mean(all_mrr)) if all_mrr else 0.0,
            "tbg": float(np.mean(all_tbg)) if all_tbg else 0.0,
            "tbg_normalized": float(np.mean(all_tbg_norm)) if all_tbg_norm else 0.0,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model weights."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"neural_rama_epoch{epoch:03d}_loss{val_loss:.4f}.weights.h5"
        self.model.save_weights(str(path))

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from checkpoint."""
        self.model.load_weights(path)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        num_epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with validation, checkpointing, and early stopping.

        Args:
            train_ds: Training tf.data.Dataset (BPR triples)
            val_ds: Validation tf.data.Dataset
            num_epochs: Maximum number of epochs
            verbose: Print progress

        Returns:
            Training history dict
        """
        best_val_loss = float("inf")
        patience_counter = 0

        if verbose:
            total_params = sum(
                tf.size(v).numpy() for v in self.model.trainable_variables
            )
            print(f"Neural RAMA Training")
            print(f"  Trainable parameters: {total_params:,}")
            print(f"  Optimizer: Adam (lr={self.optimizer.learning_rate.numpy():.1e})")
            print(f"  Early stopping patience: {self.patience}")
            print("-" * 70)

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            # Training
            train_metrics = self._run_epoch(train_ds, training=True)

            # Validation
            val_metrics = self._run_epoch(val_ds, training=False)

            dt = time.time() - t0
            lr = float(self.optimizer.learning_rate)

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_ranking_loss"].append(train_metrics["ranking_loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_ranking_loss"].append(val_metrics["ranking_loss"])
            self.history["learning_rate"].append(lr)

            pair_acc = val_metrics.get("pairwise_accuracy", 0.0)

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"pair_acc={pair_acc:.3f} | "
                    f"lr={lr:.1e} | "
                    f"{dt:.1f}s"
                )

            # Checkpointing & early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self.save_checkpoint(epoch, best_val_loss)
                if verbose:
                    print(f"  ► New best val_loss={best_val_loss:.4f} — checkpoint saved")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch} "
                              f"(no improvement for {self.patience} epochs)")
                    break

        if verbose:
            print("-" * 70)
            print(f"Training complete. Best val_loss: {best_val_loss:.4f}")

        return self.history

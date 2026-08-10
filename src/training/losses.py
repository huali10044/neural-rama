"""
Training Loss Functions for Neural RAMA (TensorFlow/Keras)

Provides:
  - BPRLoss: Bayesian Personalized Ranking — pairwise ranking loss
  - RAMALoss: Multi-component loss combining ranking + regularization

Phase 4 — Training Infrastructure

TensorFlow Version: 2.15+
"""

import tensorflow as tf
from tensorflow import keras
from typing import Dict, Optional


class BPRLoss:
    """
    Bayesian Personalized Ranking (BPR) pairwise loss.

    For a (user, positive_item, negative_item) triple:
        loss = -mean(log(sigmoid(score_positive - score_negative)))

    Encourages the model to assign higher scores to items the user
    prefers over items the user does not prefer.

    Args:
        margin: Optional margin; if > 0, uses max(0, margin - diff) hinge
                variant instead of log-sigmoid. Default 0.0 = standard BPR.
    """

    def __init__(self, margin: float = 0.0):
        self.margin = margin

    def __call__(
        self,
        score_positive: tf.Tensor,
        score_negative: tf.Tensor,
    ) -> tf.Tensor:
        """
        Args:
            score_positive: (batch,) — scores for preferred items
            score_negative: (batch,) — scores for non-preferred items

        Returns:
            Scalar loss value
        """
        diff = score_positive - score_negative  # (batch,)

        if self.margin > 0:
            # Margin-based hinge variant
            loss = tf.nn.relu(self.margin - diff)
        else:
            # Standard BPR: -log(sigmoid(diff))
            loss = -tf.math.log_sigmoid(diff + 1e-8)

        return tf.reduce_mean(loss)


class RAMALoss:
    """
    Multi-component loss for Neural RAMA training.

    Combines:
    1. Ranking loss (BPR): ensures correct item ordering
    2. Weight range regularization: soft penalty for interest vectors
       straying far from the [-1, +1] range (supplements tanh constraint)
    3. Component balance regularization: penalizes degenerate fusion
       weights where all weight collapses to one component

    Total loss:
        L = L_ranking + lambda_range * L_range + lambda_balance * L_balance

    Args:
        margin: BPR margin (0.0 = standard log-sigmoid BPR)
        lambda_range: Weight for range regularization (default 0.01)
        lambda_balance: Weight for component balance regularization (default 0.01)
    """

    def __init__(
        self,
        margin: float = 0.0,
        lambda_range: float = 0.01,
        lambda_balance: float = 0.01,
    ):
        self.margin = margin
        self.lambda_range = lambda_range
        self.lambda_balance = lambda_balance
        self.bpr = BPRLoss(margin=margin)

    def __call__(
        self,
        score_positive: tf.Tensor,
        score_negative: tf.Tensor,
        user_general: Optional[tf.Tensor] = None,
        user_specific: Optional[tf.Tensor] = None,
        fusion_weights: Optional[tf.Tensor] = None,
    ) -> Dict[str, tf.Tensor]:
        """
        Args:
            score_positive: (batch,) — scores for preferred items
            score_negative: (batch,) — scores for non-preferred items
            user_general:   (batch, embed_dim) — general interest vectors (optional)
            user_specific:  (batch, embed_dim) — specific interest vectors (optional)
            fusion_weights: (3,) — current fusion weights [Wg, Ws, Wc] (optional)

        Returns:
            Dict with 'total', 'ranking', 'range_reg', 'balance_reg' losses
        """
        # 1. Ranking loss
        ranking_loss = self.bpr(score_positive, score_negative)

        # 2. Weight range regularization
        # Penalize interest vector elements whose magnitude exceeds 1.0.
        # Since we use tanh, values are already in [-1, +1], but this
        # adds a soft gradient signal to keep values well within range.
        range_reg = tf.constant(0.0)
        if user_general is not None:
            # Penalize large magnitudes (encourage values close to 0)
            range_reg = range_reg + tf.reduce_mean(user_general ** 2)
        if user_specific is not None:
            range_reg = range_reg + tf.reduce_mean(user_specific ** 2)

        # 3. Component balance regularization
        # Penalize when one fusion weight dominates (low entropy distribution).
        # Uses negative entropy: H = -sum(w * log(w)), we minimize -H.
        balance_reg = tf.constant(0.0)
        if fusion_weights is not None:
            # fusion_weights should already be softmax-normalized
            w = fusion_weights + 1e-8  # avoid log(0)
            entropy = -tf.reduce_sum(w * tf.math.log(w))
            # Max entropy for 3 weights = log(3) ≈ 1.099
            # Penalize low entropy (degenerate solutions)
            balance_reg = -entropy

        total = (
            ranking_loss
            + self.lambda_range * range_reg
            + self.lambda_balance * balance_reg
        )

        return {
            "total": total,
            "ranking": ranking_loss,
            "range_reg": range_reg,
            "balance_reg": balance_reg,
        }

"""
Neural Component Scorer and Score Fusion for RAMA Translation (TensorFlow/Keras)

Translates RAMA's three-component scoring (Formula 2) into neural equivalents:
  - ComponentScorer: computes general, specific, and context scores
  - ScoreFusion: fuses component scores into a final recommendation score

Formula 2 (paper):
    FinalScore = Wg * GeneralScore + Ws * SpecificScore + Wc * ContextScore

Weighting schemes from Table 7:
    RUN1 (specific priority): Wg=0.09, Ws=0.9,  Wc=0.01
    RUN2 (general priority):  Wg=0.9,  Ws=0.09, Wc=0.01

The neural version starts with these fixed weights as a baseline, and
optionally learns them end-to-end.

Phase 3 — builds on Phase 2 embedding layers (embeddings.py).

TensorFlow Version: 2.15+
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Optional, Tuple

from models.neural.embeddings import ContextEmbedding, WeightConstraint


# RAMA paper weighting schemes (Table 7)
WEIGHTING_SCHEMES: Dict[str, Dict[str, float]] = {
    "RUN1": {"general": 0.09, "specific": 0.90, "context": 0.01},
    "RUN2": {"general": 0.90, "specific": 0.09, "context": 0.01},
}


class ComponentScorer(layers.Layer):
    """
    Computes general, specific, and context scores for a candidate item.

    Neural equivalent of RAMAScorer in the classical implementation.

    Classical RAMA scores:
        GeneralScore  = cosine_similarity(user_category_weights, candidate_categories)
        SpecificScore = cosine_similarity(user_term_weights, candidate_terms)
        ContextScore  = weighted_sum(distance_score, rating_score, review_score)

    Neural version:
        GeneralScore  = dot(user_general_vec, candidate_category_emb)   ∈ [-1, +1]
        SpecificScore = dot(user_specific_vec, candidate_term_emb)       ∈ [-1, +1]
        ContextScore  = MLP(distance, rating, review_count) → scalar     ∈ [-1, +1]

    Cosine similarity over L2-normalized vectors is equivalent to dot product,
    so this exactly mirrors RAMA's scoring while using learned representations.

    Args:
        embed_dim: Dimensionality of user interest and candidate embeddings
        context_embed_dim: Context embedding dimension (from ContextEmbedding)
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 128,
        context_embed_dim: int = 64,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(ComponentScorer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.context_embed_dim = context_embed_dim
        self.dropout_rate = dropout_rate

        # Context encoder: distance/rating/review_count → embed → scalar score
        self.context_embedder = ContextEmbedding(
            embedding_dim=context_embed_dim,
            dropout_rate=dropout_rate,
            name="scorer_context_emb",
        )
        # Project context embedding → scalar score in [-1, +1]
        self.context_score_head = layers.Dense(
            1,
            activation="tanh",   # output directly in [-1, +1]
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name="context_score_head",
        )

        self.dropout = layers.Dropout(dropout_rate)
        self.weight_constraint = WeightConstraint(method="tanh", name="score_constraint")

    def score_general(
        self,
        user_general: tf.Tensor,
        candidate_category_emb: tf.Tensor,
    ) -> tf.Tensor:
        """
        General interest score via dot product (= cosine similarity of L2-normalized vecs).

        Args:
            user_general:          (batch, embed_dim) — user general interest vector
            candidate_category_emb: (batch, embed_dim) — mean-pooled candidate category emb

        Returns:
            score: (batch,) in [-1, +1]
        """
        # Both vectors should already be L2-normalized (from encoders)
        user_norm = tf.nn.l2_normalize(user_general, axis=-1)
        cand_norm = tf.nn.l2_normalize(candidate_category_emb, axis=-1)
        score = tf.reduce_sum(user_norm * cand_norm, axis=-1)  # (batch,)
        return score

    def score_specific(
        self,
        user_specific: tf.Tensor,
        candidate_term_emb: tf.Tensor,
    ) -> tf.Tensor:
        """
        Specific interest score via dot product.

        Args:
            user_specific:      (batch, embed_dim) — user specific interest vector
            candidate_term_emb: (batch, embed_dim) — candidate term embedding

        Returns:
            score: (batch,) in [-1, +1]
        """
        user_norm = tf.nn.l2_normalize(user_specific, axis=-1)
        cand_norm = tf.nn.l2_normalize(candidate_term_emb, axis=-1)
        score = tf.reduce_sum(user_norm * cand_norm, axis=-1)  # (batch,)
        return score

    def score_context(
        self,
        distance: tf.Tensor,
        rating: tf.Tensor,
        review_count: tf.Tensor,
        training: bool = None,
    ) -> tf.Tensor:
        """
        Context score from distance, Yelp rating, and review count.

        Args:
            distance:     (batch,) — distance in meters
            rating:       (batch,) — Yelp rating [1, 5]
            review_count: (batch,) — number of reviews
            training: bool

        Returns:
            score: (batch,) in [-1, +1]
        """
        context_emb = self.context_embedder(
            distance, rating, review_count, training=training
        )  # (batch, context_embed_dim)
        score = self.context_score_head(context_emb, training=training)  # (batch, 1)
        return tf.squeeze(score, axis=-1)  # (batch,)

    def call(
        self,
        user_general: tf.Tensor,
        user_specific: tf.Tensor,
        candidate_category_emb: tf.Tensor,
        candidate_term_emb: tf.Tensor,
        distance: tf.Tensor,
        rating: tf.Tensor,
        review_count: tf.Tensor,
        training: bool = None,
    ) -> Dict[str, tf.Tensor]:
        """
        Compute all three component scores.

        Args:
            user_general:           (batch, embed_dim)
            user_specific:          (batch, embed_dim)
            candidate_category_emb: (batch, embed_dim)
            candidate_term_emb:     (batch, embed_dim)
            distance:               (batch,)
            rating:                 (batch,)
            review_count:           (batch,)
            training: bool

        Returns:
            dict with keys 'general', 'specific', 'context' — each (batch,) in [-1, +1]
        """
        general_score = self.score_general(user_general, candidate_category_emb)
        specific_score = self.score_specific(user_specific, candidate_term_emb)
        context_score = self.score_context(distance, rating, review_count, training=training)

        return {
            "general": general_score,
            "specific": specific_score,
            "context": context_score,
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "context_embed_dim": self.context_embed_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class ScoreFusion(layers.Layer):
    """
    Fuses general, specific, and context scores into a final recommendation score.

    Neural equivalent of Formula 2 from the RAMA paper:
        FinalScore = Wg * GeneralScore + Ws * SpecificScore + Wc * ContextScore

    Two modes:
    - fixed: use paper weights (RUN1 or RUN2) — exact RAMA baseline
    - learned: weights are trainable scalars, learned end-to-end with ranking loss

    Start with fixed weights to validate that the neural model matches RAMA,
    then switch to learned weights to improve beyond RAMA.

    Args:
        mode: 'fixed' or 'learned'
        scheme: Weighting scheme for fixed mode — 'RUN1' or 'RUN2'
    """

    def __init__(
        self,
        mode: str = "fixed",
        scheme: str = "RUN2",
        **kwargs,
    ):
        super(ScoreFusion, self).__init__(**kwargs)
        assert mode in ("fixed", "learned"), f"mode must be 'fixed' or 'learned', got {mode}"
        assert scheme in WEIGHTING_SCHEMES, f"scheme must be one of {list(WEIGHTING_SCHEMES)}"
        self.mode = mode
        self.scheme = scheme

        if mode == "fixed":
            w = WEIGHTING_SCHEMES[scheme]
            self._wg = tf.constant(w["general"], dtype=tf.float32)
            self._ws = tf.constant(w["specific"], dtype=tf.float32)
            self._wc = tf.constant(w["context"], dtype=tf.float32)
        else:
            # Learned weights — initialized to RUN2 values, softmax-normalized
            # so they always sum to 1.0
            init = WEIGHTING_SCHEMES[scheme]
            self.raw_weights = self.add_weight(
                name="raw_fusion_weights",
                shape=(3,),
                initializer=tf.keras.initializers.Constant(
                    [init["general"], init["specific"], init["context"]]
                ),
                trainable=True,
            )

    def _get_weights(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return (Wg, Ws, Wc) normalized to sum to 1."""
        if self.mode == "fixed":
            return self._wg, self._ws, self._wc
        else:
            # Softmax ensures weights are positive and sum to 1
            w = tf.nn.softmax(self.raw_weights)
            return w[0], w[1], w[2]

    def call(
        self,
        scores: Dict[str, tf.Tensor],
        training: bool = None,
    ) -> tf.Tensor:
        """
        Args:
            scores: dict with keys 'general', 'specific', 'context' — each (batch,)
            training: bool (unused for fixed mode)

        Returns:
            final_score: (batch,) — weighted sum of component scores
        """
        wg, ws, wc = self._get_weights()
        final = wg * scores["general"] + ws * scores["specific"] + wc * scores["context"]
        return final  # (batch,)

    def get_weights_summary(self) -> Dict[str, float]:
        """Return current weight values (useful for logging)."""
        wg, ws, wc = self._get_weights()
        return {
            "Wg (general)": float(wg),
            "Ws (specific)": float(ws),
            "Wc (context)": float(wc),
        }

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode, "scheme": self.scheme})
        return config


class NeuralRAMAScorer(keras.Model):
    """
    Complete neural scoring pipeline for one user-candidate pair.

    Combines ComponentScorer and ScoreFusion into a single model,
    mirroring the classical RAMAPipeline.score_candidate() method.

    Args:
        embed_dim: User interest and candidate embedding dimension
        context_embed_dim: Context embedding dimension
        fusion_mode: 'fixed' or 'learned'
        fusion_scheme: 'RUN1' or 'RUN2' (used when fusion_mode='fixed')
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 128,
        context_embed_dim: int = 64,
        fusion_mode: str = "fixed",
        fusion_scheme: str = "RUN2",
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(NeuralRAMAScorer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.context_embed_dim = context_embed_dim

        self.component_scorer = ComponentScorer(
            embed_dim=embed_dim,
            context_embed_dim=context_embed_dim,
            dropout_rate=dropout_rate,
            name="component_scorer",
        )
        self.score_fusion = ScoreFusion(
            mode=fusion_mode,
            scheme=fusion_scheme,
            name="score_fusion",
        )

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = None,
    ) -> Dict[str, tf.Tensor]:
        """
        Args:
            inputs: dict with keys:
                - 'user_general':           (batch, embed_dim)
                - 'user_specific':          (batch, embed_dim)
                - 'candidate_category_emb': (batch, embed_dim)
                - 'candidate_term_emb':     (batch, embed_dim)
                - 'distance':               (batch,)
                - 'rating':                 (batch,)
                - 'review_count':           (batch,)
            training: bool

        Returns:
            dict with 'general', 'specific', 'context', 'final' scores — each (batch,)
        """
        scores = self.component_scorer(
            user_general=inputs["user_general"],
            user_specific=inputs["user_specific"],
            candidate_category_emb=inputs["candidate_category_emb"],
            candidate_term_emb=inputs["candidate_term_emb"],
            distance=inputs["distance"],
            rating=inputs["rating"],
            review_count=inputs["review_count"],
            training=training,
        )
        final = self.score_fusion(scores, training=training)
        return {**scores, "final": final}

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "context_embed_dim": self.context_embed_dim,
        })
        return config

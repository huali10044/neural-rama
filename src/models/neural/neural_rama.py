"""
End-to-End Neural RAMA Model (TensorFlow/Keras)

Wires together all Phase 2–3 components into a single trainable model:
    Embedding → Encoder → Scoring → Fusion → Final Score

Supports:
  - Full forward pass from raw IDs/embeddings to final scores
  - User encoding (category + term history → interest vectors)
  - Candidate scoring (interest vectors + candidate features → ranked scores)
  - BPR pairwise training interface

Phase 4 — Training Infrastructure

TensorFlow Version: 2.15+
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Optional, Tuple

from models.neural.encoders import GeneralInterestEncoder, SpecificInterestEncoder
from models.neural.scoring import ComponentScorer, ScoreFusion


class NeuralRAMA(keras.Model):
    """
    Complete Neural RAMA recommendation model.

    Architecture:
        User History ──► GeneralInterestEncoder  ──► user_general  (batch, D)
                    ──► SpecificInterestEncoder ──► user_specific (batch, D)

        Candidate ──► category_emb + term_emb + context features

        (user_general, user_specific, candidate_features) ──► ComponentScorer
            ──► general_score, specific_score, context_score
            ──► ScoreFusion ──► final_score

    Args:
        num_categories: Number of Yelp categories in vocabulary
        embed_dim: Interest vector and category embedding dimension
        term_dim: Pre-computed term embedding dimension (384 for MiniLM)
        context_embed_dim: Context embedding dimension
        num_heads: Attention heads in encoders
        fusion_mode: 'fixed' or 'learned'
        fusion_scheme: 'RUN1' or 'RUN2'
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        num_categories: int = 32,
        embed_dim: int = 128,
        term_dim: int = 384,
        context_embed_dim: int = 64,
        num_heads: int = 4,
        fusion_mode: str = "fixed",
        fusion_scheme: str = "RUN2",
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(NeuralRAMA, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.term_dim = term_dim
        self.num_categories = num_categories

        # User interest encoders
        self.general_encoder = GeneralInterestEncoder(
            num_categories=num_categories,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name="general_encoder",
        )
        self.specific_encoder = SpecificInterestEncoder(
            term_dim=term_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name="specific_encoder",
        )

        # Candidate category projection: pool multi-category embeddings
        # to a single embed_dim vector via the same category embedder
        # that the general encoder uses internally.
        self.candidate_cat_pool = layers.GlobalAveragePooling1D(
            name="candidate_cat_pool"
        )
        # Project candidate term embeddings to embed_dim (same projection
        # dimensions as the specific encoder)
        self.candidate_term_proj = layers.Dense(
            embed_dim,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name="candidate_term_proj",
        )
        self.candidate_term_norm = layers.LayerNormalization(epsilon=1e-6)

        # Scoring
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

    # ------------------------------------------------------------------
    # User encoding
    # ------------------------------------------------------------------

    def encode_user(
        self,
        user_category_ids: tf.Tensor,
        user_term_embeddings: tf.Tensor,
        user_rating_weights: tf.Tensor,
        training: bool = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Encode a user's rating history into general and specific interest vectors.

        Args:
            user_category_ids:    (batch, num_items, cats_per_item) int32
            user_term_embeddings: (batch, num_items, term_dim) float32
            user_rating_weights:  (batch, num_items) float32 in [-1, +1]
            training: bool

        Returns:
            (user_general, user_specific) — each (batch, embed_dim) in [-1, +1]
        """
        user_general = self.general_encoder(
            user_category_ids, user_rating_weights, training=training
        )
        user_specific = self.specific_encoder(
            user_term_embeddings, user_rating_weights, training=training
        )
        return user_general, user_specific

    # ------------------------------------------------------------------
    # Candidate embedding
    # ------------------------------------------------------------------

    def embed_candidate(
        self,
        candidate_category_ids: tf.Tensor,
        candidate_term_embedding: tf.Tensor,
        training: bool = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Embed a candidate item's category and term features.

        Args:
            candidate_category_ids:    (batch, cats_per_item) int32
            candidate_term_embedding:  (batch, term_dim) float32
            training: bool

        Returns:
            (cand_cat_emb, cand_term_emb) — each (batch, embed_dim)
        """
        # Category: use the general encoder's embedding layer, pool across categories
        cat_emb = self.general_encoder.category_embedding(
            candidate_category_ids, training=training
        )  # (batch, cats_per_item, embed_dim)
        cand_cat_emb = self.candidate_cat_pool(cat_emb)  # (batch, embed_dim)
        cand_cat_emb = tf.nn.l2_normalize(cand_cat_emb, axis=-1)

        # Term: project from term_dim → embed_dim
        cand_term_emb = self.candidate_term_proj(
            candidate_term_embedding, training=training
        )
        cand_term_emb = self.candidate_term_norm(cand_term_emb)
        cand_term_emb = tf.nn.l2_normalize(cand_term_emb, axis=-1)

        return cand_cat_emb, cand_term_emb

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        user_general: tf.Tensor,
        user_specific: tf.Tensor,
        cand_cat_emb: tf.Tensor,
        cand_term_emb: tf.Tensor,
        distance: tf.Tensor,
        yelp_rating: tf.Tensor,
        review_count: tf.Tensor,
        training: bool = None,
    ) -> Dict[str, tf.Tensor]:
        """
        Compute component and fused scores.

        Returns:
            Dict with 'general', 'specific', 'context', 'final' — each (batch,)
        """
        scores = self.component_scorer(
            user_general=user_general,
            user_specific=user_specific,
            candidate_category_emb=cand_cat_emb,
            candidate_term_emb=cand_term_emb,
            distance=distance,
            rating=yelp_rating,
            review_count=review_count,
            training=training,
        )
        final = self.score_fusion(scores, training=training)
        return {**scores, "final": final}

    # ------------------------------------------------------------------
    # Forward pass (full pipeline)
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = None,
    ) -> Dict[str, tf.Tensor]:
        """
        End-to-end forward pass.

        Args:
            inputs: dict with keys:
                User history:
                    'user_category_ids':    (batch, num_items, cats_per_item) int32
                    'user_term_embeddings': (batch, num_items, term_dim) float32
                    'user_rating_weights':  (batch, num_items) float32
                Candidate:
                    'candidate_category_ids':    (batch, cats_per_item) int32
                    'candidate_term_embedding':  (batch, term_dim) float32
                    'distance':                  (batch,) float32
                    'yelp_rating':               (batch,) float32
                    'review_count':              (batch,) float32
            training: bool

        Returns:
            Dict with 'general', 'specific', 'context', 'final',
            'user_general', 'user_specific' — each (batch, ...) tensors
        """
        # Encode user
        user_general, user_specific = self.encode_user(
            inputs["user_category_ids"],
            inputs["user_term_embeddings"],
            inputs["user_rating_weights"],
            training=training,
        )

        # Embed candidate
        cand_cat_emb, cand_term_emb = self.embed_candidate(
            inputs["candidate_category_ids"],
            inputs["candidate_term_embedding"],
            training=training,
        )

        # Score
        scores = self.score(
            user_general, user_specific,
            cand_cat_emb, cand_term_emb,
            inputs["distance"],
            inputs["yelp_rating"],
            inputs["review_count"],
            training=training,
        )

        # Include user vectors for regularization in loss
        scores["user_general"] = user_general
        scores["user_specific"] = user_specific

        return scores

    # ------------------------------------------------------------------
    # BPR training step
    # ------------------------------------------------------------------

    def compute_bpr_scores(
        self,
        batch: Dict[str, tf.Tensor],
        training: bool = True,
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Compute scores for both positive and negative candidates in a BPR batch.

        Args:
            batch: dict from the data pipeline with user + pos + neg features
            training: bool

        Returns:
            (pos_scores, neg_scores, user_general, user_specific)
        """
        # Encode user (shared between pos and neg)
        user_general, user_specific = self.encode_user(
            batch["user_category_ids"],
            batch["user_term_embeddings"],
            batch["user_rating_weights"],
            training=training,
        )

        # Positive candidate
        pos_cat_emb, pos_term_emb = self.embed_candidate(
            batch["pos_category_ids"],
            batch["pos_term_embedding"],
            training=training,
        )
        pos_scores = self.score(
            user_general, user_specific,
            pos_cat_emb, pos_term_emb,
            batch["pos_distance"],
            batch["pos_yelp_rating"],
            batch["pos_review_count"],
            training=training,
        )

        # Negative candidate
        neg_cat_emb, neg_term_emb = self.embed_candidate(
            batch["neg_category_ids"],
            batch["neg_term_embedding"],
            training=training,
        )
        neg_scores = self.score(
            user_general, user_specific,
            neg_cat_emb, neg_term_emb,
            batch["neg_distance"],
            batch["neg_yelp_rating"],
            batch["neg_review_count"],
            training=training,
        )

        return pos_scores, neg_scores, user_general, user_specific

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_categories": self.num_categories,
            "embed_dim": self.embed_dim,
            "term_dim": self.term_dim,
        })
        return config

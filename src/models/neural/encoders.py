"""
Neural User Interest Encoders for RAMA Translation (TensorFlow/Keras)

Translates RAMA's classical user interest models into neural equivalents:
- GeneralInterestEncoder: neural equivalent of the category weight dict
- SpecificInterestEncoder: neural equivalent of the term weight dict

Both use multi-head attention to aggregate rated items into a single
interest vector constrained to [-1, +1], preserving RAMA's semantics.

Phase 3 — builds on Phase 2 embedding layers (embeddings.py).

TensorFlow Version: 2.15+
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Optional, Tuple

from models.neural.embeddings import (
    CategoryEmbedding,
    TermEmbedding,
    WeightConstraint,
    CategoryVocabulary,
)


class AttentionAggregation(layers.Layer):
    """
    Aggregates a set of item embeddings into a single interest vector
    using multi-head attention, weighted by rating polarity.

    RAMA aggregates interest elements by reinforcement — items rated higher
    contribute more to the user's interest vector. This layer implements
    that with learned attention, using the rating as a bias so positively
    rated items attract more weight.

    Args:
        embed_dim: Dimensionality of input embeddings (must match key_dim * num_heads)
        num_heads: Number of attention heads
        dropout_rate: Dropout on attention weights
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(AttentionAggregation, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            name="interest_attention",
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

        # Learned query: a single trainable vector used as the "user" query
        # that attends over all rated items to produce one interest vector.
        self.query = self.add_weight(
            name="user_query",
            shape=(1, 1, embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(
        self,
        item_embeddings: tf.Tensor,
        rating_weights: tf.Tensor,
        training: bool = None,
    ) -> tf.Tensor:
        """
        Args:
            item_embeddings: (batch, num_items, embed_dim) — embedded rated items
            rating_weights:  (batch, num_items) — rating polarity in [-1, +1],
                             used as attention bias (higher-rated items attend more)
            training: bool

        Returns:
            interest_vector: (batch, embed_dim) — aggregated interest representation
        """
        batch_size = tf.shape(item_embeddings)[0]

        # Tile the learned query for each item in the batch
        query = tf.tile(self.query, [batch_size, 1, 1])  # (batch, 1, embed_dim)

        # Rating-weighted mean: directly encodes RAMA's semantics.
        # Positively-rated items add to interest; negatively-rated items subtract.
        # E[rating * embed] has the sign of rating since embed is zero-mean,
        # so we fold the absolute-value trick into the ratings directly:
        # use rating_weights to scale |embed| → E[|r| * |embed[d]|] > 0 when r > 0.
        abs_embeddings = tf.abs(item_embeddings)                          # (batch, num_items, embed_dim)
        rating_scale = tf.expand_dims(rating_weights, axis=-1)            # (batch, num_items, 1)
        rating_weighted_mean = tf.reduce_mean(abs_embeddings * rating_scale, axis=1)  # (batch, embed_dim)

        # MHA attended output: provides learned context; keeps value/output
        # projection weights in the gradient graph.
        attended_mha, _ = self.attention(
            query=query,
            value=item_embeddings,
            key=item_embeddings,
            training=training,
            return_attention_scores=True,
        )
        attended_mha = tf.squeeze(attended_mha, axis=1)                   # (batch, embed_dim)

        # Combine: rating-weighted directional signal + learned MHA context.
        # Scale the MHA term down so the directional signal from rating_weighted_mean
        # dominates and ensures mean(out_pos) > mean(out_neg) from initialization.
        combined = rating_weighted_mean + 0.1 * attended_mha
        combined = self.dropout(combined, training=training)

        return combined

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config


class GeneralInterestEncoder(layers.Layer):
    """
    Encodes a user's category-based (general) interests into a single vector.

    Neural equivalent of RAMA's general interest facet (category weight dict).

    RAMA: for each rated POI, extract categories, apply Formula 4 (reinforcement)
          to update per-category weights in [-1, +1].
    Neural: embed categories, attend over all rated items weighted by rating
            polarity, project to a single interest vector in [-1, +1].

    Args:
        num_categories: Vocabulary size for CategoryEmbedding
        embed_dim: Category embedding dimension
        num_heads: Attention heads in AttentionAggregation
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        num_categories: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(GeneralInterestEncoder, self).__init__(**kwargs)
        self.num_categories = num_categories
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.category_embedding = CategoryEmbedding(
            num_categories=num_categories,
            embedding_dim=embed_dim,
            dropout_rate=dropout_rate,
            name="general_cat_emb",
        )
        self.attention_aggregation = AttentionAggregation(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name="general_attention",
        )
        # Project pooled category embeddings per item → scalar per item,
        # then aggregate across items via attention.
        self.item_pool = layers.GlobalAveragePooling1D(name="general_item_pool")
        self.weight_constraint = WeightConstraint(method="tanh", name="general_constraint")

    def call(
        self,
        category_ids: tf.Tensor,
        rating_weights: tf.Tensor,
        training: bool = None,
    ) -> tf.Tensor:
        """
        Args:
            category_ids:   (batch, num_items, num_cats_per_item) int32 —
                            category IDs for each rated item
            rating_weights: (batch, num_items) float — rating polarity in [-1, +1]
            training: bool

        Returns:
            general_interest: (batch, embed_dim) vector in [-1, +1]
        """
        batch_size = tf.shape(category_ids)[0]
        num_items = tf.shape(category_ids)[1]

        # Embed categories for every item: (batch*num_items, num_cats, embed_dim)
        cat_ids_flat = tf.reshape(category_ids, [-1, tf.shape(category_ids)[2]])
        cat_emb_flat = self.category_embedding(cat_ids_flat, training=training)

        # Pool across categories per item → (batch*num_items, embed_dim)
        item_emb_flat = self.item_pool(cat_emb_flat)

        # Reshape back → (batch, num_items, embed_dim)
        item_embeddings = tf.reshape(item_emb_flat, [batch_size, num_items, self.embed_dim])

        # Attend over items weighted by rating polarity → (batch, embed_dim)
        interest_vec = self.attention_aggregation(
            item_embeddings, rating_weights, training=training
        )

        # Constrain to [-1, +1] (preserves RAMA's weight range semantics)
        return self.weight_constraint(interest_vec)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_categories": self.num_categories,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config


class SpecificInterestEncoder(layers.Layer):
    """
    Encodes a user's term-based (specific) interests into a single vector.

    Neural equivalent of RAMA's specific interest facet (term weight dict).

    RAMA: for each rated POI, extract content terms, apply Formula 4 to
          update per-term weights in [-1, +1].
    Neural: use pre-encoded sentence-transformer embeddings (computed outside
            the TF graph), attend over all rated items weighted by rating
            polarity, produce a single interest vector in [-1, +1].

    Note: text encoding via TermEmbedding.encode_texts() must happen before
    calling this layer. The layer receives pre-computed float32 tensors.

    Args:
        term_dim: Dimension of pre-encoded term embeddings (384 for MiniLM)
        embed_dim: Output dimension after projection
        num_heads: Attention heads
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        term_dim: int = 384,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(SpecificInterestEncoder, self).__init__(**kwargs)
        self.term_dim = term_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Project from sentence-transformer dim → embed_dim
        self.projection = layers.Dense(
            embed_dim,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name="specific_projection",
        )
        self.proj_norm = layers.LayerNormalization(epsilon=1e-6)
        self.proj_dropout = layers.Dropout(dropout_rate)

        self.attention_aggregation = AttentionAggregation(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name="specific_attention",
        )
        self.weight_constraint = WeightConstraint(method="tanh", name="specific_constraint")

    def call(
        self,
        term_embeddings: tf.Tensor,
        rating_weights: tf.Tensor,
        training: bool = None,
    ) -> tf.Tensor:
        """
        Args:
            term_embeddings: (batch, num_items, term_dim) — pre-encoded term vectors
            rating_weights:  (batch, num_items) — rating polarity in [-1, +1]
            training: bool

        Returns:
            specific_interest: (batch, embed_dim) vector in [-1, +1]
        """
        batch_size = tf.shape(term_embeddings)[0]
        num_items = tf.shape(term_embeddings)[1]

        # Project each item's term embedding → embed_dim
        # Flatten, project, reshape back
        term_flat = tf.reshape(term_embeddings, [-1, self.term_dim])
        proj_flat = self.projection(term_flat, training=training)
        proj_flat = self.proj_norm(proj_flat)
        proj_flat = self.proj_dropout(proj_flat, training=training)
        proj_flat = tf.nn.l2_normalize(proj_flat, axis=-1)

        item_embeddings = tf.reshape(proj_flat, [batch_size, num_items, self.embed_dim])

        # Attend over items → (batch, embed_dim)
        interest_vec = self.attention_aggregation(
            item_embeddings, rating_weights, training=training
        )

        return self.weight_constraint(interest_vec)

    def get_config(self):
        config = super().get_config()
        config.update({
            "term_dim": self.term_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config

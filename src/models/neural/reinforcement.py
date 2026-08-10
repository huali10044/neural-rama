"""
Neural Reinforcement Module for RAMA Translation (TensorFlow/Keras)

Translates RAMA's classical update rules into a differentiable neural layer:
  - Formula 3 (Aging):       aged = current * (1 - attenuation_factor)
  - Formula 4 (Reinforce):   new = aged + relevance * factor * (1-|aged|) * delta
  - Formula 5 (Insertion):   initial = relevance * factor * mention_freq

The neural version replaces the scalar delta with an attention-based delta
computed over the new event embedding, allowing the model to learn which
aspects of the event are most relevant to update.

Output is always constrained to [-1, +1] via tanh, preserving RAMA semantics.

Phase 3 — builds on Phase 2 embedding layers (embeddings.py).

TensorFlow Version: 2.15+
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional

from models.neural.embeddings import WeightConstraint


class NeuralReinforcement(layers.Layer):
    """
    Differentiable neural equivalent of RAMA's reinforcement update rules.

    Classical RAMA (Formulas 3 & 4):
        aged      = old_weight * (1 - attenuation_factor)
        new_weight = aged + event_relevance * reinforcement_factor
                          * mention_frequency * (1 - |aged|)

    Neural version:
        aged      = current_state * (1 - attenuation_factor)     # Formula 3
        delta     = attention(aged_state, new_event_embedding)    # learned delta
        magnitude = event_relevance * reinforcement_factor
                    * (1 - |aged|)                                # Formula 4 magnitude
        new_state = tanh(aged + magnitude * delta)               # constrained output

    The `attenuation_factor` and `reinforcement_factor` start at the paper's
    values (0.0 and 0.5) and can optionally be made trainable.

    Args:
        embed_dim: Dimensionality of the interest state vectors
        attenuation_factor: Initial aging rate (paper: 0.0 for TREC 2014)
        reinforcement_factor: Initial reinforcement strength (paper: 0.5)
        trainable_factors: If True, attenuation and reinforcement factors
                           become learned scalar parameters
        dropout_rate: Dropout on the delta projection
    """

    def __init__(
        self,
        embed_dim: int = 128,
        attenuation_factor: float = 0.0,
        reinforcement_factor: float = 0.5,
        trainable_factors: bool = False,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(NeuralReinforcement, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self._init_attenuation = attenuation_factor
        self._init_reinforcement = reinforcement_factor
        self.trainable_factors = trainable_factors
        self.dropout_rate = dropout_rate

        # Scalar factors — optionally trainable
        self.attenuation = self.add_weight(
            name="attenuation_factor",
            shape=(),
            initializer=tf.keras.initializers.Constant(attenuation_factor),
            trainable=trainable_factors,
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=1.0, rate=1.0
            ),
        )
        self.reinforcement = self.add_weight(
            name="reinforcement_factor",
            shape=(),
            initializer=tf.keras.initializers.Constant(reinforcement_factor),
            trainable=trainable_factors,
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=1.0, rate=1.0
            ),
        )

        # Cross-attention: current state attends over new event to produce delta.
        # Using a lightweight single-head attention here for interpretability.
        self.delta_attention = layers.MultiHeadAttention(
            num_heads=1,
            key_dim=embed_dim,
            dropout=dropout_rate,
            name="delta_attention",
        )
        self.delta_norm = layers.LayerNormalization(epsilon=1e-6)
        self.delta_dropout = layers.Dropout(dropout_rate)

        # Gating: learned gate controls how much of the delta to apply,
        # providing an additional learned modulation on top of the
        # magnitude term from Formula 4.
        self.gate = layers.Dense(
            embed_dim,
            activation="sigmoid",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            name="update_gate",
        )

        self.weight_constraint = WeightConstraint(method="tanh", name="reinforcement_constraint")

    def call(
        self,
        current_state: tf.Tensor,
        new_event_embedding: tf.Tensor,
        event_relevance: tf.Tensor,
        training: bool = None,
    ) -> tf.Tensor:
        """
        Apply one reinforcement update step.

        Args:
            current_state:       (batch, embed_dim) — current user interest vector
            new_event_embedding: (batch, embed_dim) — embedding of the new rated item
            event_relevance:     (batch,) — event relevance in [-1, +1]
                                 (maps from RAMA's rating scale: 0→-1, 4→+1)
            training: bool

        Returns:
            new_state: (batch, embed_dim) — updated interest vector in [-1, +1]
        """
        # --- Formula 3: Aging ---
        aged_state = current_state * (1.0 - self.attenuation)

        # --- Delta: element-wise feature activation of the new event ---
        # We use the absolute value of the normalized event embedding as the
        # update delta. This ensures delta >= 0 element-wise, so the sign of
        # the update comes purely from `relevance` (Formula 4 semantics):
        #   positive relevance → state increases toward item's feature magnitudes
        #   negative relevance → state decreases away from item's feature magnitudes
        event_norm = tf.nn.l2_normalize(new_event_embedding, axis=-1)   # (batch, embed_dim)
        delta = tf.abs(event_norm)                                        # (batch, embed_dim) >= 0
        delta = self.delta_dropout(delta, training=training)

        # --- Formula 4: Reinforcement magnitude ---
        # magnitude = event_relevance * reinforcement_factor * (1 - |aged_state|)
        relevance = tf.reshape(event_relevance, [-1, 1])   # (batch, 1)
        magnitude = relevance * self.reinforcement * (1.0 - tf.abs(aged_state))

        # --- Learned gate (additional modulation, always positive via sigmoid) ---
        gate_input = tf.concat([aged_state, new_event_embedding], axis=-1)
        gate_val = self.gate(gate_input, training=training)

        # --- Apply update ---
        new_state = aged_state + magnitude * delta * gate_val

        # --- Constrain to [-1, +1] (Formula 4 preserves RAMA weight range) ---
        return self.weight_constraint(new_state)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "attenuation_factor": self._init_attenuation,
            "reinforcement_factor": self._init_reinforcement,
            "trainable_factors": self.trainable_factors,
            "dropout_rate": self.dropout_rate,
        })
        return config


class UserStateUpdater(keras.Model):
    """
    Manages the full user interest state using NeuralReinforcement.

    Given a sequence of (item_embedding, event_relevance) pairs representing
    a user's rating history, applies NeuralReinforcement sequentially to
    build up the user's general or specific interest vector — mirroring
    how RAMA processes rating events one by one.

    Args:
        embed_dim: Interest vector dimensionality
        attenuation_factor: Aging rate (paper: 0.0)
        reinforcement_factor: Update strength (paper: 0.5)
        trainable_factors: Whether to learn the scalar factors
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 128,
        attenuation_factor: float = 0.0,
        reinforcement_factor: float = 0.5,
        trainable_factors: bool = False,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(UserStateUpdater, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        self.reinforcement = NeuralReinforcement(
            embed_dim=embed_dim,
            attenuation_factor=attenuation_factor,
            reinforcement_factor=reinforcement_factor,
            trainable_factors=trainable_factors,
            dropout_rate=dropout_rate,
            name="neural_reinforcement",
        )

    def call(
        self,
        item_embeddings: tf.Tensor,
        event_relevances: tf.Tensor,
        training: bool = None,
    ) -> tf.Tensor:
        """
        Process a sequence of rated items to build a user interest vector.

        Args:
            item_embeddings:  (batch, num_events, embed_dim) — ordered rating history
            event_relevances: (batch, num_events) — relevance scores in [-1, +1]
            training: bool

        Returns:
            final_state: (batch, embed_dim) — user interest vector in [-1, +1]
        """
        batch_size = tf.shape(item_embeddings)[0]
        num_events = tf.shape(item_embeddings)[1]

        # Initial state: zero (neutral, no interests yet)
        state = tf.zeros([batch_size, self.embed_dim], dtype=tf.float32)

        # Process events sequentially (mirrors RAMA's event loop)
        for t in tf.range(num_events):
            event_emb = item_embeddings[:, t, :]     # (batch, embed_dim)
            relevance = event_relevances[:, t]        # (batch,)
            state = self.reinforcement(
                current_state=state,
                new_event_embedding=event_emb,
                event_relevance=relevance,
                training=training,
            )

        return state

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

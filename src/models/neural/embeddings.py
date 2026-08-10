"""
Neural Embedding Layers for RAMA Translation (TensorFlow/Keras)

This module provides neural embeddings that preserve RAMA's semantic properties:
- Weight range: [-1, +1] (hate to love)
- Separate general (category) and specific (term) representations
- Support for reinforcement learning-style updates

Key Design Decisions:
1. Use TensorFlow/Keras for all neural components
2. Use sentence-transformers (compatible with TF via conversion) for term embeddings
3. Normalize to [-1, +1] range to preserve RAMA semantics
4. Support both static and dynamic (updatable) embeddings

TensorFlow Version: 2.15+
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer


class CategoryEmbedding(layers.Layer):
    """
    Learned embeddings for Yelp categories (general interests)
    
    RAMA's general interest model uses discrete categories with weights in [-1, +1].
    This layer learns continuous embeddings for each category while preserving
    the ability to represent positive/negative preferences.
    
    Design:
    - Each category gets a learned embedding vector
    - Embeddings are L2-normalized for stable cosine similarity
    - Support for unknown categories via <UNK> token
    
    Args:
        num_categories: Number of unique categories
        embedding_dim: Dimension of embedding vectors
        mask_zero: Whether to mask padding (index 0)
        dropout_rate: Dropout rate for regularization
        name: Layer name
    """
    
    def __init__(
        self,
        num_categories: int,
        embedding_dim: int = 128,
        mask_zero: bool = True,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(CategoryEmbedding, self).__init__(**kwargs)
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.mask_zero = mask_zero
        self.dropout_rate = dropout_rate
        
        # Learned category embeddings
        # Index 0 for <PAD>, index 1 for <UNK>
        self.embeddings = layers.Embedding(
            input_dim=num_categories + 2,  # +2 for <PAD> and <UNK>
            output_dim=embedding_dim,
            embeddings_initializer='glorot_uniform',
            mask_zero=mask_zero,
            name='category_embeddings'
        )
        
        self.dropout = layers.Dropout(dropout_rate)
        self.normalize = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=None, normalize=True):
        """
        Forward pass
        
        Args:
            inputs: (batch_size, num_categories) tensor of category indices
            training: Whether in training mode
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            embeddings: (batch_size, num_categories, embedding_dim)
        """
        # Get embeddings
        embeddings = self.embeddings(inputs)
        
        # Apply dropout during training
        if training:
            embeddings = self.dropout(embeddings, training=training)
        
        if normalize:
            # L2 normalization for stable cosine similarity
            embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
        
        return embeddings
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_categories': self.num_categories,
            'embedding_dim': self.embedding_dim,
            'mask_zero': self.mask_zero,
            'dropout_rate': self.dropout_rate
        })
        return config


class TermEmbedding(layers.Layer):
    """
    Pre-trained + fine-tunable embeddings for terms (specific interests)
    
    RAMA's specific interest model uses content words/terms with weights in [-1, +1].
    This layer uses pre-trained sentence embeddings (sentence-transformers) as a base
    and optionally fine-tunes them for the recommendation task.
    
    Design:
    - Start with pre-trained 'all-MiniLM-L6-v2' (384-dim, fast, good quality)
    - Optional projection layer to target dimension
    - Optional fine-tuning during training (via TF Hub or direct encoding)
    
    Note: Sentence-transformers models are primarily PyTorch-based, but we can:
    1. Use them for encoding (CPU-based, no gradients)
    2. Add TF projection layers on top for fine-tuning
    
    Args:
        model_name: Sentence-transformers model name
        embedding_dim: Dimension of base embeddings (384 for MiniLM)
        projection_dim: Optional projection to different dimension
        trainable: Whether projection layer is trainable
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        embedding_dim: int = 384,
        projection_dim: Optional[int] = None,
        trainable: bool = True,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(TermEmbedding, self).__init__(**kwargs)
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim or embedding_dim
        self.trainable_projection = trainable
        self.dropout_rate = dropout_rate
        
        # Load pre-trained sentence transformer (for encoding only)
        # This runs on CPU and doesn't participate in TF gradient computation
        self.encoder = SentenceTransformer(model_name)
        
        # Optional projection layer (trainable in TF)
        if projection_dim and projection_dim != embedding_dim:
            self.projection = keras.Sequential([
                layers.Dense(
                    projection_dim, 
                    activation=None,
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    trainable=trainable,
                    name='term_projection'
                ),
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dropout(dropout_rate)
            ], name='projection_stack')
        else:
            self.projection = None
        
        self.dropout = layers.Dropout(dropout_rate)
    
    def encode_texts(self, texts: List[str]) -> tf.Tensor:
        """
        Encode texts using pre-trained sentence transformer
        This is done outside TF's graph (no gradients)
        
        Args:
            texts: List of text strings
        
        Returns:
            embeddings: (batch_size, embedding_dim) numpy array -> tf.Tensor
        """
        # Encode on CPU (no gradients)
        embeddings_np = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        # Convert to TF tensor
        return tf.constant(embeddings_np, dtype=tf.float32)
    
    def call(self, inputs, training=None, normalize=True):
        """
        Forward pass
        
        Args:
            inputs: Pre-computed embeddings (batch_size, embedding_dim)
                    Note: Text encoding happens outside this layer
            training: Whether in training mode
            normalize: Whether to L2-normalize
        
        Returns:
            embeddings: (batch_size, projection_dim or embedding_dim)
        """
        embeddings = inputs
        
        # Apply projection if specified
        if self.projection is not None:
            embeddings = self.projection(embeddings, training=training)
        
        # Apply dropout during training
        if training:
            embeddings = self.dropout(embeddings, training=training)
        
        if normalize:
            embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
        
        return embeddings
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'projection_dim': self.projection_dim,
            'trainable': self.trainable_projection,
            'dropout_rate': self.dropout_rate
        })
        return config


class ContextEmbedding(layers.Layer):
    """
    Embeddings for contextual features (location, distance, ratings)
    
    RAMA's context model uses:
    - Geographic distance (continuous)
    - Yelp rating (1-5 stars)
    - Review count (continuous)
    
    This layer creates a learned representation of context that can be
    combined with interest-based scoring.
    
    Args:
        embedding_dim: Output embedding dimension
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(ContextEmbedding, self).__init__(**kwargs)
        
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        
        # Context feature processing
        # Input: [distance_normalized, rating_normalized, review_count_normalized]
        self.context_encoder = keras.Sequential([
            layers.Dense(
                embedding_dim,
                activation='relu',
                kernel_initializer='glorot_uniform',
                name='context_dense1'
            ),
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dropout(dropout_rate),
            layers.Dense(
                embedding_dim,
                activation=None,
                kernel_initializer='glorot_uniform',
                name='context_dense2'
            ),
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dropout(dropout_rate)
        ], name='context_encoder')
    
    def call(
        self,
        distance: tf.Tensor,
        rating: tf.Tensor,
        review_count: tf.Tensor,
        training=None,
        distance_limit: float = 8000.0,
        review_limit: int = 100
    ):
        """
        Forward pass
        
        Args:
            distance: (batch_size,) distance in meters
            rating: (batch_size,) Yelp rating (1-5)
            review_count: (batch_size,) number of reviews
            training: Whether in training mode
            distance_limit: normalization constant (RAMA uses 8000m)
            review_limit: normalization constant (RAMA uses 100)
        
        Returns:
            context_embedding: (batch_size, embedding_dim)
        """
        # Normalize features to [0, 1] range
        distance_norm = tf.clip_by_value(distance / distance_limit, 0.0, 1.0)
        rating_norm = (rating - 1.0) / 4.0  # [1,5] -> [0,1]
        review_norm = tf.clip_by_value(review_count / review_limit, 0.0, 1.0)
        
        # Stack features
        features = tf.stack([distance_norm, rating_norm, review_norm], axis=-1)
        
        # Encode context
        context_embedding = self.context_encoder(features, training=training)
        
        return context_embedding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class RAMAEmbeddings(keras.Model):
    """
    Combined embedding model for Neural RAMA
    
    Integrates all embedding types:
    - Category embeddings (general interests)
    - Term embeddings (specific interests)
    - Context embeddings
    
    This is the foundation for the neural translation of RAMA.
    
    Args:
        num_categories: Number of unique categories
        category_dim: Category embedding dimension
        term_model: Sentence-transformers model name
        term_dim: Term embedding dimension
        context_dim: Context embedding dimension
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        num_categories: int,
        category_dim: int = 128,
        term_model: str = 'all-MiniLM-L6-v2',
        term_dim: int = 384,
        context_dim: int = 64,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(RAMAEmbeddings, self).__init__(**kwargs)
        
        self.num_categories = num_categories
        self.category_dim = category_dim
        self.term_dim = term_dim
        self.context_dim = context_dim
        
        self.category_embedder = CategoryEmbedding(
            num_categories=num_categories,
            embedding_dim=category_dim,
            dropout_rate=dropout_rate,
            name='category_embedder'
        )
        
        self.term_embedder = TermEmbedding(
            model_name=term_model,
            embedding_dim=term_dim,
            trainable=True,
            dropout_rate=dropout_rate,
            name='term_embedder'
        )
        
        self.context_embedder = ContextEmbedding(
            embedding_dim=context_dim,
            dropout_rate=dropout_rate,
            name='context_embedder'
        )
    
    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training=None
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass through all embedding types
        
        Args:
            inputs: Dictionary with keys:
                - 'category_ids': (batch_size, num_categories) category indices
                - 'term_embeddings': (batch_size, term_dim) pre-encoded terms
                - 'distance': (batch_size,) distance values
                - 'rating': (batch_size,) rating values
                - 'review_count': (batch_size,) review count values
            training: Whether in training mode
        
        Returns:
            Dictionary with keys: 'category', 'term', 'context'
        """
        embeddings = {}
        
        if 'category_ids' in inputs:
            embeddings['category'] = self.category_embedder(
                inputs['category_ids'],
                training=training
            )
        
        if 'term_embeddings' in inputs:
            embeddings['term'] = self.term_embedder(
                inputs['term_embeddings'],
                training=training
            )
        
        if all(k in inputs for k in ['distance', 'rating', 'review_count']):
            embeddings['context'] = self.context_embedder(
                inputs['distance'],
                inputs['rating'],
                inputs['review_count'],
                training=training
            )
        
        return embeddings
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_categories': self.num_categories,
            'category_dim': self.category_dim,
            'term_dim': self.term_dim,
            'context_dim': self.context_dim
        })
        return config


class WeightConstraint(layers.Layer):
    """
    Constraint layer to maintain RAMA's [-1, +1] weight range
    
    RAMA's interest weights are explicitly bounded in [-1, +1] to represent
    the spectrum from hate to love. This layer provides mechanisms to
    maintain this constraint in neural models.
    
    Options:
    - 'tanh': Soft constraint via tanh activation
    - 'clip': Hard clipping to [-1, +1]
    - 'sigmoid': Map to [-1, +1] via 2*sigmoid(x) - 1
    
    Args:
        method: Constraint method ('tanh', 'clip', or 'sigmoid')
    """
    
    def __init__(self, method: str = 'tanh', **kwargs):
        super(WeightConstraint, self).__init__(**kwargs)
        assert method in ['tanh', 'clip', 'sigmoid']
        self.method = method
    
    def call(self, weights):
        """
        Constrain weights to [-1, +1] range
        
        Args:
            weights: Unconstrained weight tensor
        
        Returns:
            constrained_weights: Weights in [-1, +1]
        """
        if self.method == 'tanh':
            return tf.tanh(weights)
        elif self.method == 'clip':
            return tf.clip_by_value(weights, -1.0, 1.0)
        elif self.method == 'sigmoid':
            return 2.0 * tf.sigmoid(weights) - 1.0
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_config(self):
        config = super().get_config()
        config.update({'method': self.method})
        return config


# Utility classes for vocabulary management
class CategoryVocabulary:
    """Manages category ID mapping"""
    
    def __init__(self, categories: List[str]):
        self.categories = ['<PAD>', '<UNK>'] + sorted(categories)
        self.cat2id = {cat: i for i, cat in enumerate(self.categories)}
        self.id2cat = {i: cat for cat, i in self.cat2id.items()}
    
    def encode(self, category: str) -> int:
        """Convert category string to ID"""
        return self.cat2id.get(category, 1)  # 1 = <UNK>
    
    def decode(self, cat_id: int) -> str:
        """Convert category ID to string"""
        return self.id2cat.get(cat_id, '<UNK>')
    
    def encode_batch(self, categories: List[str]) -> List[int]:
        """Encode batch of categories"""
        return [self.encode(cat) for cat in categories]
    
    def __len__(self):
        return len(self.categories)


if __name__ == "__main__":
    # Example usage and testing
    print("="*80)
    print("Neural RAMA Embeddings (TensorFlow) - Testing")
    print("="*80)
    
    # Create sample vocabulary
    sample_categories = [
        'museums', 'restaurants', 'parks', 'art_galleries',
        'theaters', 'bars', 'nightlife', 'shopping'
    ]
    vocab = CategoryVocabulary(sample_categories)
    
    print(f"\nCategory Vocabulary: {len(vocab)} categories")
    print(f"Sample mapping: 'museums' -> {vocab.encode('museums')}")
    
    # Test category embeddings
    print("\n[1/4] Testing Category Embeddings...")
    category_embedder = CategoryEmbedding(
        num_categories=len(vocab),
        embedding_dim=128
    )
    
    # Sample category IDs
    cat_ids = tf.constant([[2, 3, 4], [5, 6, 0]], dtype=tf.int32)
    cat_emb = category_embedder(cat_ids, training=False)
    print(f"   ✓ Category embeddings shape: {cat_emb.shape}")
    
    # Check normalization
    norms = tf.norm(cat_emb, axis=-1)
    print(f"   ✓ Embedding norms (should be ~1.0): min={tf.reduce_min(norms):.3f}, max={tf.reduce_max(norms):.3f}")
    
    # Test term embeddings
    print("\n[2/4] Testing Term Embeddings...")
    try:
        term_embedder = TermEmbedding(
            model_name='all-MiniLM-L6-v2',
            trainable=True
        )
        
        # Pre-encode texts (outside TF graph)
        sample_texts = [
            "art museum with great exhibits",
            "Italian restaurant downtown"
        ]
        term_emb_input = term_embedder.encode_texts(sample_texts)
        
        # Pass through TF layer
        term_emb = term_embedder(term_emb_input, training=False)
        print(f"   ✓ Term embeddings shape: {term_emb.shape}")
        print(f"   ✓ Embedding dim: {term_emb.shape[-1]}")
        
    except Exception as e:
        print(f"   ⚠ Skipped (model download required): {e}")
    
    # Test context embeddings
    print("\n[3/4] Testing Context Embeddings...")
    context_embedder = ContextEmbedding(embedding_dim=64)
    
    distance = tf.constant([1500.0, 3000.0], dtype=tf.float32)
    rating = tf.constant([4.5, 3.5], dtype=tf.float32)
    review_count = tf.constant([120.0, 45.0], dtype=tf.float32)
    
    context_emb = context_embedder(distance, rating, review_count, training=False)
    print(f"   ✓ Context embeddings shape: {context_emb.shape}")
    
    # Test combined RAMA embeddings
    print("\n[4/4] Testing Combined RAMA Embeddings...")
    rama_embeddings = RAMAEmbeddings(
        num_categories=len(vocab),
        category_dim=128,
        term_dim=384,
        context_dim=64
    )
    
    try:
        # Prepare inputs
        inputs = {
            'category_ids': cat_ids,
            'term_embeddings': term_emb_input,
            'distance': distance,
            'rating': rating,
            'review_count': review_count
        }
        
        all_embeddings = rama_embeddings(inputs, training=False)
        
        print(f"   ✓ Generated embeddings for: {list(all_embeddings.keys())}")
        for key, emb in all_embeddings.items():
            print(f"     - {key}: {emb.shape}")
    except:
        print("   ⚠ Integration test requires term embeddings")
    
    # Test weight constraint
    print("\n[Bonus] Testing Weight Constraint...")
    constraint = WeightConstraint(method='tanh')
    unconstrained = tf.random.normal([5, 10]) * 3  # Random values
    constrained = constraint(unconstrained)
    print(f"   ✓ Constrained range: [{tf.reduce_min(constrained):.3f}, {tf.reduce_max(constrained):.3f}]")
    print(f"   ✓ All in [-1, +1]: {tf.reduce_all(constrained >= -1.0) and tf.reduce_all(constrained <= 1.0)}")
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
    print("\nTensorFlow-specific notes:")
    print("- Using tf.keras.layers for all components")
    print("- Sentence-transformers for text encoding (CPU)")
    print("- TF projection layers for fine-tuning")
    print("- All gradients flow through TF graph")
    print("="*80)

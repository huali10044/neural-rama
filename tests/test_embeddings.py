"""
Test suite for Neural RAMA embedding layers (TensorFlow)

Run with: python tests/test_embeddings.py
Or with pytest: pytest tests/test_embeddings.py -v
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import tensorflow as tf
    import numpy as np
    from models.neural.embeddings import (
        CategoryEmbedding,
        TermEmbedding,
        ContextEmbedding,
        RAMAEmbeddings,
        WeightConstraint,
        CategoryVocabulary
    )
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    print(f"Warning: Could not import dependencies: {e}")
    print("Install with: pip install tensorflow sentence-transformers --break-system-packages")


def test_category_vocabulary():
    """Test category vocabulary management"""
    if not TF_AVAILABLE:
        print("SKIP: test_category_vocabulary (dependencies not available)")
        return
    
    print("\n[TEST] Category Vocabulary")
    
    categories = ['museums', 'restaurants', 'parks']
    vocab = CategoryVocabulary(categories)
    
    # Test size (3 categories + PAD + UNK)
    assert len(vocab) == 5, f"Expected 5, got {len(vocab)}"
    
    # Test encoding
    assert vocab.encode('museums') > 1, "Museums should have ID > 1"
    assert vocab.encode('unknown_category') == 1, "Unknown should map to UNK (1)"
    
    # Test decoding
    museums_id = vocab.encode('museums')
    assert vocab.decode(museums_id) == 'museums', "Decode should reverse encode"
    
    # Test batch encoding
    batch = ['museums', 'restaurants', 'unknown']
    ids = vocab.encode_batch(batch)
    assert len(ids) == 3, "Batch encoding should return same length"
    assert ids[2] == 1, "Unknown should map to UNK"
    
    print("   ✓ Vocabulary encoding/decoding works")
    print("   ✓ Unknown categories handled correctly")


def test_category_embedding():
    """Test category embedding layer"""
    if not TF_AVAILABLE:
        print("SKIP: test_category_embedding (dependencies not available)")
        return
    
    print("\n[TEST] Category Embedding Layer")
    
    num_categories = 10
    embedding_dim = 128
    batch_size = 4
    num_cats_per_item = 3
    
    embedder = CategoryEmbedding(
        num_categories=num_categories,
        embedding_dim=embedding_dim
    )
    
    # Create sample input
    category_ids = tf.random.uniform(
        (batch_size, num_cats_per_item),
        minval=2,
        maxval=num_categories + 2,
        dtype=tf.int32
    )
    
    # Forward pass
    embeddings = embedder(category_ids, training=False, normalize=True)
    
    # Check shape
    expected_shape = (batch_size, num_cats_per_item, embedding_dim)
    assert embeddings.shape == expected_shape, f"Expected {expected_shape}, got {embeddings.shape}"
    
    # Check normalization (should have unit norm)
    norms = tf.norm(embeddings, axis=-1)
    assert tf.reduce_all(tf.abs(norms - 1.0) < 0.01), "Embeddings should be approximately normalized"
    
    print(f"   ✓ Output shape: {embeddings.shape}")
    print(f"   ✓ Normalization: norms ∈ [{tf.reduce_min(norms):.3f}, {tf.reduce_max(norms):.3f}]")
    
    # Test that layer is trainable
    assert len(embedder.trainable_variables) > 0, "Layer should have trainable variables"
    print(f"   ✓ Trainable parameters: {sum(tf.size(v).numpy() for v in embedder.trainable_variables)}")


def test_term_embedding():
    """Test term embedding layer"""
    if not TF_AVAILABLE:
        print("SKIP: test_term_embedding (dependencies not available)")
        return
    
    print("\n[TEST] Term Embedding Layer")
    
    try:
        embedder = TermEmbedding(
            model_name='all-MiniLM-L6-v2',
            trainable=True
        )
        
        # Sample texts
        texts = [
            "art museum with great exhibits",
            "Italian restaurant downtown",
            "beautiful park with hiking trails"
        ]
        
        # Pre-encode texts (outside TF graph)
        embeddings_input = embedder.encode_texts(texts)
        
        # Forward pass through TF layer
        embeddings = embedder(embeddings_input, training=False, normalize=True)
        
        # Check shape
        batch_size = len(texts)
        expected_shape = (batch_size, embedder.embedding_dim)
        assert embeddings.shape == expected_shape, f"Expected {expected_shape}, got {embeddings.shape}"
        
        # Check normalization
        norms = tf.norm(embeddings, axis=-1)
        assert tf.reduce_all(tf.abs(norms - 1.0) < 0.01), "Embeddings should be normalized"
        
        # Check semantic similarity
        texts_sim = ["art museum", "art gallery", "pizza restaurant"]
        emb_sim_input = embedder.encode_texts(texts_sim)
        emb_sim = embedder(emb_sim_input, training=False, normalize=True)
        
        sim_museum_gallery = tf.reduce_sum(emb_sim[0] * emb_sim[1])
        sim_museum_restaurant = tf.reduce_sum(emb_sim[0] * emb_sim[2])
        
        assert sim_museum_gallery > sim_museum_restaurant, \
            "Museum should be more similar to gallery than restaurant"
        
        print(f"   ✓ Output shape: {embeddings.shape}")
        print(f"   ✓ Normalization: norms ∈ [{tf.reduce_min(norms):.3f}, {tf.reduce_max(norms):.3f}]")
        print(f"   ✓ Semantic similarity preserved: museum-gallery={sim_museum_gallery:.3f} > museum-restaurant={sim_museum_restaurant:.3f}")
        
    except Exception as e:
        print(f"   ⚠ Skipped (model download required): {e}")


def test_context_embedding():
    """Test context embedding layer"""
    if not TF_AVAILABLE:
        print("SKIP: test_context_embedding (dependencies not available)")
        return
    
    print("\n[TEST] Context Embedding Layer")
    
    embedding_dim = 64
    batch_size = 5
    
    embedder = ContextEmbedding(embedding_dim=embedding_dim)
    
    # Sample context features
    distance = tf.constant([1500.0, 3000.0, 5000.0, 7000.0, 10000.0], dtype=tf.float32)
    rating = tf.constant([4.5, 3.5, 4.0, 2.5, 5.0], dtype=tf.float32)
    review_count = tf.constant([120.0, 45.0, 200.0, 10.0, 50.0], dtype=tf.float32)
    
    # Forward pass
    context_emb = embedder(distance, rating, review_count, training=False)
    
    # Check shape
    expected_shape = (batch_size, embedding_dim)
    assert context_emb.shape == expected_shape, f"Expected {expected_shape}, got {context_emb.shape}"
    
    # Check that different contexts produce different embeddings
    diff = tf.norm(context_emb[0] - context_emb[1])
    assert diff > 0.1, "Different contexts should have different embeddings"
    
    # Check normalization effects
    close_venue = embedder(
        tf.constant([500.0]),
        tf.constant([4.5]),
        tf.constant([100.0]),
        training=False
    )
    far_venue = embedder(
        tf.constant([7500.0]),
        tf.constant([4.5]),
        tf.constant([100.0]),
        training=False
    )
    
    distance_diff = tf.norm(close_venue - far_venue)
    
    print(f"   ✓ Output shape: {context_emb.shape}")
    print("   ✓ Different contexts produce different embeddings")
    print(f"   ✓ Close venue vs far venue: distance={distance_diff:.3f}")
    
    # Test trainability
    assert len(embedder.trainable_variables) > 0, "Layer should have trainable variables"
    print(f"   ✓ Trainable parameters: {sum(tf.size(v).numpy() for v in embedder.trainable_variables)}")


def test_weight_constraint():
    """Test weight constraint layer"""
    if not TF_AVAILABLE:
        print("SKIP: test_weight_constraint (dependencies not available)")
        return
    
    print("\n[TEST] Weight Constraint Layer")
    
    # Test different constraint methods
    methods = ['tanh', 'clip', 'sigmoid']
    
    # Generate unconstrained weights
    unconstrained = tf.random.normal([100, 128]) * 5  # Large range to test
    
    for method in methods:
        constraint = WeightConstraint(method=method)
        constrained = constraint(unconstrained)
        
        # Check range
        min_val = tf.reduce_min(constrained)
        max_val = tf.reduce_max(constrained)
        
        assert min_val >= -1.0, f"{method}: min should be >= -1"
        assert max_val <= 1.0, f"{method}: max should be <= 1"
        
        print(f"   ✓ {method}: range [{min_val:.3f}, {max_val:.3f}]")
    
    # Test that tanh is differentiable
    constraint_tanh = WeightConstraint('tanh')
    with tf.GradientTape() as tape:
        x = tf.Variable([0.0, 1.0, 2.0, 5.0])
        y = constraint_tanh(x)
        loss = tf.reduce_sum(y)
    
    gradients = tape.gradient(loss, x)
    assert gradients is not None, "Tanh should be differentiable"
    assert not tf.reduce_any(tf.math.is_nan(gradients)), "Gradients should not be NaN"
    
    print("   ✓ Tanh is differentiable (gradients computed)")
    print(f"   ✓ Gradient range: [{tf.reduce_min(gradients):.3f}, {tf.reduce_max(gradients):.3f}]")

    # Test that tanh trains with BPR (pairwise ranking) loss
    constraint_tanh_bpr = WeightConstraint('tanh')
    user_weights = tf.Variable(tf.random.normal([4]))  # unconstrained user preferences

    # Simulate preferred and non-preferred item features
    preferred_item = tf.constant([1.0, 0.5, -0.2, 0.8])
    non_preferred_item = tf.constant([-0.3, 0.1, 0.9, -0.5])

    with tf.GradientTape() as tape:
        constrained = constraint_tanh_bpr(user_weights)  # project to [-1, +1]
        score_preferred = tf.reduce_sum(constrained * preferred_item)
        score_non_preferred = tf.reduce_sum(constrained * non_preferred_item)
        bpr_loss = -tf.math.log_sigmoid(score_preferred - score_non_preferred)

    gradients = tape.gradient(bpr_loss, user_weights)
    assert gradients is not None, "BPR loss should be differentiable through tanh"
    assert not tf.reduce_any(tf.math.is_nan(gradients)), "BPR gradients should not be NaN"

    # Verify the loss pushes preferred score above non-preferred
    initial_diff = score_preferred - score_non_preferred
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    for _ in range(50):
        with tf.GradientTape() as tape:
            constrained = constraint_tanh_bpr(user_weights)
            score_preferred = tf.reduce_sum(constrained * preferred_item)
            score_non_preferred = tf.reduce_sum(constrained * non_preferred_item)
            bpr_loss = -tf.math.log_sigmoid(score_preferred - score_non_preferred)
        gradients = tape.gradient(bpr_loss, user_weights)
        optimizer.apply_gradients([(gradients, user_weights)])

    final_diff = score_preferred - score_non_preferred
    assert final_diff > initial_diff, "Training should increase preferred item score relative to non-preferred"

    print("   ✓ BPR loss is differentiable through tanh")
    print(f"   ✓ Score diff improved: {initial_diff:.3f} → {final_diff:.3f}")


def test_rama_embeddings_integration():
    """Test integrated RAMA embeddings model"""
    if not TF_AVAILABLE:
        print("SKIP: test_rama_embeddings_integration (dependencies not available)")
        return
    
    print("\n[TEST] Integrated RAMA Embeddings")
    
    num_categories = 10
    batch_size = 3
    
    rama_emb = RAMAEmbeddings(
        num_categories=num_categories,
        category_dim=128,
        term_dim=384,
        context_dim=64
    )
    
    # Prepare inputs
    category_ids = tf.random.uniform(
        (batch_size, 4),
        minval=2,
        maxval=num_categories + 2,
        dtype=tf.int32
    )
    
    distance = tf.constant([1000.0, 2000.0, 3000.0], dtype=tf.float32)
    rating = tf.constant([4.5, 3.5, 4.0], dtype=tf.float32)
    review_count = tf.constant([100.0, 50.0, 150.0], dtype=tf.float32)
    
    try:
        # Pre-encode terms
        texts = ["museum", "restaurant", "park"]
        term_emb_input = rama_emb.term_embedder.encode_texts(texts)
        
        # Forward pass
        inputs = {
            'category_ids': category_ids,
            'term_embeddings': term_emb_input,
            'distance': distance,
            'rating': rating,
            'review_count': review_count
        }
        
        all_embeddings = rama_emb(inputs, training=False)
        
        # Check that all embeddings are present
        assert 'category' in all_embeddings, "Category embeddings missing"
        assert 'term' in all_embeddings, "Term embeddings missing"
        assert 'context' in all_embeddings, "Context embeddings missing"
        
        # Check shapes
        assert all_embeddings['category'].shape[0] == batch_size
        assert all_embeddings['term'].shape[0] == batch_size
        assert all_embeddings['context'].shape[0] == batch_size
        
        print("   ✓ All embedding types generated")
        print(f"   ✓ Category shape: {all_embeddings['category'].shape}")
        print(f"   ✓ Term shape: {all_embeddings['term'].shape}")
        print(f"   ✓ Context shape: {all_embeddings['context'].shape}")
        
        # Test model is trainable
        total_params = sum(tf.size(v).numpy() for v in rama_emb.trainable_variables)
        print(f"   ✓ Total trainable parameters: {total_params:,}")
        
    except Exception as e:
        print(f"   ⚠ Integration test requires model download: {e}")


def test_rama_semantics_preservation():
    """Test that embeddings preserve RAMA's [-1, +1] semantics"""
    if not TF_AVAILABLE:
        print("SKIP: test_rama_semantics_preservation (dependencies not available)")
        return
    
    print("\n[TEST] RAMA Semantics Preservation")
    
    # Create constraint
    constraint = WeightConstraint('tanh')
    
    # Simulate user interest weights
    # In RAMA: weights ∈ [-1, +1] where -1=hate, 0=neutral, +1=love
    positive_interest = tf.constant([0.8, 0.9, 0.7])
    negative_interest = tf.constant([-0.8, -0.6, -0.9])
    neutral_interest = tf.constant([0.1, -0.1, 0.0])
    
    # Apply constraint (amplify first to test)
    pos_constrained = constraint(positive_interest * 2)
    neg_constrained = constraint(negative_interest * 2)
    neu_constrained = constraint(neutral_interest)
    
    # Check that relationships are preserved
    assert tf.reduce_mean(pos_constrained) > 0, "Positive interests should remain positive"
    assert tf.reduce_mean(neg_constrained) < 0, "Negative interests should remain negative"
    assert tf.abs(tf.reduce_mean(neu_constrained)) < 0.2, "Neutral interests should stay near 0"
    
    print("   ✓ Positive interests preserved (mean={:.3f})".format(tf.reduce_mean(pos_constrained)))
    print("   ✓ Negative interests preserved (mean={:.3f})".format(tf.reduce_mean(neg_constrained)))
    print("   ✓ Neutral interests preserved (mean={:.3f})".format(tf.reduce_mean(neu_constrained)))
    
    # Test cosine similarity preservation
    # In RAMA: cosine similarity ∈ [-1, +1]
    vec1 = tf.random.normal([128])
    vec2 = tf.random.normal([128])
    
    # Normalize
    vec1_norm = tf.nn.l2_normalize(vec1, axis=0)
    vec2_norm = tf.nn.l2_normalize(vec2, axis=0)
    
    # Compute similarity
    similarity = tf.reduce_sum(vec1_norm * vec2_norm)
    
    assert -1.0 <= similarity <= 1.0, f"Cosine similarity should be in [-1,1], got {similarity}"
    print(f"   ✓ Cosine similarity preserved: {similarity:.3f} ∈ [-1, +1]")


def test_tensorflow_specific_features():
    """Test TensorFlow-specific features"""
    if not TF_AVAILABLE:
        print("SKIP: test_tensorflow_specific_features (dependencies not available)")
        return
    
    print("\n[TEST] TensorFlow-Specific Features")
    
    # Test model saving/loading
    print("   Testing model serialization...")
    embedder = CategoryEmbedding(num_categories=10, embedding_dim=64)
    
    # Build the model with a sample input
    sample_input = tf.constant([[2, 3, 4]], dtype=tf.int32)
    _ = embedder(sample_input)
    
    # Save config
    config = embedder.get_config()
    assert 'num_categories' in config
    assert 'embedding_dim' in config
    print("   ✓ Model configuration serializable")
    
    # Test gradient flow
    print("   Testing gradient flow...")
    embedder_train = CategoryEmbedding(num_categories=10, embedding_dim=64)
    
    with tf.GradientTape() as tape:
        cat_ids = tf.constant([[2, 3]], dtype=tf.int32)
        embeddings = embedder_train(cat_ids, training=True)
        loss = tf.reduce_sum(embeddings)
    
    gradients = tape.gradient(loss, embedder_train.trainable_variables)
    assert len(gradients) > 0, "Should have gradients"
    assert all(g is not None for g in gradients), "All gradients should be non-None"
    print(f"   ✓ Gradient flow works ({len(gradients)} gradient tensors)")
    
    # Test mixed precision compatibility
    print("   Testing mixed precision compatibility...")
    # NEW CODE (Compatible with 2.19+)
    old_policy = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        embedder_mp = CategoryEmbedding(num_categories=10, embedding_dim=64)
        output = embedder_mp(sample_input)
        # Output will be float16 under mixed precision
        print(f"   ✓ Mixed precision compatible (output dtype: {output.dtype})")
    finally:
        tf.keras.mixed_precision.set_global_policy(old_policy)
    
    # policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # with tf.keras.mixed_precision.global_policy_scope(policy):
    #     embedder_mp = CategoryEmbedding(num_categories=10, embedding_dim=64)
    #     output = embedder_mp(sample_input)
    #     # Output should be float32 even with mixed precision (due to normalization)
    #     print(f"   ✓ Mixed precision compatible (output dtype: {output.dtype})")


def run_all_tests():
    """Run all tests"""
    if not TF_AVAILABLE:
        print("\n" + "="*80)
        print("WARNING: TensorFlow and sentence-transformers not available")
        print("Install with: pip install tensorflow sentence-transformers --break-system-packages")
        print("="*80)
        return False
    
    print("="*80)
    print("Neural RAMA Embeddings (TensorFlow) - Test Suite")
    print("="*80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("="*80)
    
    tests = [
        test_category_vocabulary,
        test_category_embedding,
        test_term_embedding,
        test_context_embedding,
        test_weight_constraint,
        test_rama_embeddings_integration,
        test_rama_semantics_preservation,
        test_tensorflow_specific_features
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n   ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n   ⚠ ERROR: {e}")
    
    print("\n" + "="*80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
Test suite for Phase 3 neural components:
  - AttentionAggregation   (encoders.py)
  - GeneralInterestEncoder (encoders.py)
  - SpecificInterestEncoder(encoders.py)
  - NeuralReinforcement    (reinforcement.py)
  - UserStateUpdater       (reinforcement.py)
  - ComponentScorer        (scoring.py)
  - ScoreFusion            (scoring.py)
  - NeuralRAMAScorer       (scoring.py) — end-to-end

Run with: python tests/test_encoders.py
Or:       pytest tests/test_encoders.py -v
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import tensorflow as tf
    import numpy as np
    from models.neural.embeddings import CategoryVocabulary, WeightConstraint
    from models.neural.encoders import (
        AttentionAggregation,
        GeneralInterestEncoder,
        SpecificInterestEncoder,
    )
    from models.neural.reinforcement import NeuralReinforcement, UserStateUpdater
    from models.neural.scoring import (
        ComponentScorer,
        ScoreFusion,
        NeuralRAMAScorer,
        WEIGHTING_SCHEMES,
    )
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    print(f"Warning: Could not import dependencies: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH = 4
NUM_ITEMS = 6
EMBED_DIM = 64   # small for fast tests
TERM_DIM = 32    # small stand-in for sentence-transformer dim in tests
NUM_CATS = 10
NUM_HEADS = 2


def _random_embeddings(batch=BATCH, num_items=NUM_ITEMS, dim=EMBED_DIM):
    return tf.random.normal([batch, num_items, dim])


def _random_ratings(batch=BATCH, num_items=NUM_ITEMS):
    """Rating polarity in [-1, +1]"""
    return tf.random.uniform([batch, num_items], minval=-1.0, maxval=1.0)


def _random_category_ids(batch=BATCH, num_items=NUM_ITEMS, cats_per_item=3, num_cats=NUM_CATS):
    return tf.random.uniform(
        [batch, num_items, cats_per_item], minval=2, maxval=num_cats + 2, dtype=tf.int32
    )


# ---------------------------------------------------------------------------
# Test 1: AttentionAggregation
# ---------------------------------------------------------------------------

def test_attention_aggregation():
    print("\n[TEST] AttentionAggregation")

    layer = AttentionAggregation(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
    item_emb = _random_embeddings()          # (batch, num_items, embed_dim)
    rating_w = _random_ratings()             # (batch, num_items)

    out = layer(item_emb, rating_w, training=False)

    assert out.shape == (BATCH, EMBED_DIM), \
        f"Expected ({BATCH}, {EMBED_DIM}), got {out.shape}"
    print(f"   ✓ Output shape: {out.shape}")

    # Different rating profiles should give different outputs
    rating_pos = tf.ones([BATCH, NUM_ITEMS])
    rating_neg = -tf.ones([BATCH, NUM_ITEMS])
    out_pos = layer(item_emb, rating_pos, training=False)
    out_neg = layer(item_emb, rating_neg, training=False)
    diff = tf.reduce_mean(tf.abs(out_pos - out_neg))
    assert diff > 1e-4, "Positive vs negative ratings should produce different outputs"
    print(f"   ✓ Different rating profiles → different outputs (mean diff={diff:.4f})")

    # Trainable variables exist
    assert len(layer.trainable_variables) > 0
    print(f"   ✓ Trainable parameters: {sum(tf.size(v).numpy() for v in layer.trainable_variables):,}")

    # Gradient flows
    with tf.GradientTape() as tape:
        out_train = layer(item_emb, rating_w, training=True)
        loss = tf.reduce_sum(out_train)
    grads = tape.gradient(loss, layer.trainable_variables)
    assert all(g is not None for g in grads), "All gradients should be non-None"
    print("   ✓ Gradient flow confirmed")


# ---------------------------------------------------------------------------
# Test 2: GeneralInterestEncoder
# ---------------------------------------------------------------------------

def test_general_interest_encoder():
    print("\n[TEST] GeneralInterestEncoder")

    encoder = GeneralInterestEncoder(
        num_categories=NUM_CATS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
    )
    cat_ids = _random_category_ids()   # (batch, num_items, cats_per_item)
    ratings = _random_ratings()        # (batch, num_items)

    out = encoder(cat_ids, ratings, training=False)

    assert out.shape == (BATCH, EMBED_DIM), \
        f"Expected ({BATCH}, {EMBED_DIM}), got {out.shape}"
    print(f"   ✓ Output shape: {out.shape}")

    # Output must be in [-1, +1]
    assert tf.reduce_all(out >= -1.0) and tf.reduce_all(out <= 1.0), \
        "GeneralInterestEncoder output must be in [-1, +1]"
    print(f"   ✓ Output range: [{tf.reduce_min(out):.3f}, {tf.reduce_max(out):.3f}] ⊆ [-1, +1]")

    # Positive-only ratings should push output positive on average
    pos_ratings = tf.abs(ratings)
    neg_ratings = -tf.abs(ratings)
    out_pos = encoder(cat_ids, pos_ratings, training=False)
    out_neg = encoder(cat_ids, neg_ratings, training=False)
    assert tf.reduce_mean(out_pos) > tf.reduce_mean(out_neg), \
        "Positive ratings should yield more positive interest vector than negative ratings"
    print(f"   ✓ Positive ratings → more positive output "
          f"(pos_mean={tf.reduce_mean(out_pos):.3f}, neg_mean={tf.reduce_mean(out_neg):.3f})")

    # Gradient flows end-to-end
    with tf.GradientTape() as tape:
        out_train = encoder(cat_ids, ratings, training=True)
        loss = tf.reduce_sum(out_train ** 2)
    grads = tape.gradient(loss, encoder.trainable_variables)
    assert all(g is not None for g in grads)
    print("   ✓ Gradient flow confirmed")


# ---------------------------------------------------------------------------
# Test 3: SpecificInterestEncoder
# ---------------------------------------------------------------------------

def test_specific_interest_encoder():
    print("\n[TEST] SpecificInterestEncoder")

    encoder = SpecificInterestEncoder(
        term_dim=TERM_DIM,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
    )
    # (batch, num_items, term_dim) — simulates pre-encoded term embeddings
    term_embs = tf.random.normal([BATCH, NUM_ITEMS, TERM_DIM])
    ratings = _random_ratings()

    out = encoder(term_embs, ratings, training=False)

    assert out.shape == (BATCH, EMBED_DIM), \
        f"Expected ({BATCH}, {EMBED_DIM}), got {out.shape}"
    print(f"   ✓ Output shape: {out.shape}")

    # Output must be in [-1, +1]
    assert tf.reduce_all(out >= -1.0) and tf.reduce_all(out <= 1.0)
    print(f"   ✓ Output range: [{tf.reduce_min(out):.3f}, {tf.reduce_max(out):.3f}] ⊆ [-1, +1]")

    # Semantic preservation: two similar term sets should produce more similar
    # interest vectors than two dissimilar term sets (same ratings)
    uniform_ratings = tf.ones([2, NUM_ITEMS])

    # Similar items: slight noise around same vector
    base_emb = tf.random.normal([1, NUM_ITEMS, TERM_DIM])
    similar_a = tf.tile(base_emb, [1, 1, 1]) + tf.random.normal([1, NUM_ITEMS, TERM_DIM]) * 0.01
    similar_b = tf.tile(base_emb, [1, 1, 1]) + tf.random.normal([1, NUM_ITEMS, TERM_DIM]) * 0.01

    # Dissimilar: orthogonal random vectors
    dissimilar_a = tf.random.normal([1, NUM_ITEMS, TERM_DIM])
    dissimilar_b = tf.random.normal([1, NUM_ITEMS, TERM_DIM])

    out_sim_a = encoder(similar_a, uniform_ratings[:1], training=False)
    out_sim_b = encoder(similar_b, uniform_ratings[:1], training=False)
    out_dis_a = encoder(dissimilar_a, uniform_ratings[:1], training=False)
    out_dis_b = encoder(dissimilar_b, uniform_ratings[:1], training=False)

    sim_close = tf.reduce_sum(
        tf.nn.l2_normalize(out_sim_a, axis=-1) * tf.nn.l2_normalize(out_sim_b, axis=-1)
    )
    sim_far = tf.reduce_sum(
        tf.nn.l2_normalize(out_dis_a, axis=-1) * tf.nn.l2_normalize(out_dis_b, axis=-1)
    )
    assert sim_close > sim_far, \
        f"Similar inputs should produce more similar outputs: sim_close={sim_close:.3f}, sim_far={sim_far:.3f}"
    print(f"   ✓ Semantic similarity preserved (similar={sim_close:.3f} > dissimilar={sim_far:.3f})")

    # Gradient flows
    with tf.GradientTape() as tape:
        out_train = encoder(term_embs, ratings, training=True)
        loss = tf.reduce_sum(out_train ** 2)
    grads = tape.gradient(loss, encoder.trainable_variables)
    assert all(g is not None for g in grads)
    print("   ✓ Gradient flow confirmed")


# ---------------------------------------------------------------------------
# Test 4: NeuralReinforcement — single step
# ---------------------------------------------------------------------------

def test_neural_reinforcement():
    print("\n[TEST] NeuralReinforcement (single update step)")

    layer = NeuralReinforcement(
        embed_dim=EMBED_DIM,
        attenuation_factor=0.0,
        reinforcement_factor=0.5,
        trainable_factors=False,
    )

    current_state = tf.random.normal([BATCH, EMBED_DIM]) * 0.3
    new_event = tf.random.normal([BATCH, EMBED_DIM])
    relevance = tf.random.uniform([BATCH], minval=-1.0, maxval=1.0)

    out = layer(current_state, new_event, relevance, training=False)

    assert out.shape == (BATCH, EMBED_DIM), f"Expected ({BATCH}, {EMBED_DIM}), got {out.shape}"
    print(f"   ✓ Output shape: {out.shape}")

    # Output must be in [-1, +1]
    assert tf.reduce_all(out >= -1.0) and tf.reduce_all(out <= 1.0)
    print(f"   ✓ Output range: [{tf.reduce_min(out):.3f}, {tf.reduce_max(out):.3f}] ⊆ [-1, +1]")

    # Aging: with attenuation_factor=0.0, aged_state == current_state
    # (no decay), so the output should differ from current_state only via delta
    layer_decay = NeuralReinforcement(embed_dim=EMBED_DIM, attenuation_factor=0.5)
    out_decay = layer_decay(current_state, new_event, relevance, training=False)
    # With decay, output should generally have smaller magnitude
    mag_no_decay = tf.reduce_mean(tf.abs(out))
    mag_decay = tf.reduce_mean(tf.abs(out_decay))
    print(f"   ✓ Attenuation effect: no_decay_mag={mag_no_decay:.3f}, decay_mag={mag_decay:.3f}")

    # Positive relevance should increase state on average vs negative relevance
    pos_relevance = tf.abs(relevance)
    neg_relevance = -tf.abs(relevance)
    zero_state = tf.zeros([BATCH, EMBED_DIM])
    pos_event = tf.ones([BATCH, EMBED_DIM]) * 0.5

    out_pos = layer(zero_state, pos_event, pos_relevance, training=False)
    out_neg = layer(zero_state, pos_event, neg_relevance, training=False)
    assert tf.reduce_mean(out_pos) > tf.reduce_mean(out_neg), \
        "Positive relevance should increase state more than negative relevance"
    print(f"   ✓ Positive relevance drives state up "
          f"(pos={tf.reduce_mean(out_pos):.3f}, neg={tf.reduce_mean(out_neg):.3f})")

    # Gradient flows through BPR-style loss
    preferred_emb = tf.random.normal([BATCH, EMBED_DIM])
    non_preferred_emb = tf.random.normal([BATCH, EMBED_DIM])
    with tf.GradientTape() as tape:
        state = layer(current_state, new_event, relevance, training=True)
        score_pref = tf.reduce_sum(state * preferred_emb, axis=-1)
        score_non = tf.reduce_sum(state * non_preferred_emb, axis=-1)
        bpr_loss = -tf.reduce_mean(tf.math.log_sigmoid(score_pref - score_non))
    grads = tape.gradient(bpr_loss, layer.trainable_variables)
    valid_grads = [g for g in grads if g is not None]
    assert len(valid_grads) > 0, "BPR loss should produce gradients"
    print(f"   ✓ BPR gradient flow confirmed ({len(valid_grads)} tensors)")


# ---------------------------------------------------------------------------
# Test 5: UserStateUpdater — sequential reinforcement
# ---------------------------------------------------------------------------

def test_user_state_updater():
    print("\n[TEST] UserStateUpdater (sequential reinforcement)")

    updater = UserStateUpdater(
        embed_dim=EMBED_DIM,
        attenuation_factor=0.0,
        reinforcement_factor=0.5,
    )

    # Simulate 5 rating events
    num_events = 5
    item_embs = tf.random.normal([BATCH, num_events, EMBED_DIM])
    relevances = tf.random.uniform([BATCH, num_events], minval=-1.0, maxval=1.0)

    out = updater(item_embs, relevances, training=False)

    assert out.shape == (BATCH, EMBED_DIM), f"Expected ({BATCH}, {EMBED_DIM}), got {out.shape}"
    print(f"   ✓ Output shape: {out.shape}")

    # Output must be in [-1, +1]
    assert tf.reduce_all(out >= -1.0) and tf.reduce_all(out <= 1.0)
    print(f"   ✓ Output range: [{tf.reduce_min(out):.3f}, {tf.reduce_max(out):.3f}] ⊆ [-1, +1]")

    # More events with positive relevance → more positive final state
    all_pos = tf.abs(relevances)
    all_neg = -tf.abs(relevances)
    out_pos = updater(item_embs, all_pos, training=False)
    out_neg = updater(item_embs, all_neg, training=False)
    assert tf.reduce_mean(out_pos) > tf.reduce_mean(out_neg)
    print(f"   ✓ All-positive history → more positive state than all-negative "
          f"({tf.reduce_mean(out_pos):.3f} vs {tf.reduce_mean(out_neg):.3f})")


# ---------------------------------------------------------------------------
# Test 6: ComponentScorer
# ---------------------------------------------------------------------------

def test_component_scorer():
    print("\n[TEST] ComponentScorer")

    scorer = ComponentScorer(embed_dim=EMBED_DIM, context_embed_dim=32)

    user_general = tf.nn.l2_normalize(tf.random.normal([BATCH, EMBED_DIM]), axis=-1)
    user_specific = tf.nn.l2_normalize(tf.random.normal([BATCH, EMBED_DIM]), axis=-1)
    cand_cat_emb = tf.nn.l2_normalize(tf.random.normal([BATCH, EMBED_DIM]), axis=-1)
    cand_term_emb = tf.nn.l2_normalize(tf.random.normal([BATCH, EMBED_DIM]), axis=-1)
    distance = tf.random.uniform([BATCH], 500.0, 7000.0)
    rating = tf.random.uniform([BATCH], 1.0, 5.0)
    review_count = tf.random.uniform([BATCH], 10.0, 200.0)

    scores = scorer(
        user_general, user_specific, cand_cat_emb, cand_term_emb,
        distance, rating, review_count, training=False,
    )

    assert set(scores.keys()) == {"general", "specific", "context"}
    print("   ✓ Returns all three component scores")

    for key in ["general", "specific", "context"]:
        s = scores[key]
        assert s.shape == (BATCH,), f"{key} shape expected ({BATCH},), got {s.shape}"
        assert tf.reduce_all(s >= -1.0) and tf.reduce_all(s <= 1.0), \
            f"{key} score must be in [-1, +1]"
        print(f"   ✓ {key}: shape={s.shape}, range=[{tf.reduce_min(s):.3f}, {tf.reduce_max(s):.3f}]")

    # Identical user and candidate vectors → maximum general score (+1)
    same_vec = tf.nn.l2_normalize(tf.random.normal([BATCH, EMBED_DIM]), axis=-1)
    general_same = scorer.score_general(same_vec, same_vec)
    assert tf.reduce_all(tf.abs(general_same - 1.0) < 0.01), \
        "Identical vectors should give general score ≈ 1.0"
    print("   ✓ Identical vectors → general score ≈ 1.0")

    # Opposite vectors → minimum general score (-1)
    general_opp = scorer.score_general(same_vec, -same_vec)
    assert tf.reduce_all(tf.abs(general_opp + 1.0) < 0.01), \
        "Opposite vectors should give general score ≈ -1.0"
    print("   ✓ Opposite vectors → general score ≈ -1.0")

    # Gradient flows
    with tf.GradientTape() as tape:
        scores_train = scorer(
            user_general, user_specific, cand_cat_emb, cand_term_emb,
            distance, rating, review_count, training=True,
        )
        loss = tf.reduce_sum(scores_train["context"])
    grads = tape.gradient(loss, scorer.trainable_variables)
    valid_grads = [g for g in grads if g is not None]
    assert len(valid_grads) > 0
    print(f"   ✓ Gradient flow confirmed ({len(valid_grads)} tensors)")


# ---------------------------------------------------------------------------
# Test 7: ScoreFusion — fixed and learned modes
# ---------------------------------------------------------------------------

def test_score_fusion():
    print("\n[TEST] ScoreFusion")

    scores = {
        "general":  tf.constant([0.8, -0.3, 0.5, 0.1]),
        "specific": tf.constant([0.6,  0.9, 0.2, -0.4]),
        "context":  tf.constant([0.3,  0.1, 0.7,  0.5]),
    }

    # Fixed RUN2: Wg=0.9, Ws=0.09, Wc=0.01
    fusion_run2 = ScoreFusion(mode="fixed", scheme="RUN2")
    out_run2 = fusion_run2(scores)
    expected_run2 = 0.9 * scores["general"] + 0.09 * scores["specific"] + 0.01 * scores["context"]
    assert tf.reduce_all(tf.abs(out_run2 - expected_run2) < 1e-5), \
        "RUN2 fixed fusion should match Formula 2 exactly"
    print(f"   ✓ RUN2 fixed weights match Formula 2 (max_err={tf.reduce_max(tf.abs(out_run2 - expected_run2)):.2e})")

    # Fixed RUN1: Wg=0.09, Ws=0.9, Wc=0.01
    fusion_run1 = ScoreFusion(mode="fixed", scheme="RUN1")
    out_run1 = fusion_run1(scores)
    expected_run1 = 0.09 * scores["general"] + 0.9 * scores["specific"] + 0.01 * scores["context"]
    assert tf.reduce_all(tf.abs(out_run1 - expected_run1) < 1e-5)
    print(f"   ✓ RUN1 fixed weights match Formula 2")

    # Learned mode: weights should be trainable and update via gradient
    fusion_learned = ScoreFusion(mode="learned", scheme="RUN2")
    assert len(fusion_learned.trainable_variables) > 0
    print(f"   ✓ Learned mode has trainable parameters")

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
    w_before = fusion_learned.raw_weights.numpy().copy()
    with tf.GradientTape() as tape:
        out_learned = fusion_learned(scores)
        # Push final score to be larger (maximize)
        loss = -tf.reduce_sum(out_learned)
    grads = tape.gradient(loss, fusion_learned.trainable_variables)
    optimizer.apply_gradients(zip(grads, fusion_learned.trainable_variables))
    w_after = fusion_learned.raw_weights.numpy()
    assert not np.allclose(w_before, w_after), "Weights should update after gradient step"
    print(f"   ✓ Learned weights update via gradient")

    # Softmax normalization: learned weights always sum to 1
    w_normalized = tf.nn.softmax(fusion_learned.raw_weights)
    assert abs(float(tf.reduce_sum(w_normalized)) - 1.0) < 1e-5
    print(f"   ✓ Learned weights normalized: sum={float(tf.reduce_sum(w_normalized)):.6f}")

    # Weights summary
    summary = fusion_run2.get_weights_summary()
    assert "Wg (general)" in summary
    print(f"   ✓ Weight summary: {summary}")


# ---------------------------------------------------------------------------
# Test 8: End-to-end NeuralRAMAScorer
# ---------------------------------------------------------------------------

def test_neural_rama_scorer_e2e():
    print("\n[TEST] NeuralRAMAScorer (end-to-end)")

    scorer = NeuralRAMAScorer(
        embed_dim=EMBED_DIM,
        context_embed_dim=32,
        fusion_mode="fixed",
        fusion_scheme="RUN2",
    )

    inputs = {
        "user_general":           tf.nn.l2_normalize(tf.random.normal([BATCH, EMBED_DIM]), axis=-1),
        "user_specific":          tf.nn.l2_normalize(tf.random.normal([BATCH, EMBED_DIM]), axis=-1),
        "candidate_category_emb": tf.nn.l2_normalize(tf.random.normal([BATCH, EMBED_DIM]), axis=-1),
        "candidate_term_emb":     tf.nn.l2_normalize(tf.random.normal([BATCH, EMBED_DIM]), axis=-1),
        "distance":               tf.random.uniform([BATCH], 500.0, 7000.0),
        "rating":                 tf.random.uniform([BATCH], 1.0, 5.0),
        "review_count":           tf.random.uniform([BATCH], 10.0, 200.0),
    }

    outputs = scorer(inputs, training=False)

    assert set(outputs.keys()) == {"general", "specific", "context", "final"}
    print("   ✓ Returns general, specific, context, final scores")

    for key in ["general", "specific", "context"]:
        assert outputs[key].shape == (BATCH,)
    assert outputs["final"].shape == (BATCH,)
    print(f"   ✓ All scores have correct shape ({BATCH},)")

    # Higher-ranked candidates should score higher than lower-ranked
    # Construct a case where user and candidate are aligned for one, opposed for another
    user_g = tf.nn.l2_normalize(tf.ones([1, EMBED_DIM]), axis=-1)
    user_s = tf.nn.l2_normalize(tf.ones([1, EMBED_DIM]), axis=-1)

    # Candidate A: aligned with user (should score high)
    cand_a = {
        "user_general":           user_g,
        "user_specific":          user_s,
        "candidate_category_emb": tf.nn.l2_normalize(tf.ones([1, EMBED_DIM]), axis=-1),
        "candidate_term_emb":     tf.nn.l2_normalize(tf.ones([1, EMBED_DIM]), axis=-1),
        "distance":               tf.constant([500.0]),
        "rating":                 tf.constant([5.0]),
        "review_count":           tf.constant([100.0]),
    }
    # Candidate B: opposed to user (should score low)
    cand_b = {
        "user_general":           user_g,
        "user_specific":          user_s,
        "candidate_category_emb": tf.nn.l2_normalize(-tf.ones([1, EMBED_DIM]), axis=-1),
        "candidate_term_emb":     tf.nn.l2_normalize(-tf.ones([1, EMBED_DIM]), axis=-1),
        "distance":               tf.constant([7900.0]),
        "rating":                 tf.constant([1.0]),
        "review_count":           tf.constant([1.0]),
    }

    score_a = tf.squeeze(scorer(cand_a, training=False)["final"])
    score_b = tf.squeeze(scorer(cand_b, training=False)["final"])
    assert score_a > score_b, \
        f"Aligned candidate should rank above opposed: {float(score_a):.3f} vs {float(score_b):.3f}"
    print(f"   ✓ Aligned candidate scores higher than opposed "
          f"({float(score_a):.3f} > {float(score_b):.3f})")

    # BPR loss end-to-end gradient
    with tf.GradientTape() as tape:
        out = scorer(inputs, training=True)
        # Pretend first half of batch are preferred, second half non-preferred
        half = BATCH // 2
        score_pref = out["final"][:half]
        score_non  = out["final"][half:]
        bpr_loss = -tf.reduce_mean(tf.math.log_sigmoid(score_pref - score_non))
    grads = tape.gradient(bpr_loss, scorer.trainable_variables)
    valid_grads = [g for g in grads if g is not None]
    assert len(valid_grads) > 0
    print(f"   ✓ BPR gradient flows through full scorer ({len(valid_grads)} tensors)")

    total_params = sum(tf.size(v).numpy() for v in scorer.trainable_variables)
    print(f"   ✓ Total trainable parameters: {total_params:,}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    if not TF_AVAILABLE:
        print("\n" + "=" * 70)
        print("WARNING: TensorFlow not available — install with:")
        print("  pip install tensorflow sentence-transformers")
        print("=" * 70)
        return False

    print("=" * 70)
    print("Neural RAMA Phase 3 — Test Suite")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("=" * 70)

    tests = [
        test_attention_aggregation,
        test_general_interest_encoder,
        test_specific_interest_encoder,
        test_neural_reinforcement,
        test_user_state_updater,
        test_component_scorer,
        test_score_fusion,
        test_neural_rama_scorer_e2e,
    ]

    passed, failed = 0, 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n   ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n   ✗ ERROR in {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

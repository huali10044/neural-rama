"""
Test suite for classical RAMA (Phase 1):
  - InterestElement, InterestFacet, RAMAUserModel  (user_model.py)
  - RAMA: aging, reinforcement, insertion          (rama.py)
  - RAMAScorer: cosine similarity, context, fusion (scoring.py)

Run with: python tests/test_classical_rama.py
Or:       pytest tests/test_classical_rama.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from types import SimpleNamespace

from models.classical.user_model import InterestElement, InterestFacet, RAMAUserModel
from models.classical.rama import RAMA
from models.classical.scoring import RAMAScorer


# ---------------------------------------------------------------------------
# Test 1: InterestElement / InterestFacet / RAMAUserModel
# ---------------------------------------------------------------------------

def test_interest_element_clamping():
    print("\n[TEST] InterestElement weight clamping")

    elem = InterestElement("museums", weight=0.5)
    assert elem.weight == 0.5
    print("   [OK] Weight within range preserved")

    elem_high = InterestElement("art", weight=2.5)
    assert elem_high.weight == 1.0
    print("   [OK] Weight > 1.0 clamped to 1.0")

    elem_low = InterestElement("noise", weight=-3.0)
    assert elem_low.weight == -1.0
    print("   [OK] Weight < -1.0 clamped to -1.0")


def test_interest_facet():
    print("\n[TEST] InterestFacet")

    facet = InterestFacet("general")
    assert len(facet) == 0
    print("   [OK] Empty facet has length 0")

    facet.add_element("museums", 0.7)
    facet.add_element("parks", -0.3)
    assert len(facet) == 2
    print("   [OK] add_element increases length")

    assert facet.get_weight("museums") == 0.7
    assert facet.get_weight("unknown") == 0.0
    print("   [OK] get_weight returns 0.0 for missing elements")

    weights = facet.get_all_weights()
    assert weights == {"museums": 0.7, "parks": -0.3}
    print("   [OK] get_all_weights returns full dict")

    # Re-adding an element overwrites it
    facet.add_element("museums", 0.9)
    assert facet.get_weight("museums") == 0.9
    assert len(facet) == 2
    print("   [OK] Re-adding an element overwrites weight, not length")


def test_rama_user_model():
    print("\n[TEST] RAMAUserModel")

    model = RAMAUserModel("user_001")
    assert model.user_id == "user_001"
    assert len(model.general_facet) == 0
    assert len(model.specific_facet) == 0
    assert model.general_facet.facet_type == "general"
    assert model.specific_facet.facet_type == "specific"
    print("   [OK] User model initializes with empty general/specific facets")


# ---------------------------------------------------------------------------
# Test 2: RAMA — Formula 3 (aging), Formula 4 (reinforcement), Formula 5 (insertion)
# ---------------------------------------------------------------------------

def test_formula_5_new_element_insertion():
    print("\n[TEST] RAMA Formula 5 — new element insertion")

    rama = RAMA(reinforcement_factor=0.5, attenuation_factor=0.0)
    facet = InterestFacet("general")

    # initialWeight = eventRelevance * reinforcementFactor * mentionFrequency
    rama._update_element(facet, "museums", event_relevance=1.0, mention_frequency=1)
    expected = 1.0 * 0.5 * 1
    assert abs(facet.get_weight("museums") - expected) == 0.0
    print(f"   [OK] New element weight = {facet.get_weight('museums'):.3f} (expected {expected})")

    # Mention frequency scales the initial weight
    facet2 = InterestFacet("general")
    rama._update_element(facet2, "parks", event_relevance=1.0, mention_frequency=3)
    expected2 = min(1.0, 1.0 * 0.5 * 3)  # clamped to [-1, +1]
    assert abs(facet2.get_weight("parks") - expected2) < 1e-9
    print(f"   [OK] Higher mention frequency clamps at +1.0: {facet2.get_weight('parks'):.3f}")


def test_formula_4_reinforcement():
    print("\n[TEST] RAMA Formula 4 — reinforcement")

    rama = RAMA(reinforcement_factor=0.5, attenuation_factor=0.0)

    # newWeight = oldWeight + eventRelevance * reinforcementFactor * mentionFrequency * (1 - |oldWeight|)
    old_weight = 0.2
    new_weight = rama._reinforce(old_weight, event_relevance=1.0, mention_frequency=1)
    expected = old_weight + 1.0 * 0.5 * 1 * (1 - abs(old_weight))
    assert abs(new_weight - expected) < 1e-9
    print(f"   [OK] Reinforced weight = {new_weight:.3f} (expected {expected:.3f})")

    # Positive reinforcement moves weight toward +1, never exceeding it
    weight = 0.0
    for _ in range(20):
        weight = rama._reinforce(weight, event_relevance=1.0, mention_frequency=1)
    assert weight <= 1.0
    print(f"   [OK] Repeated positive reinforcement converges toward +1.0: {weight:.6f}")

    # Negative reinforcement moves weight toward -1, never below it
    weight = 0.0
    for _ in range(20):
        weight = rama._reinforce(weight, event_relevance=-1.0, mention_frequency=1)
    assert weight >= -1.0
    print(f"   [OK] Repeated negative reinforcement converges toward -1.0: {weight:.6f}")

    # Zero relevance leaves weight unchanged
    unchanged = rama._reinforce(0.4, event_relevance=0.0, mention_frequency=2)
    assert abs(unchanged - 0.4) < 1e-9
    print("   [OK] Zero event relevance leaves weight unchanged")


def test_formula_3_aging():
    print("\n[TEST] RAMA Formula 3 — aging/decay")

    # No attenuation: weights unchanged
    rama_no_decay = RAMA(reinforcement_factor=0.5, attenuation_factor=0.0)
    facet = InterestFacet("general")
    facet.add_element("museums", 0.8)
    rama_no_decay._age_model(facet)
    assert facet.get_weight("museums") == 0.8
    print("   [OK] attenuation_factor=0.0 leaves weights unchanged")

    # With attenuation: newWeight = oldWeight * (1 - attenuationFactor)
    rama_decay = RAMA(reinforcement_factor=0.5, attenuation_factor=0.1)
    facet2 = InterestFacet("general")
    facet2.add_element("parks", 0.8)
    rama_decay._age_model(facet2)
    expected = 0.8 * (1 - 0.1)
    assert abs(facet2.get_weight("parks") - expected) < 1e-9
    print(f"   [OK] attenuation_factor=0.1 decays weight to {facet2.get_weight('parks'):.3f} (expected {expected:.3f})")


def test_count_frequencies():
    print("\n[TEST] RAMA._count_frequencies")

    rama = RAMA()
    freq = rama._count_frequencies(["museum", "art", "museum", "sculpture", "art", "art"])
    assert freq == {"museum": 2, "art": 3, "sculpture": 1}
    print(f"   [OK] Frequency counts correct: {freq}")

    assert rama._count_frequencies([]) == {}
    print("   [OK] Empty input returns empty dict")


def test_build_user_model_integration():
    print("\n[TEST] RAMA.build_user_model (end-to-end)")

    rama = RAMA(reinforcement_factor=0.5, attenuation_factor=0.0)

    # Fake rating events: minimal objects with event_relevance, poi, timestamp
    event1 = SimpleNamespace(event_relevance=1.0, poi="poi_1", timestamp=100)
    event2 = SimpleNamespace(event_relevance=-0.5, poi="poi_2", timestamp=200)

    def extract_categories_fn(poi):
        return {"poi_1": ["museums"], "poi_2": ["parks"]}[poi]

    def extract_terms_fn(poi):
        return {"poi_1": ["art", "sculpture"], "poi_2": ["trail"]}[poi]

    model = rama.build_user_model(
        user_id="user_42",
        rating_events=[event1, event2],
        extract_categories_fn=extract_categories_fn,
        extract_terms_fn=extract_terms_fn,
    )

    assert model.user_id == "user_42"
    assert len(model.general_facet) == 2  # museums, parks
    assert len(model.specific_facet) == 3  # art, sculpture, trail
    print(f"   [OK] Built model: general={len(model.general_facet)}, specific={len(model.specific_facet)}")

    # Positive event → positive weight; negative event → negative weight
    assert model.general_facet.get_weight("museums") > 0
    assert model.general_facet.get_weight("parks") < 0
    print("   [OK] Weight sign matches event relevance sign")

    # Timestamps and event counts tracked
    assert model.general_facet.num_events == 2
    assert model.general_facet.timestamps == [100, 200]
    print("   [OK] num_events and timestamps tracked correctly")


# ---------------------------------------------------------------------------
# Test 3: RAMAScorer — cosine similarity, component scores, fusion (Formula 2)
# ---------------------------------------------------------------------------

def test_cosine_similarity():
    print("\n[TEST] RAMAScorer.cosine_similarity")

    scorer = RAMAScorer()

    # Identical vectors → similarity 1.0
    sim = scorer.cosine_similarity({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0})
    assert abs(sim - 1.0) < 1e-9
    print(f"   [OK] Identical vectors: similarity={sim:.3f}")

    # Orthogonal (disjoint) vectors → similarity 0.0
    sim_orth = scorer.cosine_similarity({"a": 1.0}, {"b": 1.0})
    assert abs(sim_orth - 0.0) < 1e-9
    print(f"   [OK] Disjoint vectors: similarity={sim_orth:.3f}")

    # Empty vectors → 0.0, no division by zero
    assert scorer.cosine_similarity({}, {}) == 0.0
    assert scorer.cosine_similarity({"a": 0.0}, {"a": 0.0}) == 0.0
    print("   [OK] Empty/zero-norm vectors return 0.0 without error")

    # Opposite vectors → similarity -1.0
    sim_opp = scorer.cosine_similarity({"a": 1.0}, {"a": -1.0})
    assert abs(sim_opp - (-1.0)) < 1e-9
    print(f"   [OK] Opposite vectors: similarity={sim_opp:.3f}")


def test_score_general_and_specific_interest():
    print("\n[TEST] RAMAScorer general/specific interest scoring")

    scorer = RAMAScorer()
    model = RAMAUserModel("user_1")
    model.general_facet.add_element("museums", 1.0)
    model.general_facet.add_element("parks", -1.0)
    model.specific_facet.add_element("art", 1.0)

    # Candidate matching a liked category scores positively
    score_like = scorer.score_general_interest(model, ["museums"])
    assert score_like > 0
    print(f"   [OK] Candidate matching liked category: score={score_like:.3f}")

    # Candidate matching a disliked category scores negatively
    score_dislike = scorer.score_general_interest(model, ["parks"])
    assert score_dislike < 0
    print(f"   [OK] Candidate matching disliked category: score={score_dislike:.3f}")

    score_specific = scorer.score_specific_interest(model, ["art"])
    assert score_specific > 0
    print(f"   [OK] Specific interest scoring: score={score_specific:.3f}")


def test_score_context():
    print("\n[TEST] RAMAScorer.score_context")

    scorer = RAMAScorer()

    # Close, well-rated, popular candidate scores higher than far, poorly-rated, obscure one
    score_good = scorer.score_context(distance_meters=500, yelp_rating=5.0, review_count=100)
    score_bad = scorer.score_context(distance_meters=8000, yelp_rating=1.0, review_count=0)
    assert score_good > score_bad
    print(f"   [OK] Good context ({score_good:.3f}) > bad context ({score_bad:.3f})")


def test_score_candidate_formula_2():
    print("\n[TEST] RAMAScorer.score_candidate — Formula 2 fusion")

    scorer = RAMAScorer()
    model = RAMAUserModel("user_1")
    model.general_facet.add_element("museums", 1.0)
    model.specific_facet.add_element("art", 1.0)

    candidate = SimpleNamespace(
        categories=["museums"],
        name="Art Museum",
        snippet="art and sculpture",
        distance_meters=500,
        yelp_rating=4.5,
        review_count=80,
    )

    scores = scorer.score_candidate(
        model, candidate, general_weight=0.09, specific_weight=0.9, context_weight=0.01
    )

    assert set(scores.keys()) == {"general_score", "specific_score", "context_score", "final_score"}
    print(f"   [OK] All score components present: {scores}")

    # Formula 2: final = Wg*general + Ws*specific + Wc*context
    expected_final = (
        0.09 * scores["general_score"]
        + 0.9 * scores["specific_score"]
        + 0.01 * scores["context_score"]
    )
    assert abs(scores["final_score"] - expected_final) < 1e-9
    print(f"   [OK] Final score matches weighted fusion formula: {scores['final_score']:.4f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    print("=" * 70)
    print("Neural RAMA Phase 1 — Classical RAMA Tests")
    print("=" * 70)

    tests = [
        test_interest_element_clamping,
        test_interest_facet,
        test_rama_user_model,
        test_formula_5_new_element_insertion,
        test_formula_4_reinforcement,
        test_formula_3_aging,
        test_count_frequencies,
        test_build_user_model_integration,
        test_cosine_similarity,
        test_score_general_and_specific_interest,
        test_score_context,
        test_score_candidate_formula_2,
    ]

    passed, failed = 0, 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n   X FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n   X ERROR in {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Phase 1 Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

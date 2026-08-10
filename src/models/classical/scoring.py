"""
Classical RAMA candidate scoring.

Cosine similarity-based component scoring (general / specific / context)
and final score fusion (Formula 2 from the paper).
"""
from typing import Dict, List
import numpy as np

from models.classical.user_model import RAMAUserModel


class RAMAScorer:
    """Score candidates using cosine similarity"""

    @staticmethod
    def cosine_similarity(weights1: Dict[str, float], weights2: Dict[str, float]) -> float:
        """Compute cosine similarity between two weight vectors"""
        all_elements = set(weights1.keys()) | set(weights2.keys())

        if not all_elements:
            return 0.0

        vec1 = np.array([weights1.get(e, 0.0) for e in all_elements])
        vec2 = np.array([weights2.get(e, 0.0) for e in all_elements])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def score_general_interest(self, user_model: RAMAUserModel, candidate_categories: List[str]) -> float:
        """Score based on general interests (categories)"""
        candidate_weights = {cat: 1.0 for cat in candidate_categories}
        user_weights = user_model.general_facet.get_all_weights()
        return self.cosine_similarity(user_weights, candidate_weights)

    def score_specific_interest(self, user_model: RAMAUserModel, candidate_terms: List[str]) -> float:
        """Score based on specific interests (terms)"""
        candidate_weights = {term: 1.0 for term in candidate_terms}
        user_weights = user_model.specific_facet.get_all_weights()
        return self.cosine_similarity(user_weights, candidate_weights)

    def score_context(self, distance_meters: float, yelp_rating: float, review_count: int,
                     distance_weight: float = 0.6, rating_weight: float = 0.3, review_weight: float = 0.1,
                     distance_limit: float = 8000.0, review_limit: int = 100) -> float:
        """Context score based on distance, rating, and review count"""
        distance_score = -2 * (distance_meters / distance_limit) + 1
        distance_score = max(-1.0, min(1.0, distance_score))

        rating_score = 0.5 * yelp_rating - 1.5

        review_score = 2.0 * (review_count / review_limit) - 1
        review_score = max(-1.0, min(1.0, review_score))

        context_score = distance_weight * distance_score + rating_weight * rating_score + review_weight * review_score
        return context_score

    def score_candidate(self, user_model: RAMAUserModel, candidate, general_weight: float = 0.09,
                       specific_weight: float = 0.9, context_weight: float = 0.01) -> Dict[str, float]:
        """Compute final score for candidate (Formula 2)"""
        general_score = self.score_general_interest(user_model, candidate.categories)

        terms = candidate.name.lower().split() + candidate.snippet.lower().split()
        specific_score = self.score_specific_interest(user_model, terms)

        context_score = self.score_context(candidate.distance_meters, candidate.yelp_rating, candidate.review_count)

        final_score = general_weight * general_score + specific_weight * specific_score + context_weight * context_score

        return {
            "general_score": general_score,
            "specific_score": specific_score,
            "context_score": context_score,
            "final_score": final_score
        }

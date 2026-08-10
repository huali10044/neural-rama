"""
Classical RAMA (Reinforcement and Aging Modeling Algorithm)
Exact implementation of formulas from the paper
"""
from typing import Dict, List
from collections import defaultdict

from models.classical.user_model import InterestFacet, RAMAUserModel


class RAMA:
    """Reinforcement and Aging Modeling Algorithm"""

    def __init__(self, reinforcement_factor: float = 0.5, attenuation_factor: float = 0.0):
        self.reinforcement_factor = reinforcement_factor
        self.attenuation_factor = attenuation_factor

    def build_user_model(self, user_id: str, rating_events: List, extract_categories_fn, extract_terms_fn) -> RAMAUserModel:
        """Build user model from rating events"""
        model = RAMAUserModel(user_id)

        for event in rating_events:
            self._process_event(
                model=model,
                event_relevance=event.event_relevance,
                categories=extract_categories_fn(event.poi),
                terms=extract_terms_fn(event.poi),
                timestamp=event.timestamp
            )

        return model

    def _process_event(self, model: RAMAUserModel, event_relevance: float, categories: List[str], terms: List[str], timestamp: int = None):
        """Process single rating event"""
        self._age_model(model.general_facet)
        self._age_model(model.specific_facet)

        category_freq = self._count_frequencies(categories)
        term_freq = self._count_frequencies(terms)

        for category, freq in category_freq.items():
            self._update_element(model.general_facet, category, event_relevance, freq)

        for term, freq in term_freq.items():
            self._update_element(model.specific_facet, term, event_relevance, freq)

        model.general_facet.num_events += 1
        model.specific_facet.num_events += 1
        if timestamp is not None:
            model.general_facet.timestamps.append(timestamp)
            model.specific_facet.timestamps.append(timestamp)

    def _age_model(self, facet: InterestFacet):
        """Formula 3: newWeight = oldWeight * (1 - attenuationFactor)"""
        for element in facet.elements.values():
            element.weight *= (1 - self.attenuation_factor)

    def _update_element(self, facet: InterestFacet, element: str, event_relevance: float, mention_frequency: int):
        """Reinforce existing element or insert new element"""
        if element in facet.elements:
            old_weight = facet.elements[element].weight
            new_weight = self._reinforce(old_weight, event_relevance, mention_frequency)
            facet.elements[element].weight = new_weight
        else:
            initial_weight = self._compute_initial_weight(event_relevance, mention_frequency)
            facet.add_element(element, initial_weight)

    def _reinforce(self, old_weight: float, event_relevance: float, mention_frequency: int) -> float:
        """Formula 4: Reinforcement mechanism"""
        delta = event_relevance * self.reinforcement_factor * mention_frequency * (1 - abs(old_weight))
        new_weight = old_weight + delta
        return max(-1.0, min(1.0, new_weight))

    def _compute_initial_weight(self, event_relevance: float, mention_frequency: int) -> float:
        """Formula 5: Initial weight for new elements"""
        weight = event_relevance * self.reinforcement_factor * mention_frequency
        return max(-1.0, min(1.0, weight))

    @staticmethod
    def _count_frequencies(elements: List[str]) -> Dict[str, int]:
        """Count mention frequency of each element"""
        freq = defaultdict(int)
        for elem in elements:
            freq[elem] += 1
        return dict(freq)

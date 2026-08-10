"""
User interest model for classical RAMA.

Holds the general (category) and specific (term) interest facets that
together represent a user's profile, as described in Section 2.1 of the
paper.
"""
from typing import Dict


class InterestElement:
    """Single interest element with weight in [-1, +1]"""

    def __init__(self, element: str, weight: float = 0.0):
        self.element = element
        self.weight = max(-1.0, min(1.0, weight))

    def __repr__(self):
        return f"IE({self.element}, w={self.weight:.3f})"


class InterestFacet:
    """A facet contains interest elements for one aspect of user interests"""

    def __init__(self, facet_type: str = "general"):
        self.facet_type = facet_type
        self.elements: Dict[str, InterestElement] = {}
        self.num_events = 0
        self.timestamps = []

    def add_element(self, element: str, weight: float):
        """Add or update interest element"""
        self.elements[element] = InterestElement(element, weight)

    def get_weight(self, element: str) -> float:
        """Get weight of element, 0 if not present"""
        return self.elements[element].weight if element in self.elements else 0.0

    def get_all_weights(self) -> Dict[str, float]:
        """Get all element weights as dict"""
        return {elem: ie.weight for elem, ie in self.elements.items()}

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return f"Facet({self.facet_type}, {len(self)} elements)"


class RAMAUserModel:
    """User model containing general and specific interest facets"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.general_facet = InterestFacet("general")
        self.specific_facet = InterestFacet("specific")

    def __repr__(self):
        return f"UserModel({self.user_id}, G:{len(self.general_facet)}, S:{len(self.specific_facet)})"

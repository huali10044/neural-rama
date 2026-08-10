"""
Core data structures matching TREC format
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class Rating(Enum):
    """User ratings from 0 (strongly uninterested) to 4 (strongly interested)"""
    STRONGLY_UNINTERESTED = 0
    UNINTERESTED = 1
    NEUTRAL = 2
    INTERESTED = 3
    STRONGLY_INTERESTED = 4


@dataclass
class PointOfInterest:
    """Represents an attraction/venue (TREC example)"""
    poi_id: str
    title: str
    description: str
    url: str
    categories: List[str] = field(default_factory=list)
    location: Optional[str] = None
    
    def __repr__(self):
        return f"POI({self.poi_id}, {self.title[:30]}...)"


@dataclass
class RatingEvent:
    """A user rating event - explicit relevance feedback"""
    poi: PointOfInterest
    rating: Rating
    timestamp: Optional[int] = None
    
    @property
    def event_relevance(self) -> float:
        """
        Map rating [0-4] to event relevance [-1, +1]
        Table 6 from paper
        """
        mapping = {
            Rating.STRONGLY_UNINTERESTED: -1.0,
            Rating.UNINTERESTED: -0.5,
            Rating.NEUTRAL: 0.0,
            Rating.INTERESTED: 0.5,
            Rating.STRONGLY_INTERESTED: 1.0
        }
        return mapping[self.rating]
    
    @property
    def is_positive(self) -> bool:
        """Positive event (user likes it)"""
        return self.event_relevance > 0
    
    @property
    def is_negative(self) -> bool:
        """Negative event (user dislikes it)"""
        return self.event_relevance < 0


@dataclass
class UserProfile:
    """User profile with rated examples"""
    user_id: str
    rating_events: List[RatingEvent] = field(default_factory=list)
    
    def add_rating(self, poi: PointOfInterest, rating: Rating, timestamp: Optional[int] = None):
        """Add a rating event to profile"""
        event = RatingEvent(poi=poi, rating=rating, timestamp=timestamp)
        self.rating_events.append(event)
    
    @property
    def num_events(self) -> int:
        return len(self.rating_events)
    
    @property
    def positive_events(self) -> List[RatingEvent]:
        return [e for e in self.rating_events if e.is_positive]
    
    @property
    def negative_events(self) -> List[RatingEvent]:
        return [e for e in self.rating_events if e.is_negative]


@dataclass
class Context:
    """Context for recommendation (location-based)"""
    context_id: str
    city: str
    state: str
    latitude: float
    longitude: float
    
    def __repr__(self):
        return f"Context({self.city}, {self.state})"


@dataclass
class Candidate:
    """Candidate suggestion with Yelp-like data"""
    candidate_id: str
    name: str
    categories: List[str]
    snippet: str
    url: str
    distance_meters: float
    yelp_rating: float  # 1-5 stars
    review_count: int
    latitude: float
    longitude: float
    
    def __repr__(self):
        return f"Candidate({self.name}, {self.yelp_rating}★)"


@dataclass
class ScoredCandidate:
    """Candidate with computed scores"""
    candidate: Candidate
    general_interest_score: float
    specific_interest_score: float
    context_score: float
    final_score: float
    rank: Optional[int] = None
    
    def __repr__(self):
        return f"Scored({self.candidate.name}, final={self.final_score:.3f}, rank={self.rank})"

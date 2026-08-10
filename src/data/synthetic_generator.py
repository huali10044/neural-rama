"""
Generate synthetic TREC-format data for development
Mimics TREC 2014 Contextual Suggestion Track data
"""
import random
import json
from typing import List
from pathlib import Path
import numpy as np

from .data_loader import (
    PointOfInterest, Rating, UserProfile, Context, Candidate
)


# Yelp categories from Figure 2 of paper
YELP_CATEGORIES = [
    "museums", "landmarks", "art_galleries", "performing_arts", "theater",
    "music_venues", "tours", "historical_sites", "zoos", "aquariums",
    "parks", "beaches", "hiking", "gardens", "shopping", "bookstores",
    "restaurants", "bars", "nightlife", "coffee", "breweries", "wineries",
    "cafes", "american", "italian", "mexican", "asian", "seafood",
    "festivals", "sporting_events", "active_life", "arcades"
]


# Sample attractions
ATTRACTION_TEMPLATES = [
    ("The {adj} Museum of {topic}", ["museums"], "Explore fascinating exhibits on {topic}"),
    ("{topic} Art Gallery", ["art_galleries"], "Contemporary and classical {topic} art"),
    ("{name} Theater", ["theater", "performing_arts"], "Historic venue for live performances"),
    ("{name} Park", ["parks"], "Beautiful green space perfect for relaxation"),
    ("{adj} {cuisine} Restaurant", ["restaurants"], "Authentic {cuisine} cuisine in cozy setting"),
    ("{name} Brewery", ["breweries", "bars"], "Craft beers and local favorites"),
    ("{topic} Historical Site", ["historical_sites", "landmarks"], "Learn about {topic} history"),
    ("{name} Zoo", ["zoos"], "Home to diverse animal species and conservation programs"),
    ("{name} Music Hall", ["music_venues"], "Live music and entertainment venue"),
    ("{adj} Bookstore", ["bookstores", "shopping"], "Independent bookstore with curated selection"),
]


ADJECTIVES = ["Grand", "Historic", "Modern", "Contemporary", "Classic", "Royal", "National", 
              "Public", "Urban", "Riverside", "Downtown", "Central"]
TOPICS = ["Science", "Natural History", "Aviation", "Maritime", "Local Heritage", 
          "Modern Art", "Photography", "Architecture"]
NAMES = ["Lincoln", "Washington", "Jefferson", "Roosevelt", "Madison", "Monroe", "Jackson",
         "Grant", "Wilson", "Kennedy", "Reagan"]
CUISINES = ["Italian", "Mexican", "Chinese", "Japanese", "Thai", "French", "Indian", 
            "American", "Mediterranean", "Korean"]


# Cities (contexts) similar to TREC
CONTEXTS = [
    ("Chicago", "IL", 41.8781, -87.6298),
    ("New York", "NY", 40.7128, -74.0060),
    ("San Francisco", "CA", 37.7749, -122.4194),
    ("Boston", "MA", 42.3601, -71.0589),
    ("Seattle", "WA", 47.6062, -122.3321),
    ("Austin", "TX", 30.2672, -97.7431),
    ("Portland", "OR", 45.5152, -122.6784),
    ("Denver", "CO", 39.7392, -104.9903),
]


class SyntheticDataGenerator:
    """Generate synthetic TREC-like data"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_poi(self, poi_id: str, location: str) -> PointOfInterest:
        """Generate a random point of interest"""
        template, categories, desc_template = random.choice(ATTRACTION_TEMPLATES)
        
        title = template.format(
            adj=random.choice(ADJECTIVES),
            topic=random.choice(TOPICS),
            name=random.choice(NAMES),
            cuisine=random.choice(CUISINES)
        )
        
        description = desc_template.format(
            topic=random.choice(TOPICS).lower(),
            cuisine=random.choice(CUISINES)
        )
        
        return PointOfInterest(
            poi_id=poi_id,
            title=title,
            description=description,
            url=f"http://example.com/poi/{poi_id}",
            categories=categories,
            location=location
        )
    
    def generate_user_profile(self, user_id: str, num_examples: int = 80) -> UserProfile:
        """Generate user profile with rated examples"""
        profile = UserProfile(user_id=user_id)
        
        preferred_categories = random.sample(YELP_CATEGORIES, k=random.randint(3, 8))
        disliked_categories = random.sample(
            [c for c in YELP_CATEGORIES if c not in preferred_categories], 
            k=random.randint(2, 5)
        )
        
        for i in range(num_examples):
            poi = self.generate_poi(
                poi_id=f"{user_id}_poi_{i:03d}",
                location=random.choice(["Chicago, IL", "Santa Fe, NM"])
            )
            
            category_overlap = any(c in preferred_categories for c in poi.categories)
            category_dislike = any(c in disliked_categories for c in poi.categories)
            
            if category_overlap:
                rating = random.choice([Rating.INTERESTED, Rating.STRONGLY_INTERESTED])
            elif category_dislike:
                rating = random.choice([Rating.UNINTERESTED, Rating.STRONGLY_UNINTERESTED])
            else:
                rating = random.choice([Rating.NEUTRAL, Rating.INTERESTED, Rating.UNINTERESTED])
            
            profile.add_rating(poi, rating, timestamp=i)
        
        return profile
    
    def generate_contexts(self) -> List[Context]:
        """Generate contexts (locations for recommendations)"""
        contexts = []
        for i, (city, state, lat, lon) in enumerate(CONTEXTS):
            contexts.append(Context(
                context_id=f"context_{i:03d}",
                city=city,
                state=state,
                latitude=lat,
                longitude=lon
            ))
        return contexts
    
    def generate_candidates(self, context: Context, num_candidates: int = 100) -> List[Candidate]:
        """Generate candidate suggestions for a context"""
        candidates = []
        
        for i in range(num_candidates):
            template, categories, _ = random.choice(ATTRACTION_TEMPLATES)
            name = template.format(
                adj=random.choice(ADJECTIVES),
                topic=random.choice(TOPICS),
                name=random.choice(NAMES),
                cuisine=random.choice(CUISINES)
            )
            
            lat_offset = random.uniform(-0.05, 0.05)
            lon_offset = random.uniform(-0.05, 0.05)
            lat = context.latitude + lat_offset
            lon = context.longitude + lon_offset
            
            distance = np.sqrt(lat_offset**2 + lon_offset**2) * 111000
            
            candidates.append(Candidate(
                candidate_id=f"{context.context_id}_cand_{i:03d}",
                name=name,
                categories=categories,
                snippet=f"Great {categories[0]} in {context.city}. " + 
                        random.choice(["Highly recommended!", "A must-visit!", "Popular spot!"]),
                url=f"http://yelp.com/{context.context_id}_cand_{i:03d}",
                distance_meters=distance,
                yelp_rating=random.uniform(2.5, 5.0),
                review_count=random.randint(10, 500),
                latitude=lat,
                longitude=lon
            ))
        
        return candidates
    
    def generate_dataset(self, num_users: int = 30, output_dir: str = "data/synthetic"):
        """Generate complete dataset and save to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {num_users} user profiles...")
        profiles = []
        for i in range(num_users):
            user_id = f"user_{i:04d}"
            profile = self.generate_user_profile(user_id)
            profiles.append(profile)
            self._save_profile(profile, output_path / f"profiles/{user_id}.json")
        
        print(f"Generating contexts...")
        contexts = self.generate_contexts()
        
        print(f"Generating candidates for each context...")
        for context in contexts:
            candidates = self.generate_candidates(context)
            self._save_candidates(candidates, output_path / f"candidates/{context.context_id}.json")
        
        metadata = {
            "num_users": num_users,
            "num_contexts": len(contexts),
            "categories": YELP_CATEGORIES
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset generated at {output_path}")
        return profiles, contexts
    
    def _save_profile(self, profile: UserProfile, path: Path):
        """Save user profile to JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "user_id": profile.user_id,
            "rating_events": [
                {
                    "poi": {
                        "poi_id": e.poi.poi_id,
                        "title": e.poi.title,
                        "description": e.poi.description,
                        "url": e.poi.url,
                        "categories": e.poi.categories,
                        "location": e.poi.location
                    },
                    "rating": e.rating.value,
                    "timestamp": e.timestamp
                }
                for e in profile.rating_events
            ]
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _save_candidates(self, candidates: List[Candidate], path: Path):
        """Save candidates to JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [
            {
                "candidate_id": c.candidate_id,
                "name": c.name,
                "categories": c.categories,
                "snippet": c.snippet,
                "url": c.url,
                "distance_meters": c.distance_meters,
                "yelp_rating": c.yelp_rating,
                "review_count": c.review_count,
                "latitude": c.latitude,
                "longitude": c.longitude
            }
            for c in candidates
        ]
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

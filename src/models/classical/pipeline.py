"""
Complete RAMA recommendation pipeline
Ties together all components
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import UserProfile, Context, Candidate, ScoredCandidate, Rating, PointOfInterest, RatingEvent
from models.classical.rama import RAMA
from models.classical.user_model import RAMAUserModel
from models.classical.scoring import RAMAScorer
from utils.text_processing import TermExtractor, CategoryExtractor


class RAMAPipeline:
    """
    Complete RAMA recommendation pipeline
    Matches architecture from Figure 1 of paper
    """
    
    def __init__(
        self,
        reinforcement_factor: float = 0.5,
        attenuation_factor: float = 0.0,
        use_pos_tagging: bool = False,  # Faster without POS
    ):
        self.rama = RAMA(reinforcement_factor, attenuation_factor)
        self.scorer = RAMAScorer()
        self.term_extractor = TermExtractor(use_pos_tagging)
        self.category_extractor = CategoryExtractor()
        
        # Cache for user models
        self.user_models: Dict[str, RAMAUserModel] = {}
    
    def build_user_model(self, profile: UserProfile) -> RAMAUserModel:
        """
        Step 1: User Interest Modeling (Section 2.1)
        Build general and specific interest models from profile
        """
        if profile.user_id in self.user_models:
            return self.user_models[profile.user_id]
        
        model = self.rama.build_user_model(
            user_id=profile.user_id,
            rating_events=profile.rating_events,
            extract_categories_fn=self.category_extractor.extract_from_poi,
            extract_terms_fn=self.term_extractor.extract_from_poi
        )
        
        self.user_models[profile.user_id] = model
        return model
    
    def score_candidate(
        self,
        user_model: RAMAUserModel,
        candidate: Candidate,
        weighting_scheme: str = "RUN2"  # RUN1 or RUN2 from Table 7
    ) -> ScoredCandidate:
        """
        Steps 3-4: Component Scoring and Aggregation
        """
        # Weighting schemes from Table 7
        weights = {
            "RUN1": {"general": 0.09, "specific": 0.9, "context": 0.01},
            "RUN2": {"general": 0.9, "specific": 0.09, "context": 0.01},
        }
        
        w = weights[weighting_scheme]
        
        scores = self.scorer.score_candidate(
            user_model=user_model,
            candidate=candidate,
            general_weight=w["general"],
            specific_weight=w["specific"],
            context_weight=w["context"]
        )
        
        return ScoredCandidate(
            candidate=candidate,
            general_interest_score=scores["general_score"],
            specific_interest_score=scores["specific_score"],
            context_score=scores["context_score"],
            final_score=scores["final_score"]
        )
    
    def rank_candidates(
        self,
        scored_candidates: List[ScoredCandidate]
    ) -> List[ScoredCandidate]:
        """
        Step 5: Suggestion Ranking (Section 2.5)
        Sort by final score and assign ranks
        """
        # Sort by final score (descending)
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x.final_score,
            reverse=True
        )
        
        # Assign ranks
        for rank, candidate in enumerate(sorted_candidates, start=1):
            candidate.rank = rank
        
        return sorted_candidates
    
    def get_recommendations(
        self,
        profile: UserProfile,
        candidates: List[Candidate],
        context: Context,
        top_k: int = 50,
        weighting_scheme: str = "RUN2"
    ) -> List[ScoredCandidate]:
        """
        Complete recommendation pipeline for user-context pair
        """
        # Step 1: Build user model
        user_model = self.build_user_model(profile)
        
        # Step 3-4: Score all candidates
        scored_candidates = [
            self.score_candidate(user_model, candidate, weighting_scheme)
            for candidate in candidates
        ]
        
        # Step 5: Rank
        ranked_candidates = self.rank_candidates(scored_candidates)
        
        # Return top K
        return ranked_candidates[:top_k]
    
    def evaluate_user_context_pair(
        self,
        profile: UserProfile,
        candidates: List[Candidate],
        context: Context,
        weighting_scheme: str = "RUN2"
    ) -> Dict:
        """Get recommendations and return evaluation data"""
        recommendations = self.get_recommendations(
            profile, candidates, context, top_k=50, weighting_scheme=weighting_scheme
        )
        
        return {
            "user_id": profile.user_id,
            "context_id": context.context_id,
            "recommendations": recommendations,
            "num_candidates": len(candidates),
            "weighting_scheme": weighting_scheme
        }
    
    def print_user_model_summary(self, user_model: RAMAUserModel, top_k: int = 10):
        """Print user model summary (like Table 1 & 2 in paper)"""
        print(f"\n{'='*60}")
        print(f"User Model: {user_model.user_id}")
        print(f"{'='*60}")
        
        print(f"\nGeneral Interests (Categories):")
        general_weights = user_model.general_facet.get_all_weights()
        sorted_general = sorted(general_weights.items(), key=lambda x: x[1], reverse=True)
        for elem, weight in sorted_general[:top_k]:
            print(f"  {elem:20s} {weight:+.3f}")
        
        print(f"\nSpecific Interests (Terms):")
        specific_weights = user_model.specific_facet.get_all_weights()
        sorted_specific = sorted(specific_weights.items(), key=lambda x: x[1], reverse=True)
        for elem, weight in sorted_specific[:top_k]:
            print(f"  {elem:20s} {weight:+.3f}")
    
    def print_recommendations(self, recommendations: List[ScoredCandidate], top_k: int = 10):
        """Print top recommendations"""
        size = 80 + 25
        print(f"\n{'='*size}")
        print(f"Top {top_k} Recommendations")
        print(f"{'='*size}")
        print(f"{'Rank':<6} {'ID':<25} {'Name':<30} {'General':<10} {'Specific':<10} {'Context':<10} {'Final':<10}")
        print(f"{'-'*size}")
        
        for rec in recommendations[:top_k]:
            print(
                f"{rec.rank:<6} "
                f"{rec.candidate.candidate_id:<25} "
                f"{rec.candidate.name[:28]:<30} "
                f"{rec.general_interest_score:+.3f}     "
                f"{rec.specific_interest_score:+.3f}     "
                f"{rec.context_score:+.3f}     "
                f"{rec.final_score:+.3f}"
            )


def load_synthetic_data(data_dir: str = "data/synthetic"):
    """Load synthetic data from disk"""
    data_path = Path(data_dir)
    
    # Load user profiles
    profiles = []
    profile_dir = data_path / "profiles"
    for profile_file in sorted(profile_dir.glob("*.json")):
        with open(profile_file) as f:
            data = json.load(f)
        
        profile = UserProfile(user_id=data["user_id"])
        for event_data in data["rating_events"]:
            poi_data = event_data["poi"]
            poi = PointOfInterest(
                poi_id=poi_data["poi_id"],
                title=poi_data["title"],
                description=poi_data["description"],
                url=poi_data["url"],
                categories=poi_data["categories"],
                location=poi_data.get("location")
            )
            rating = Rating(event_data["rating"])
            profile.add_rating(poi, rating, event_data.get("timestamp"))
        
        profiles.append(profile)
    
    # Load candidates for first context (for testing)
    candidate_files = sorted((data_path / "candidates").glob("*.json"))
    if candidate_files:
        with open(candidate_files[0]) as f:
            cand_data = json.load(f)
        
        candidates = [
            Candidate(
                candidate_id=c["candidate_id"],
                name=c["name"],
                categories=c["categories"],
                snippet=c["snippet"],
                url=c["url"],
                distance_meters=c["distance_meters"],
                yelp_rating=c["yelp_rating"],
                review_count=c["review_count"],
                latitude=c["latitude"],
                longitude=c["longitude"]
            )
            for c in cand_data
        ]
    else:
        candidates = []
    
    # Create context
    context = Context(
        context_id="context_000",
        city="Chicago",
        state="IL",
        latitude=41.8781,
        longitude=-87.6298
    )
    
    return profiles, candidates, context


# Demo
if __name__ == "__main__":
    print("Loading synthetic data...")
    
    # First, generate data if not exists
    from data.synthetic_generator import SyntheticDataGenerator
    
    data_dir = Path("data/synthetic")
    if not data_dir.exists():
        print("Generating synthetic data...")
        generator = SyntheticDataGenerator(seed=42)
        generator.generate_dataset(num_users=10)
    
    # Load data
    profiles, candidates, context = load_synthetic_data()
    
    print(f"Loaded {len(profiles)} profiles and {len(candidates)} candidates")
    
    # Create pipeline
    pipeline = RAMAPipeline()
    
    # Get recommendations for first user
    if profiles and candidates:
        profile = profiles[0]
        
        print(f"\nBuilding user model for {profile.user_id}...")
        user_model = pipeline.build_user_model(profile)
        
        # Print user model
        pipeline.print_user_model_summary(user_model, top_k=15)
        
        # Get recommendations
        print(f"\nGenerating recommendations for {context.city}, {context.state}...")
        
        # Try both weighting schemes
        for scheme in ["RUN1", "RUN2"]:
            print(f"\n\n{'='*80}")
            print(f"Weighting Scheme: {scheme}")
            print(f"{'='*80}")
            
            recommendations = pipeline.get_recommendations(
                profile, candidates, context, top_k=50, weighting_scheme=scheme
            )
            
            pipeline.print_recommendations(recommendations, top_k=10)
        
        print("\n\nRAMA Pipeline Demo Complete!")
#!/usr/bin/env python3
"""
Quick start script for Neural RAMA project
Run this to see classical RAMA in action
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.synthetic_generator import SyntheticDataGenerator
from models.classical.pipeline import RAMAPipeline, load_synthetic_data


def main():
    """Run complete demo of classical RAMA"""
    
    print("="*80)
    print("NEURAL RAMA PROJECT - Classical RAMA Demo")
    print("="*80)
    print()
    
    # Step 1: Generate synthetic data
    print("[1/5] Generating synthetic TREC-format data...")
    data_dir = Path("data/synthetic")
    
    # Check if data files actually exist
    profile_files = list(data_dir.glob("profiles/*.json")) if data_dir.exists() else []
    candidate_files = list(data_dir.glob("candidates/*.json")) if data_dir.exists() else []
    
    if profile_files and candidate_files:
        print("   ✓ Using existing synthetic data")
    else:
        print("   Generating new synthetic data...")
        generator = SyntheticDataGenerator(seed=42)
        profiles, contexts = generator.generate_dataset(num_users=10)
        print(f"   ✓ Generated {len(profiles)} user profiles")
        print(f"   ✓ Generated {len(contexts)} contexts")
    
    # Step 2: Load data
    print("\n[2/5] Loading data...")
    profiles, candidates, context = load_synthetic_data()
    print(f"   ✓ Loaded {len(profiles)} profiles")
    print(f"   ✓ Loaded {len(candidates)} candidates for {context.city}")
    
    # Step 3: Initialize RAMA pipeline
    print("\n[3/5] Initializing RAMA pipeline...")
    pipeline = RAMAPipeline(
        reinforcement_factor=0.5,  # From paper
        attenuation_factor=0.0,     # From paper (TREC setting)
        use_pos_tagging=False       # Faster for demo
    )
    print("   ✓ Pipeline initialized with paper parameters")
    
    # Step 4: Build user models
    print("\n[4/5] Building user interest models...")
    profile = profiles[0]  # Use first user
    print(f"   Processing user: {profile.user_id}")
    print(f"   Rating events: {profile.num_events}")
    print(f"   Positive ratings: {len(profile.positive_events)}")
    print(f"   Negative ratings: {len(profile.negative_events)}")
    
    user_model = pipeline.build_user_model(profile)
    print(f"   ✓ Built user model:")
    print(f"     - General interests: {len(user_model.general_facet)} categories")
    print(f"     - Specific interests: {len(user_model.specific_facet)} terms")
    
    # Show user model
    pipeline.print_user_model_summary(user_model, top_k=10)
    
    # Step 5: Generate recommendations
    print("\n[5/5] Generating recommendations...")
    print(f"   Context: {context.city}, {context.state}")
    print(f"   Candidates: {len(candidates)}")
    
    # Compare both weighting schemes from paper (Table 7)
    schemes = {
        "RUN1": "Specific Interest Priority (Ws=0.9)",
        "RUN2": "General Interest Priority (Wg=0.9)"
    }
    
    for scheme_name, description in schemes.items():
        print(f"\n   Weighting Scheme: {scheme_name} - {description}")
        
        recommendations = pipeline.get_recommendations(
            profile=profile,
            candidates=candidates,
            context=context,
            top_k=50,
            weighting_scheme=scheme_name
        )
        
        print(f"   ✓ Generated {len(recommendations)} ranked recommendations")
        
        # Show top 5
        pipeline.print_recommendations(recommendations, top_k=5)
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Explore user models in notebooks/01_data_exploration.ipynb")
    print("2. Run tests: pytest tests/")
    print("3. Start neural translation: src/models/neural/")
    print("="*80)
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

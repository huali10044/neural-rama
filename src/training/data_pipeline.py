"""
Training Data Pipeline for Neural RAMA (TensorFlow)

Loads synthetic TREC data and creates tf.data.Dataset batches for
pairwise BPR training:
  - Each sample: (user_features, positive_candidate, negative_candidate)
  - User features: category IDs, term embeddings, and rating polarities
    derived from the user's rating history
  - Positive candidate: a candidate aligned with user preferences
  - Negative candidate: a randomly sampled or low-rated candidate

Phase 4 — Training Infrastructure

TensorFlow Version: 2.15+
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from data.data_loader import (
    UserProfile,
    PointOfInterest,
    RatingEvent,
    Rating,
    Candidate,
    Context,
)


# Rating → event relevance mapping (Table 6 from RAMA paper)
RATING_TO_RELEVANCE = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}


@dataclass
class ProcessedUser:
    """Pre-processed user data ready for tensor conversion."""

    user_id: str
    # Per rated-item arrays (length = num_rating_events)
    category_ids: List[List[int]]  # (num_items, cats_per_item)
    term_embeddings: np.ndarray    # (num_items, term_dim) — pre-computed
    rating_weights: List[float]    # (num_items,) in [-1, +1]


@dataclass
class ProcessedCandidate:
    """Pre-processed candidate data ready for tensor conversion."""

    candidate_id: str
    category_ids: List[int]     # (cats_per_item,)
    term_embedding: np.ndarray  # (term_dim,) — pre-computed
    distance: float
    yelp_rating: float
    review_count: int


class RAMADataPipeline:
    """
    Loads synthetic data, pre-computes embeddings, and produces
    tf.data.Dataset instances for BPR training.

    Usage:
        pipeline = RAMADataPipeline(data_dir="data/synthetic")
        pipeline.prepare(term_embedder)   # pre-compute term embeddings
        train_ds, val_ds = pipeline.create_datasets(batch_size=32)

    Args:
        data_dir: Path to synthetic data directory
        max_history: Maximum number of rating events per user
        max_cats_per_item: Maximum categories per POI/candidate (padded)
        val_fraction: Fraction of users held out for validation
        seed: Random seed
    """

    def __init__(
        self,
        data_dir: str = "data/synthetic",
        max_history: int = 30,
        max_cats_per_item: int = 5,
        val_fraction: float = 0.2,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.max_history = max_history
        self.max_cats_per_item = max_cats_per_item
        self.val_fraction = val_fraction
        self.seed = seed

        # Built during prepare()
        self.category_to_id: Dict[str, int] = {}
        self.users: List[ProcessedUser] = []
        self.candidates_by_context: Dict[str, List[ProcessedCandidate]] = {}
        self.all_candidates: List[ProcessedCandidate] = []
        self.term_dim: int = 0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_metadata(self) -> dict:
        with open(self.data_dir / "metadata.json") as f:
            return json.load(f)

    def _load_user_profiles(self) -> List[dict]:
        profiles_dir = self.data_dir / "profiles"
        profiles = []
        for p in sorted(profiles_dir.glob("user_*.json")):
            with open(p) as f:
                profiles.append(json.load(f))
        return profiles

    def _load_candidates(self) -> Dict[str, List[dict]]:
        cands_dir = self.data_dir / "candidates"
        candidates = {}
        for p in sorted(cands_dir.glob("context_*.json")):
            ctx_id = p.stem  # e.g. "context_000"
            with open(p) as f:
                candidates[ctx_id] = json.load(f)
        return candidates

    def _build_category_vocab(self, metadata: dict) -> None:
        """Build category → int ID mapping (0=PAD, 1=UNK, 2+=categories)."""
        self.category_to_id = {"<PAD>": 0, "<UNK>": 1}
        for i, cat in enumerate(metadata["categories"]):
            self.category_to_id[cat] = i + 2

    def _encode_categories(self, categories: List[str]) -> List[int]:
        """Encode a list of category strings to int IDs, pad/truncate."""
        ids = [
            self.category_to_id.get(c, self.category_to_id["<UNK>"])
            for c in categories
        ]
        # Truncate or pad
        ids = ids[: self.max_cats_per_item]
        ids += [0] * (self.max_cats_per_item - len(ids))
        return ids

    # ------------------------------------------------------------------
    # Embedding pre-computation
    # ------------------------------------------------------------------

    def _encode_all_texts(self, term_embedder, texts: List[str]) -> np.ndarray:
        """Pre-compute sentence-transformer embeddings for all texts."""
        if hasattr(term_embedder, "encode_texts"):
            embs = term_embedder.encode_texts(texts)
            return embs.numpy()
        elif hasattr(term_embedder, "encode"):
            return term_embedder.encode(texts, show_progress_bar=False)
        else:
            raise ValueError("term_embedder must have encode_texts() or encode() method")

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare(self, term_embedder) -> None:
        """
        Load data, build vocabulary, pre-compute term embeddings.

        Args:
            term_embedder: A TermEmbedding instance or SentenceTransformer
                           with an encode_texts() or encode() method.
        """
        metadata = self._load_metadata()
        self._build_category_vocab(metadata)
        raw_profiles = self._load_user_profiles()
        raw_candidates = self._load_candidates()

        # Collect ALL texts for batch encoding
        all_texts = []
        text_index = {}  # text → index into all_texts

        def _register_text(text: str) -> int:
            if text not in text_index:
                text_index[text] = len(all_texts)
                all_texts.append(text)
            return text_index[text]

        # Register POI texts from user profiles
        user_text_indices = []
        for profile in raw_profiles:
            indices = []
            for event in profile["rating_events"]:
                poi = event["poi"]
                text = f"{poi['title']}. {poi['description']}"
                indices.append(_register_text(text))
            user_text_indices.append(indices)

        # Register candidate texts
        candidate_text_indices: Dict[str, List[int]] = {}
        for ctx_id, cands in raw_candidates.items():
            indices = []
            for c in cands:
                text = f"{c['name']}. {c['snippet']}"
                indices.append(_register_text(text))
            candidate_text_indices[ctx_id] = indices

        # Batch-encode all texts at once
        all_embeddings = self._encode_all_texts(term_embedder, all_texts)
        self.term_dim = all_embeddings.shape[1]

        # Process users
        self.users = []
        for profile, txt_indices in zip(raw_profiles, user_text_indices):
            events = profile["rating_events"]
            n = min(len(events), self.max_history)

            cat_ids = []
            term_embs = []
            ratings = []
            for i in range(n):
                event = events[i]
                cat_ids.append(self._encode_categories(event["poi"]["categories"]))
                term_embs.append(all_embeddings[txt_indices[i]])
                ratings.append(RATING_TO_RELEVANCE[event["rating"]])

            self.users.append(ProcessedUser(
                user_id=profile["user_id"],
                category_ids=cat_ids,
                term_embeddings=np.array(term_embs, dtype=np.float32),
                rating_weights=ratings,
            ))

        # Process candidates
        self.candidates_by_context = {}
        self.all_candidates = []
        for ctx_id, cands in raw_candidates.items():
            processed = []
            for c, txt_idx in zip(cands, candidate_text_indices[ctx_id]):
                pc = ProcessedCandidate(
                    candidate_id=c["candidate_id"],
                    category_ids=self._encode_categories(c["categories"]),
                    term_embedding=all_embeddings[txt_idx],
                    distance=c["distance_meters"],
                    yelp_rating=c["yelp_rating"],
                    review_count=c["review_count"],
                )
                processed.append(pc)
                self.all_candidates.append(pc)
            self.candidates_by_context[ctx_id] = processed

    # ------------------------------------------------------------------
    # BPR triple generation
    # ------------------------------------------------------------------

    def _generate_triples(
        self, users: List[ProcessedUser], num_negatives: int = 1
    ) -> List[dict]:
        """
        Generate (user, positive_candidate, negative_candidate) triples.

        Strategy:
        - For each user, iterate over their positively-rated POIs as
          positive examples.
        - Sample negative candidates randomly from the candidate pool.
        - User features come from their full rating history.
        """
        rng = random.Random(self.seed)
        triples = []

        for user in users:
            n_items = len(user.rating_weights)
            if n_items == 0:
                continue

            # Build padded user feature tensors
            pad_n = self.max_history - n_items
            user_cat_ids = np.array(user.category_ids, dtype=np.int32)
            if pad_n > 0:
                user_cat_ids = np.pad(
                    user_cat_ids,
                    ((0, pad_n), (0, 0)),
                    constant_values=0,
                )
            else:
                user_cat_ids = user_cat_ids[: self.max_history]

            user_term_embs = user.term_embeddings
            if pad_n > 0:
                user_term_embs = np.pad(
                    user_term_embs,
                    ((0, pad_n), (0, 0)),
                    constant_values=0.0,
                )
            else:
                user_term_embs = user_term_embs[: self.max_history]

            user_ratings = np.array(
                user.rating_weights[: self.max_history], dtype=np.float32
            )
            if pad_n > 0:
                user_ratings = np.pad(user_ratings, (0, pad_n), constant_values=0.0)

            num_valid = min(n_items, self.max_history)

            # Create positive/negative pairs from candidates
            # Positive: candidates with categories overlapping user's
            # liked categories; Negative: random sample
            positive_indices = [
                i for i, r in enumerate(user.rating_weights[:num_valid]) if r > 0
            ]
            if not positive_indices:
                continue

            # Determine the categories the user has rated positively, so
            # positive/negative candidates can be sampled in a way that
            # actually reflects the user's preferences (rather than being
            # drawn uniformly at random, which carries no learning signal).
            liked_cat_ids = set()
            for i in positive_indices:
                for cid in user.category_ids[i]:
                    if cid >= 2:  # skip <PAD>=0 and <UNK>=1
                        liked_cat_ids.add(cid)

            pos_pool = [
                c for c in self.all_candidates
                if any(cid in liked_cat_ids for cid in c.category_ids if cid >= 2)
            ]
            neg_pool = [
                c for c in self.all_candidates
                if not any(cid in liked_cat_ids for cid in c.category_ids if cid >= 2)
            ]
            # Fall back to the full pool if either side is empty (e.g. no
            # candidate happens to share a liked category)
            if not pos_pool:
                pos_pool = self.all_candidates
            if not neg_pool:
                neg_pool = self.all_candidates

            for pos_idx in positive_indices:
                # Positive candidate: shares a category the user likes
                pos_cand = rng.choice(pos_pool)

                for _ in range(num_negatives):
                    # Negative candidate: does not share a liked category
                    neg_cand = rng.choice(neg_pool)
                    # Ensure different candidate
                    while neg_cand.candidate_id == pos_cand.candidate_id:
                        neg_cand = rng.choice(neg_pool)

                    triples.append({
                        # User features
                        "user_category_ids": user_cat_ids,
                        "user_term_embeddings": user_term_embs,
                        "user_rating_weights": user_ratings,
                        "user_num_items": num_valid,
                        # Positive candidate
                        "pos_category_ids": np.array(pos_cand.category_ids, dtype=np.int32),
                        "pos_term_embedding": pos_cand.term_embedding,
                        "pos_distance": np.float32(pos_cand.distance),
                        "pos_yelp_rating": np.float32(pos_cand.yelp_rating),
                        "pos_review_count": np.float32(pos_cand.review_count),
                        # Negative candidate
                        "neg_category_ids": np.array(neg_cand.category_ids, dtype=np.int32),
                        "neg_term_embedding": neg_cand.term_embedding,
                        "neg_distance": np.float32(neg_cand.distance),
                        "neg_yelp_rating": np.float32(neg_cand.yelp_rating),
                        "neg_review_count": np.float32(neg_cand.review_count),
                    })

        rng.shuffle(triples)
        return triples

    def _triples_to_dataset(
        self, triples: List[dict], batch_size: int, shuffle: bool = True
    ) -> tf.data.Dataset:
        """Convert list of triple dicts into a batched tf.data.Dataset."""
        if not triples:
            raise ValueError("No training triples generated. Check data.")

        # Stack all fields into arrays
        data = {}
        for key in triples[0]:
            data[key] = np.array([t[key] for t in triples])

        ds = tf.data.Dataset.from_tensor_slices(data)
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(triples), 10000), seed=self.seed)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_datasets(
        self,
        batch_size: int = 32,
        num_negatives: int = 1,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create training and validation tf.data.Datasets.

        Splits users (not triples) into train/val to avoid data leakage.

        Args:
            batch_size: Training batch size
            num_negatives: Number of negative samples per positive

        Returns:
            (train_dataset, val_dataset) — each yielding batched dicts
        """
        rng = random.Random(self.seed)
        indices = list(range(len(self.users)))
        rng.shuffle(indices)

        val_size = max(1, int(len(self.users) * self.val_fraction))
        val_indices = set(indices[:val_size])
        train_users = [u for i, u in enumerate(self.users) if i not in val_indices]
        val_users = [u for i, u in enumerate(self.users) if i in val_indices]

        train_triples = self._generate_triples(train_users, num_negatives)
        val_triples = self._generate_triples(val_users, num_negatives)

        train_ds = self._triples_to_dataset(train_triples, batch_size, shuffle=True)
        val_ds = self._triples_to_dataset(val_triples, batch_size, shuffle=False)

        return train_ds, val_ds

    def get_candidate_features(
        self, context_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Get all candidate features for a context (for evaluation).

        Returns dict with:
            category_ids:   (num_candidates, max_cats_per_item)
            term_embeddings: (num_candidates, term_dim)
            distances:       (num_candidates,)
            yelp_ratings:    (num_candidates,)
            review_counts:   (num_candidates,)
        """
        cands = self.candidates_by_context[context_id]
        return {
            "category_ids": np.array([c.category_ids for c in cands], dtype=np.int32),
            "term_embeddings": np.array([c.term_embedding for c in cands], dtype=np.float32),
            "distances": np.array([c.distance for c in cands], dtype=np.float32),
            "yelp_ratings": np.array([c.yelp_rating for c in cands], dtype=np.float32),
            "review_counts": np.array([c.review_count for c in cands], dtype=np.float32),
        }

    @property
    def num_categories(self) -> int:
        """Number of categories in vocabulary (excluding PAD and UNK)."""
        return len(self.category_to_id) - 2

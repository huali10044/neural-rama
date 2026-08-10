# Neural RAMA: Classical to Neural Recommender System

Faithful translation of the RAMA (Reinforcement and Aging Modeling Algorithm) contextual recommendation system from classical implementation to modern neural architecture using TensorFlow/Keras.

Based on: **"User Modeling for Contextual Suggestion"** by Hua Li & Rafael Alonso (TREC 2014)

## Project Goal

Translate RAMA's classical user modeling approach into a production-grade neural recommender system while **preserving** its core modeling assumptions:
- Separate general (category) vs. specific (term) interests
- Weight range: [-1, +1] representing hate-to-love spectrum
- Explicit reinforcement and decay mechanisms
- Multi-component scoring (general, specific, context)

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Classical RAMA implementation | Complete |
| Phase 2 | Neural embedding layers (TF/Keras) | Complete |
| Phase 3 | User interest encoders + reinforcement + scoring | Complete |
| Phase 4 | Training infrastructure | Complete |
| Phase 5 | Training & validation | Planned |
| Phase 6 | Production engineering (Django API) | Planned |
| Phase 7 | Testing & documentation | Planned |
| Phase 8 | Validation & iteration | Planned |
| Phase 9 | Deployment (Local Docker / AWS EC2 / SageMaker) | Planned |

## Project Structure

```
neural-rama/
├── run_demo.py                           # End-to-end classical RAMA demo (RUN1 vs RUN2)
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
│
├── src/
│   ├── data/
│   │   ├── data_loader.py                # Core data structures (UserProfile, Candidate, Context)
│   │   └── synthetic_generator.py        # TREC-format synthetic data (30 users, 8 contexts)
│   ├── models/
│   │   ├── classical/                    # Phase 1 — complete
│   │   │   ├── rama.py                   # Core RAMA algorithm (Formulas 3, 4, 5)
│   │   │   ├── user_model.py             # General and specific interest facets
│   │   │   ├── scoring.py                # Cosine similarity component scoring
│   │   │   └── pipeline.py               # End-to-end recommendation pipeline
│   │   └── neural/                       # Phases 2–3 — complete
│   │       ├── embeddings.py             # CategoryEmb, TermEmb, ContextEmb, WeightConstraint
│   │       ├── encoders.py               # AttentionAggregation, GeneralEncoder, SpecificEncoder
│   │       ├── reinforcement.py          # NeuralReinforcement, UserStateUpdater
│   │       └── scoring.py                # ComponentScorer, ScoreFusion, NeuralRAMAScorer
│   ├── training/                         # Phase 4 — complete
│   │   ├── data_pipeline.py              # RAMADataPipeline — BPR triple generation
│   │   ├── losses.py                     # BPRLoss, RAMALoss (ranking + regularization)
│   │   ├── metrics.py                    # P@5, MRR, TBG (TREC 2014 targets)
│   │   └── trainer.py                    # RAMATrainer (GradientTape loop, checkpointing, early stopping)
│   ├── serving/                          # Phase 6 — planned
│   └── utils/
│       └── text_processing.py            # NLP utilities (term/category extraction)
│
├── tests/
│   ├── test_classical_rama.py            # Phase 1 classical RAMA tests
│   ├── test_embeddings.py                # Phase 2 embedding layer tests (8 tests)
│   ├── test_encoders.py                  # Phase 3 encoder/reinforcement/scoring tests (8 tests)
│   └── test_training.py                  # Phase 4 training infrastructure tests
│
├── configs/
│   └── rama_config.yaml                  # RAMA hyperparameters and weighting schemes
│
├── data/                                 # Generated synthetic data (git-ignored)
└── notebooks/                            # Analysis notebooks
```

Internal design notes, progress reports, and planning docs live in `docs/` locally but are not committed to this repo (see `.gitignore`).

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/hualinyc/neural-rama.git
cd neural-rama

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"
```

### 2. Run Classical RAMA Demo

```bash
python run_demo.py
```

This will:
1. Generate synthetic TREC-format data (30 users, 8 contexts)
2. Build user interest models using RAMA
3. Generate recommendations for sample user-context pairs
4. Compare RUN1 (specific priority) vs RUN2 (general priority) weighting schemes

### 3. Run Neural Tests

```bash
# Phase 2: embedding layer tests
pytest tests/test_embeddings.py -v

# Phase 3: encoder, reinforcement, and scoring tests
pytest tests/test_encoders.py -v

# Phase 4: training infrastructure tests
pytest tests/test_training.py -v

# All neural tests
pytest tests/test_embeddings.py tests/test_encoders.py tests/test_training.py -v
```

## Classical RAMA Implementation (Phase 1)

### Core Formulas

**Formula 3 — Aging/Decay:**
```python
newWeight = oldWeight * (1 - attenuationFactor)
```

**Formula 4 — Reinforcement:**
```python
newWeight = oldWeight + (eventRelevance * reinforcementFactor *
            mentionFrequency * (1 - abs(oldWeight)))
```

**Formula 5 — New Element Insertion:**
```python
initialWeight = eventRelevance * reinforcementFactor * mentionFrequency
```

**Formula 2 — Final Ranking:**
```python
FinalScore = Wg * GeneralScore + Ws * SpecificScore + Wc * ContextScore
```

Weighting schemes:
- **RUN1** (specific priority): Wg=0.09, Ws=0.9, Wc=0.01
- **RUN2** (general priority): Wg=0.9, Ws=0.09, Wc=0.01

### Example Usage

```python
from src.models.classical.pipeline import RAMAPipeline
from src.data.synthetic_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)
profiles, contexts = generator.generate_dataset(num_users=10)

pipeline = RAMAPipeline(
    reinforcement_factor=0.5,
    attenuation_factor=0.0
)

user_model = pipeline.build_user_model(profiles[0])

recommendations = pipeline.get_recommendations(
    profile=profiles[0],
    candidates=candidates,
    context=context,
    top_k=50,
    weighting_scheme="RUN2"
)
```

### TREC 2014 Results (from paper)

RAMARUN2 ranked #2 out of 31 submissions:
- P@5: 0.50 (vs Track Median: 0.35, +43%)
- MRR: 0.71 (vs Track Median: 0.57, +25%)
- TBG: 0.70 (vs Track Median: 0.40, +75%)

## Neural Implementation (Phases 2–3)

**Framework**: TensorFlow 2.15+ / Keras

### Architecture Overview

```
User Profile (rated examples)
         │
         ▼
┌────────────────────┐
│  Embedding Layers  │  ← Phase 2 (complete) — embeddings.py
│  - CategoryEmb     │
│  - TermEmb (ST)    │
│  - ContextEmb      │
└────────┬───────────┘
         │
┌────────▼───────────┐
│  Interest Encoders │  ← Phase 3 (complete) — encoders.py
│  - GeneralEncoder  │    AttentionAggregation over rated items
│  - SpecificEncoder │    weighted by rating polarity
└────────┬───────────┘
         │
┌────────▼───────────┐
│  Reinforcement     │  ← Phase 3 (complete) — reinforcement.py
│  NeuralReinforcement    Formulas 3 & 4 + attention delta + gate
│  UserStateUpdater  │    sequential event loop
└────────┬───────────┘
         │
┌────────▼───────────┐
│  Scoring & Fusion  │  ← Phase 3 (complete) — scoring.py
│  ComponentScorer   │    dot-product general/specific + MLP context
│  ScoreFusion       │    fixed (RUN1/RUN2) or learned weights
└────────┬───────────┘
         │
    Final Score
```

### Phase 2: Embedding Layers (`src/models/neural/embeddings.py`)

| Layer | Input | Output | Notes |
|-------|-------|--------|-------|
| `CategoryEmbedding` | category IDs `(B, N)` | embeddings `(B, N, 128)` | learned + L2 normalized |
| `TermEmbedding` | pre-encoded vectors `(B, 384)` | projected `(B, 128)` | sentence-transformers base |
| `ContextEmbedding` | distance, rating, count | context vector `(B, 64)` | MLP encoder |
| `WeightConstraint` | any tensor | constrained tensor | enforces [-1, +1] via tanh/clip/sigmoid |
| `RAMAEmbeddings` | dict of all inputs | dict of all outputs | integrated model |

Key design — sentence-transformers runs on CPU outside the TF graph; TF projection layers handle fine-tuning:

```python
# Step 1: encode text outside TF (no gradients)
term_embeddings = term_embedder.encode_texts(["museum", "park"])

# Step 2: project inside TF (with gradients)
outputs = term_embedder(term_embeddings, training=True)
```

### Phase 3: Interest Encoders (`src/models/neural/encoders.py`)

| Layer | Input | Output | RAMA Equivalent |
|-------|-------|--------|-----------------|
| `AttentionAggregation` | `(B, N, D)` item embs + `(B, N)` ratings | `(B, D)` | weighted aggregation |
| `GeneralInterestEncoder` | category IDs + ratings | `(B, D)` in [-1,+1] | general facet (category weight dict) |
| `SpecificInterestEncoder` | term embeddings + ratings | `(B, D)` in [-1,+1] | specific facet (term weight dict) |

`AttentionAggregation` uses a hybrid aggregation: a rating-weighted mean of `|item_embeddings|` (directional signal that reliably reflects rating sign from initialization) summed with a down-scaled MHA context term (keeps all projection weights in the gradient graph):

```python
# Directional signal: E[|r| * |embed[d]|] > 0 when r > 0
rating_weighted_mean = mean(|item_embeddings| * rating_weights, axis=items)

# Learned context via MHA (scaled down so directional signal dominates)
attended_mha = MHA(query=learned_query, key=item_embeddings, value=item_embeddings)

output = rating_weighted_mean + 0.1 * attended_mha   # then dropout → WeightConstraint(tanh)
```

### Phase 3: Neural Reinforcement (`src/models/neural/reinforcement.py`)

Differentiable translation of RAMA Formulas 3 & 4:

```python
# Formula 3: aging
aged_state = current_state * (1 - attenuation_factor)       # default: 0.0

# Delta: abs(l2_normalize(new_event)) — always >= 0 element-wise,
# so the sign of the update comes purely from event_relevance
delta = abs(l2_normalize(new_event_embedding))

# Formula 4: reinforcement magnitude
magnitude = event_relevance * reinforcement_factor * (1 - |aged_state|)  # default: 0.5

# Learned sigmoid gate + tanh constraint
new_state = tanh(aged_state + magnitude * delta * sigmoid_gate)  # in [-1, +1]
```

`UserStateUpdater` wraps `NeuralReinforcement` in a sequential loop over the full rating history, mirroring RAMA's event-by-event processing. Both `attenuation_factor` and `reinforcement_factor` are optionally trainable.

### Phase 3: Scoring (`src/models/neural/scoring.py`)

| Component | Description | RAMA Equivalent |
|-----------|-------------|-----------------|
| `ComponentScorer` | dot product (general/specific) + MLP→tanh (context); all in [-1,+1] | `RAMAScorer.score_candidate()` |
| `ScoreFusion` | `fixed` mode: exact RUN1/RUN2 paper weights; `learned` mode: softmax-normalized trainable weights | Formula 2 |
| `NeuralRAMAScorer` | integrates both into one `keras.Model` | `RAMAPipeline.score_candidate()` |

End-to-end example:

```python
import tensorflow as tf
from src.models.neural.scoring import NeuralRAMAScorer

scorer = NeuralRAMAScorer(
    embed_dim=128,
    fusion_mode="fixed",   # start with paper weights (RUN2)
    fusion_scheme="RUN2"
)

outputs = scorer({
    "user_general":           user_general_vec,    # from GeneralInterestEncoder
    "user_specific":          user_specific_vec,   # from SpecificInterestEncoder
    "candidate_category_emb": cand_cat_emb,
    "candidate_term_emb":     cand_term_emb,
    "distance":               tf.constant([1500.0]),
    "rating":                 tf.constant([4.5]),
    "review_count":           tf.constant([120.0]),
}, training=False)
# outputs: {'general': ..., 'specific': ..., 'context': ..., 'final': ...}
```

## Phase 4: Training Infrastructure

**Framework**: TensorFlow 2.15+ / Keras

### Data Pipeline (`src/training/data_pipeline.py`)

`RAMADataPipeline` converts a `SyntheticDataGenerator` dataset into BPR triples suitable for `tf.data.Dataset` consumption:
- Encodes category IDs and term embeddings per user
- Pairs each positively-rated candidate with a randomly sampled negative
- Pads history sequences to a fixed `max_history` length
- Splits into train / validation sets

### Loss Functions (`src/training/losses.py`)

| Class | Description |
|-------|-------------|
| `BPRLoss` | Standard BPR: `-mean(log(sigmoid(score_pos − score_neg)))`, with optional margin-hinge variant |
| `RAMALoss` | `L = L_BPR + λ_range · L_range + λ_balance · L_balance` — adds soft penalties to keep interest vectors well within [-1, +1] and prevent degenerate score fusion weights |

### Evaluation Metrics (`src/training/metrics.py`)

TREC 2014 targets (matching classical RAMARUN2 results):

| Metric | Target | Description |
|--------|--------|-------------|
| P@5 | ≥ 0.50 | Precision at rank 5 |
| MRR | ≥ 0.71 | Mean Reciprocal Rank |
| TBG | ≥ 0.70 | Time-Biased Gain (half-life = 224s) |

Helper functions: `precision_at_k`, `mean_reciprocal_rank`, `time_biased_gain`, `evaluate_ranking`, `evaluate_all_users`.

### Training Loop (`src/training/trainer.py`)

`RAMATrainer` manages the full training lifecycle:

```python
from src.models.neural.neural_rama import NeuralRAMA
from src.training.trainer import RAMATrainer

model = NeuralRAMA(embed_dim=128, fusion_mode="fixed", fusion_scheme="RUN2")
trainer = RAMATrainer(
    model,
    learning_rate=1e-3,
    lambda_range=0.01,
    lambda_balance=0.01,
    checkpoint_dir="checkpoints",
    patience=10,
)

history = trainer.train(train_ds, val_ds, num_epochs=100)
```

Features:
- `tf.GradientTape` loop with gradient clipping (global norm 5.0)
- Adam optimizer with configurable learning rate
- Automatic checkpointing on validation loss improvement (`*.weights.h5`)
- Early stopping with configurable patience
- Per-epoch logging: train/val loss, pairwise accuracy, learning rate
- Full ranking evaluation via `evaluate_ranking(users, pipeline, k=5)`

---

## Testing

```bash
# All tests
pytest tests/ -v

# Classical RAMA only
pytest tests/test_classical_rama.py -v

# Neural tests by phase
pytest tests/test_embeddings.py -v   # Phase 2: 8 tests
pytest tests/test_encoders.py -v     # Phase 3: 8 tests

# With coverage
pytest tests/ --cov=src --cov-report=html
```

**Phase 2 — `test_embeddings.py`** (8 tests):
1. Category vocabulary encoding/decoding
2. Category embedding shape and normalization
3. Term embedding semantic similarity
4. Context embedding feature encoding
5. Weight constraint range preservation (tanh/clip/sigmoid)
6. Weight constraint BPR loss differentiability
7. RAMAEmbeddings integration
8. TensorFlow features (gradients, serialization, mixed precision)

**Phase 3 — `test_encoders.py`** (8 tests):
1. `AttentionAggregation` — shape, rating sensitivity, gradient flow
2. `GeneralInterestEncoder` — shape, [-1,+1] range, positive > negative ratings
3. `SpecificInterestEncoder` — shape, [-1,+1] range, semantic similarity preserved
4. `NeuralReinforcement` — shape, [-1,+1] range, attenuation effect, BPR gradient
5. `UserStateUpdater` — shape, [-1,+1] range, positive history → positive state
6. `ComponentScorer` — all three scores, identical→+1, opposite→-1, gradient
7. `ScoreFusion` — fixed RUN1/RUN2 match Formula 2 exactly; learned weights train
8. `NeuralRAMAScorer` — end-to-end shapes, aligned > opposed ranking, BPR gradient

**Phase 4 — `test_training.py`**:
- `BPRLoss` — correct loss direction, margin variant, gradient flow
- `RAMALoss` — range + balance regularization, combined loss structure
- Metrics — P@K, MRR, TBG correctness on known rankings
- `RAMADataPipeline` — BPR triple generation, padding, train/val split
- `NeuralRAMA` — end-to-end BPR forward pass shapes and gradient flow
- `RAMATrainer` — single train step, checkpoint save/load

## Configuration

`configs/rama_config.yaml`:

```yaml
rama:
  reinforcement_factor: 0.5
  attenuation_factor: 0.0

scoring:
  distance_limit_meters: 8000
  review_count_limit: 100

weighting:
  run1:
    general: 0.09
    specific: 0.9
    context: 0.01
  run2:
    general: 0.9
    specific: 0.09
    context: 0.01
```

## Key Concepts

### Event Relevance Mapping

User ratings [0-4] → event relevance [-1, +1]:

| Rating | Meaning | Relevance |
|--------|---------|-----------|
| 0 | Strongly Uninterested | -1.0 |
| 1 | Uninterested | -0.5 |
| 2 | Neutral | 0.0 |
| 3 | Interested | +0.5 |
| 4 | Strongly Interested | +1.0 |

### Interest Facets

- **General**: High-level categories (e.g., "museums", "restaurants")
- **Specific**: Fine-grained terms (e.g., "art", "sculpture", "impressionism")

### Classical vs Neural Equivalences

| RAMA (Classical) | Neural Equivalent |
|------------------|-------------------|
| Interest weight scalar | Embedding vector dimension |
| Cosine similarity | Dot-product attention / learned similarity |
| Reinforcement update (Formula 4) | `magnitude * abs(l2_norm(event)) * sigmoid_gate` |
| Aging/decay (Formula 3) | `current_state * (1 - attenuation_factor)` |
| General model | GeneralInterestEncoder (multi-head attention) |
| Specific model | SpecificInterestEncoder (attention over term embeddings) |
| Fixed Wg, Ws, Wc | Learned or context-dependent gating |

## Citation

```bibtex
@inproceedings{li2014user,
  title={User Modeling for Contextual Suggestion},
  author={Li, Hua and Alonso, Rafael},
  booktitle={TREC},
  year={2014}
}
```

## License

MIT License — See LICENSE file for details.

---

**Phase 1**: Classical RAMA — Complete
**Phase 2**: Neural Embeddings (TF/Keras) — Complete
**Phase 3**: Encoders + Reinforcement + Scoring — Complete
**Phase 4**: Training Infrastructure — Complete
**Phase 5**: Training & Validation — Planned

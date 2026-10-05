# Neural RAMA

![tests](https://github.com/huali10044/neural-rama/actions/workflows/tests.yml/badge.svg)

A TensorFlow/Keras reimplementation of RAMA — the user-interest model behind a
**#2 of 17 groups** finish in the NIST TREC 2014 Contextual Suggestion Track
([our paper](https://trec.nist.gov/pubs/trec23/papers/pro-RAMA_cs.pdf), [local copy](2014-User%20Modeling%20for%20Contextual%20Suggestion-pro-RAMA_cs.pdf) ·
[track overview with full results](https://trec.nist.gov/pubs/trec23/papers/overview-context.pdf), [local copy](overview-context.pdf))
— rebuilt as a neural recommender while preserving the original model's semantics.

## The original result

From the track overview, Table 1 (25 open-web runs from 17 groups; ClueWeb12 runs
are ranked separately and not compared against these):

| Run | P@5 | MRR | TBG |
|---|---|---|---|
| UDInfoCS2014_2 | 0.5585 (1) | 0.7482 (1) | 2.7021 (1) |
| **RAMARUN2** | **0.5017 (2)** | **0.6846 (2)** | **2.3718 (2)** |
| BJUTa | 0.5010 (3) | 0.6677 (4) | 2.2209 (4) |

RAMA submitted two runs. RAMARUN2 placed second on **all three metrics** — the
overview notes that the top two runs hold their rank regardless of which measure
is used. The second run, RUN1, placed 6th on P@5 and 3rd on TBG. Only one group
finished ahead.

The organizers' own summary of the approach: three components — a general
interest score over categories, a specific interest score over nouns extracted
from venue titles and descriptions, and a context score from distance, review
count and average rating — linearly combined. RAMARUN2 weighted general interest
high; RUN1 weighted specific interest high. That separation of general from
specific interest is what this reimplementation preserves.

## Status

**Architecture, training stack, and test suite are complete. The model has not
been scored against the original TREC benchmark, and cannot be.**

The 2014 judgments are public, but they are not a reusable test collection, and
the track organizers said so in the same overview: ClueWeb12 provided a reusable
collection while most groups worked from the open web, and building a reusable
collection participants would actually use was named as a goal for the following
year. RAMA's runs were open-web runs.

The mechanism is visible in the qrels format. Assessors rated only the **top 5
suggestions from each submitted run**, and judgments are keyed by run ID —
`desc-doc.qrels` lines look like `BJUTa 843 124 http://... 2 2 6 2`. There is no
document-level judgment pool. Any venue a model surfaces today that the 2014
assessors never saw is simply unjudged, and `trec_eval` treats unjudged as
non-relevant. A score produced that way would look comparable to the table above
and would not be.

What *is* reusable are the inputs: 299 crowdsourced assessor profiles, 50 city
contexts, and 100 example attractions with two 5-point ratings each. That is real
preference data. It is what the synthetic generator in this repository is modelled
on, and the [score-fusion ablation](#score-fusion-ablation) below evaluates against
it directly.

## What's Implemented

**12 custom Keras layers**

- Category, term, and context embeddings
- Multi-head attention aggregation **weighted by rating polarity** — attention
  biased by the sign of the interaction rather than magnitude alone
- Differentiable translation of the paper's reinforcement and aging formulas,
  preserving the original `[-1, +1]` interest-state semantics
- Component scoring with both fixed and learned score-fusion modes

**Training stack**

- BPR pairwise ranking loss
- Custom `GradientTape` loop with checkpointing and early stopping
- Evaluation harness targeting P@5, MRR, and Time-Biased Gain

**35 unit tests (including 23 for the neural stack)**

Covering gradient flow through every custom layer, output-range constraints, and
behavioral invariants carried over from the classical model — positive history
yields positive interest state; aligned pairs outrank opposed pairs.

## Why Translate a Classical Model

RAMA separated general interests from context-specific ones and updated them
through explicit reinforcement and decay rules. Those rules are interpretable and
were tuned by hand. The question this project asks is whether the same structure
survives being made differentiable — whether the separation and the decay
semantics can be learned rather than specified, without losing the behavior that
made the original work.

The unit tests exist to answer that: each one asserts a property the classical
model guaranteed by construction and the neural version has to preserve by
training.

## Experiments

### Score-fusion ablation

RAMA scored candidate venues by linearly combining three component scores with weights set by hand for TREC 2014. Would fitting those weights to data have done better? The classical algorithm is reimplemented, validated against the user interest models published in the 2014 paper, and evaluated on a surrogate task built from the track's input data. Conditions differ only in how the general- and specific-interest scores are combined.

- Fitting the mixing weight beats both hand-set 2014 settings on graded ranking quality — consistently, and by a small margin.
- The gain lies in how the top of the ranking is ordered, not in which items reach it.
- The interest models carry real signal: every personalized condition ranks well clear of a random baseline, and for a large majority of users.
- The 2014 preference for general interest over specific interest replicates, descriptively.

The fitted weight's *value* is representation-dependent, and a population-level popularity model outperforms every per-user condition here; both are discussed in the full write-up.

Full write-up — setup, results, sensitivity analysis, and discussion: **[experiments/fusion_ablation/README.md](experiments/fusion_ablation/README.md)**

## Running It

**Python 3.12 is required.** TensorFlow 2.19 and numpy 1.26 publish no wheels for 3.13 or
later, so a newer interpreter fails at install with `No matching distribution found`.

Using [uv](https://docs.astral.sh/uv/) (it fetches the interpreter for you):

```bash
uv python install 3.12           # leaves your system Python alone
uv venv --python 3.12
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Or with the standard library, if you already have Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the test suite, install `requirements-test.txt` instead — it pins the versions CI uses.

```bash
pytest -q                        # 35 tests
```

The score-fusion ablation has its own pinned environment and is run separately; see
[experiments/fusion_ablation/README.md](experiments/fusion_ablation/README.md).

Run the end-to-end classical demo on generated data:

```bash
python run_demo.py
```

Run neural training end-to-end on generated data:

```bash
python src/training/run_training.py --regenerate-data
```

## Next

- Comparison against the classical implementation on matched synthetic data —
  the meaningful benchmark here, since both models face identical inputs
- A collaborative-filtering baseline for the fusion ablation. Its popularity
  reference is a population prior, not CF; the profile matrix is dense enough
  (299 users x 70 items) to fit item-kNN or matrix factorization directly
- Evaluation against a later Contextual Suggestion edition built on the fixed
  corpus, where the judgments are designed to be reusable

## License

Licensed under the [MIT License](LICENSE).

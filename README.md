# Neural RAMA

![tests](https://github.com/huali10044/neural-rama/actions/workflows/tests.yml/badge.svg)

A TensorFlow/Keras reimplementation of RAMA, the user-interest model behind a
**#2 of 17** finish in the NIST TREC 2014 Contextual Suggestion Track
([paper](https://apps.dtic.mil/sti/html/tr/ADA618623/)), rebuilt as a neural
recommender while preserving the original model's semantics.

## Status

**Architecture, training stack, and test suite are complete. Benchmark training
has not been run.**

The published TREC 2014 evaluation requires the official qrels, which are
distributed under a NIST data agreement and cannot be redistributed here. Rather
than include data that cannot be shipped, this repository provides a synthetic
data generator matching the format and interaction semantics of the original
task. The test suite uses generated inputs to validate the implementation.

Target metrics from the published paper, for reference:
P@5 = 0.50 | MRR = 0.71 | Time-Biased Gain = 0.70

## What's Implemented

**12 custom Keras layers**

- Category, term, and context embeddings
- Multi-head attention aggregation weighted by rating polarity: attention is
  biased by the sign of the interaction rather than magnitude alone
- Differentiable translation of the paper's reinforcement and aging formulas,
  preserving the original `[-1, +1]` interest-state semantics
- Component scoring with both fixed and learned score-fusion modes

**Training stack**

- BPR pairwise ranking loss
- Custom `GradientTape` loop with checkpointing and early stopping
- Evaluation harness for P@5, MRR, and Time-Biased Gain

**35 unit tests, including 23 for the neural stack**

The tests cover classical-model behavior, gradient flow through the neural
components, output-range constraints, and behavioral invariants carried over
from the classical model. Positive history yields a positive interest state,
and aligned pairs outrank opposed pairs.

## Why Translate a Classical Model

RAMA separated general interests from context-specific interests and updated
them through explicit reinforcement and decay rules. Those rules are
interpretable and were tuned by hand. This project asks whether the same
structure survives being made differentiable: whether the separation and decay
semantics can be learned rather than specified without losing the behavior that
made the original work.

The unit tests assert properties the classical model guaranteed by construction
and the neural version must preserve during training.

## Running It

Python 3.12 is used in CI.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Run the end-to-end classical demo on generated data:

```bash
python run_demo.py
```

Run neural training end-to-end on generated data:

```bash
python src/training/run_training.py --regenerate-data
```

## Next

- Train and evaluate with the real qrels under a NIST data agreement
- Ablate fixed versus learned score fusion
- Compare the neural and classical implementations on matched synthetic data

## License

Licensed under the [MIT License](LICENSE).
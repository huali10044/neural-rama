"""Classical RAMA, as specified in Li & Alonso (2014), with two rules recovered from the published models."""
import os, re, math, sys
from collections import Counter
from nltk.corpus import wordnet as wn

_WORDNET_READY = False

def _ensure_wordnet():
    """WordNet is an NLTK corpus, not a pip package, so requirements.txt cannot install it.
    Fetch it on first use into the active environment (nltk searches sys.prefix/nltk_data)."""
    global _WORDNET_READY
    if _WORDNET_READY: return
    try:
        wn.synsets("museum", pos="n")
    except LookupError:
        import nltk
        target = os.path.join(sys.prefix, "nltk_data")
        print(f"downloading the WordNet corpus (~11 MB) to {target} ...", file=sys.stderr)
        nltk.download("wordnet", download_dir=target, quiet=True)
        wn.synsets("museum", pos="n")          # fail loudly if the download did not work
    _WORDNET_READY = True

REL = {4: 1.0, 3: 0.5, 2: 0.0, 1: -0.5, 0: -1.0, -1: 0.0}   # paper Table 6
RF = 0.5                                                    # reinforcement factor, paper §3.2.3
LUCENE_STOP = {"a","an","and","are","as","at","be","but","by","for","if","in","into","is","it","no","not",
               "of","on","or","such","that","the","their","then","there","these","they","this","to","was",
               "will","with"}

def update(model, elements, relevance):
    """One RAMA event. elements: {element: mention_frequency}. Decay is off (attenuation 0)."""
    for el, f in elements.items():
        inc = relevance * RF * f
        model[el] = inc if el not in model else model[el] + inc * (1 - abs(model[el]))
    return model

def category_elements(cats):
    """Recovered rule: frequency 1/k for an example carrying k categories."""
    return {c: 1.0 / len(cats) for c in cats} if cats else {}

def term_elements(title, description):
    """Recovered rule: count / (tokens after stopword removal), nouns only."""
    _ensure_wordnet()
    toks = [t for t in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", f"{title} {description}".lower())
            if t not in LUCENE_STOP]
    nouns = [t for t in toks if not t.isdigit() and wn.synsets(t, pos="n")]
    n = len(toks) or 1
    return {t: c / n for t, c in Counter(nouns).items()}

def build_user(events, elements_of):
    """events: [(example_id, description_rating)], processed in ascending example ID."""
    m = {}
    for ex, rating in sorted(events):
        update(m, elements_of(ex), REL[rating])
    return m

def item_vector(elements):
    """A candidate's own model: one positive event (relevance 1)."""
    return update({}, elements, 1.0)

def cosine(u, v):
    """Full [-1, +1] range, paper §2.3.1. Zero vector -> 0."""
    dot = sum(w * v[k] for k, w in u.items() if k in v)
    nu = math.sqrt(sum(w * w for w in u.values())); nv = math.sqrt(sum(w * w for w in v.values()))
    return dot / (nu * nv) if nu and nv else 0.0

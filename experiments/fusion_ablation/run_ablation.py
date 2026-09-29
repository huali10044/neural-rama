"""Score-fusion ablation runner. Usage: python run_ablation.py [--users N] [--seeds S]"""
import csv, random, argparse, numpy as np
from rama_core import *
from fusion import fit_fusion, gains, relevant, metrics_at5

FIXED = {"fixed-general": 0.9 / 0.99, "fixed-specific": 0.09 / 0.99}   # Table 7, context dropped, renormalised

def load(data="data", categories="examples2014_categories_reconstructed.csv"):
    ex = {int(r["id"]): r for r in csv.DictReader(open(f"{data}/examples2014.csv", encoding="utf-8"))}
    cats = {int(r["example_id"]): r["categories"].split("|")
            for r in csv.DictReader(open(f"{data}/{categories}", encoding="utf-8"))}
    prof = {}
    for fn in ("profiles2014-70.csv", "profiles2014-100.csv"):
        for r in csv.reader(open(f"{data}/{fn}")):
            if r and r[0].strip().isdigit():
                prof.setdefault(r[0], {})[int(r[1])] = (int(r[2]), int(r[3]))
    shared = set.intersection(*(set(p) for p in prof.values()))          # the 70 shared attractions
    assert len(shared) == 70
    prof = {u: {e: v for e, v in p.items() if e in shared} for u, p in prof.items()}
    return ex, cats, prof, shared

def main(n_users=None, seeds=10, data="data", out="results", dump_pairs=False,
         categories="examples2014_categories_reconstructed.csv", exclude_users=(), label_noise=0.0):
    ex, cats, prof, shared = load(data, categories)
    if label_noise:                                    # corrupt a fraction of examples' category labels
        rng0 = random.Random(12345)                    # fixed: the same corruption for every seed
        vocab = sorted({c for v in cats.values() for c in v})
        for e in sorted(cats):
            if rng0.random() < label_noise:
                cats[e] = rng0.sample(vocab, len(cats[e]))
    prof = {u: p for u, p in prof.items() if u not in set(exclude_users)}
    CE = {e: category_elements(cats[e]) for e in shared}
    TE = {e: term_elements(ex[e]["title"], ex[e]["description"]) for e in shared}
    CV = {e: item_vector(CE[e]) for e in shared}; TV = {e: item_vector(TE[e]) for e in shared}
    users = sorted(prof)[:n_users] if n_users else sorted(prof)
    rows, weights = [], []
    for seed in range(seeds):
        rng = random.Random(seed)
        split, DG, DS = {}, [], []
        for u in users:                                    # --- split + LOO training scores
            items = sorted(e for e, (d, w) in prof[u].items() if d != -1 and w != -1)
            rng.shuffle(items); k = round(0.3 * len(items))
            test, train = items[:k], items[k:]
            split[u] = (train, test)
            loo = {}
            for e in train:
                ev = [(x, prof[u][x][0]) for x in train if x != e]
                loo[e] = (cosine(build_user(ev, CE.get), CV[e]), cosine(build_user(ev, TE.get), TV[e]))
            for i in train:
                for j in train:
                    if gains(*prof[u][i]) > gains(*prof[u][j]):
                        DG.append(loo[i][0] - loo[j][0]); DS.append(loo[i][1] - loo[j][1])
        DG, DS = np.array(DG), np.array(DS)
        if dump_pairs:                                     # for the TensorFlow cross-check
            import os; os.makedirs(out, exist_ok=True)
            np.savez_compressed(f"{out}/pairs_seed{seed}.npz",
                                dg=DG.astype(np.float32), ds=DS.astype(np.float32))
        wg, alpha, loss, u, v = fit_fusion(DG, DS)
        weights.append({"seed": seed, "Wg": wg, "Ws": 1 - wg, "alpha": alpha, "bpr_loss": loss,
                        "u_raw": u, "v_raw": v, "pairs": len(DG)})
        conds = dict(FIXED, learned=wg)
        for u in users:                                    # --- evaluation on held-out items
            train, test = split[u]
            ev = [(x, prof[u][x][0]) for x in train]
            G, S = build_user(ev, CE.get), build_user(ev, TE.get)
            g = {e: cosine(G, CV[e]) for e in test}; s = {e: cosine(S, TV[e]) for e in test}
            gain = {e: gains(*prof[u][e]) for e in test}; rel = {e: relevant(*prof[u][e]) for e in test}
            if not any(rel.values()): continue             # unjudgeable user for this split
            pop = {}
            for e in test:                                 # mean description rating from other users' training data
                vals = [prof[v][e][0] for v in users if v != u and e in split[v][0]]
                pop[e] = float(np.mean(vals)) if vals else 2.0   # no data: scale midpoint (only with tiny --users)
            ranks = {c: sorted(test, key=lambda e: (-(w * g[e] + (1 - w) * s[e]), e)) for c, w in conds.items()}
            ranks["popularity"] = sorted(test, key=lambda e: (-pop[e], e))
            ranks["random"] = rng.sample(test, len(test))
            for c, r in ranks.items():
                p5, mrr, ndcg = metrics_at5(r, gain, rel)
                rows.append({"seed": seed, "user": u, "condition": c, "P5": p5, "MRR": mrr, "NDCG5": ndcg})
    import os, json, platform, subprocess, datetime, scipy, nltk
    os.makedirs(out, exist_ok=True)
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: commit = "unknown"
    json.dump({"date": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
               "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "nltk": nltk.__version__, "users": len(users), "seeds": seeds, "holdout": 0.3,
               "categories": categories, "excluded_users": list(exclude_users), "label_noise": label_noise,
               "fixed_weights": FIXED}, open(f"{out}/run_info.json", "w"), indent=2)
    for name, data_ in (("per_user_scores.csv", rows), ("learned_weights.csv", weights)):
        with open(f"{out}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data_[0])); w.writeheader(); w.writerows(data_)
    return rows, weights

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--users", type=int); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--dump-pairs", action="store_true", help="save training pair differences for cross_check.py")
    ap.add_argument("--out", default="results", help="output directory (use a separate one for --dump-pairs runs)")
    ap.add_argument("--categories", default="examples2014_categories_reconstructed.csv",
                    help="category mapping file in data/")
    ap.add_argument("--exclude-users", nargs="*", default=[], help="user IDs to drop entirely")
    ap.add_argument("--label-noise", type=float, default=0.0,
                    help="fraction of examples whose categories are randomly reassigned")
    a = ap.parse_args()
    main(a.users, a.seeds, out=a.out, dump_pairs=a.dump_pairs, categories=a.categories,
         exclude_users=a.exclude_users, label_noise=a.label_noise)

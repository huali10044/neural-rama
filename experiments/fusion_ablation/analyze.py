"""Paired per-user analysis. Usage: python analyze.py [results_dir]"""
import csv, sys, numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon

def main(d="results", metric_list=("NDCG5", "P5", "MRR"), n_boot=10000):
    rows = list(csv.DictReader(open(f"{d}/per_user_scores.csv")))
    W = list(csv.DictReader(open(f"{d}/learned_weights.csv")))
    seeds_rows = {int(r["seed"]) for r in rows}; seeds_w = {int(r["seed"]) for r in W}
    if seeds_rows != seeds_w:
        raise SystemExit(f"inconsistent results: per_user_scores.csv has seeds {sorted(seeds_rows)} but "
                         f"learned_weights.csv has {sorted(seeds_w)}. A later run overwrote part of {d}/ — re-run.")
    wg = np.array([float(r["Wg"]) for r in W])
    print(f"Learned Wg: mean {wg.mean():.3f}  sd {wg.std(ddof=1) if len(wg) > 1 else 0:.3f}  "
          f"range [{wg.min():.3f}, {wg.max():.3f}] over {len(wg)} seeds   (2014: RAMARUN2 0.909, RUN1 0.091)")
    u = np.array([float(r["u_raw"]) for r in W]); v = np.array([float(r["v_raw"]) for r in W])
    print(f"Unconstrained weights: general u<0 in {(u < 0).sum()}/{len(u)} seeds, specific v<0 in {(v < 0).sum()}/{len(v)} seeds"
          "   (a negative weight means that component hurts ranking; Wg is then on the 0/1 boundary)\n")
    # 1. per-seed condition means -> between-seed spread (report BEFORE any comparison)
    for metric in metric_list:
        by = defaultdict(list)
        for r in rows: by[(r["condition"], r["seed"])].append(float(r[metric]))
        conds = sorted({c for c, _ in by})
        print(f"{metric} per-seed spread:")
        for c in conds:
            m = [np.mean(v) for (cc, s), v in by.items() if cc == c]
            print(f"  {c:<15} mean {np.mean(m):.4f}  sd across seeds {np.std(m, ddof=1) if len(m) > 1 else 0:.4f}")
        print()
    # 2. average each user's score across seeds (damps split noise), then pair by user
    for metric in metric_list:
        u = defaultdict(lambda: defaultdict(list))
        for r in rows: u[r["condition"]][r["user"]].append(float(r[metric]))
        um = {c: {k: np.mean(v) for k, v in d_.items()} for c, d_ in u.items()}
        print(f"\n{metric}: learned vs fixed (paired by user, Holm across 2 comparisons)")
        res = []
        for base in ("fixed-general", "fixed-specific"):
            users = sorted(set(um["learned"]) & set(um[base]))
            diff = np.array([um["learned"][k] - um[base][k] for k in users])
            nz = int((np.abs(diff) > 1e-12).sum())
            p = wilcoxon(diff[np.abs(diff) > 1e-12]).pvalue if nz >= 6 else float("nan")
            rng = np.random.default_rng(0)
            boots = [rng.choice(diff, len(diff)).mean() for _ in range(n_boot)]
            res.append([base, len(users), nz, diff.mean(), np.median(diff), *np.percentile(boots, [2.5, 97.5]), p])
        order = np.argsort([r[-1] for r in res]); m = len(res); prev = 0
        for rank, i in enumerate(order):                  # Holm step-down
            adj = min(1.0, max(prev, res[i][-1] * (m - rank))); res[i].append(adj); prev = adj
        for b, n, nz, mean, med, lo, hi, p, padj in res:
            print(f"  vs {b:<15} users {n}  non-zero pairs {nz}  mean diff {mean:+.4f} "
                  f"[95% CI {lo:+.4f}, {hi:+.4f}]  median {med:+.4f}  p {p:.3g}  Holm p {padj:.3g}")
        for ref in ("popularity", "random"):
            if ref not in um: continue
            users = sorted(set(um["learned"]) & set(um[ref]))
            diff = np.array([um["learned"][k] - um[ref][k] for k in users])
            nz = diff[np.abs(diff) > 1e-12]
            p = wilcoxon(nz).pvalue if len(nz) >= 6 else float("nan")
            print(f"  reference {ref:<11} mean {np.mean(list(um[ref].values())):.4f}   "
                  f"learned - {ref}: {diff.mean():+.4f}  p {p:.2g}  learned better for {(diff > 0).sum()}/{len(diff)} users")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")

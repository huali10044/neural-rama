import math
import numpy as np

def fit_fusion(pairs_dg, pairs_ds, l2=1e-6):
    """BPR fit of one global Wg (Ws = 1 - Wg) with a free scale alpha.

    pairs_dg, pairs_ds: arrays of (g_i - g_j), (s_i - s_j) over training pairs with gain_i > gain_j,
    computed from leave-one-out component scores.

    Reparameterise u = alpha * Wg, v = alpha * (1 - Wg). The BPR loss
        mean log(1 + exp(-(u * dg + v * ds)))
    is then convex in (u, v): logistic regression on pair differences with no intercept.
    It has one global optimum, found exactly; Wg = u / (u + v). The tiny l2 term only guarantees
    a finite optimum if the pairs were ever perfectly separable, and does not move the ratio.

    Wg is constrained to [0, 1] (paper, formula 2). If the unconstrained optimum puts a negative
    weight on a component, the constrained optimum is on the boundary, and the raw (u, v) are
    returned so that the sign can be reported: a negative weight means a component hurts ranking.
    Returns (Wg, alpha, loss, u, v)."""
    from scipy.optimize import minimize
    X = np.column_stack([pairs_dg, pairs_ds])
    def f(p):
        z = X @ p
        loss = np.logaddexp(0, -z).mean() + l2 * (p @ p)
        grad = -(X * (1 / (1 + np.exp(z)))[:, None]).mean(axis=0) + 2 * l2 * p
        return loss, grad
    u, v = minimize(f, np.array([1.0, 1.0]), jac=True, method="BFGS").x
    if u >= 0 and v >= 0 and u + v > 0:
        return float(u / (u + v)), float(u + v), float(f(np.array([u, v]))[0]), float(u), float(v)
    # boundary: fit the single component that keeps a non-negative weight
    best = None
    for wg, col in ((1.0, 0), (0.0, 1)):
        g = lambda a: (np.logaddexp(0, -a[0] * X[:, col]).mean() + l2 * a[0] ** 2,
                       np.array([-(X[:, col] / (1 + np.exp(a[0] * X[:, col]))).mean() + 2 * l2 * a[0]]))
        r = minimize(g, np.array([1.0]), jac=True, method="L-BFGS-B", bounds=[(0, None)])
        if best is None or r.fun < best[2]: best = (wg, float(r.x[0]), float(r.fun))
    return best[0], best[1], best[2], float(u), float(v)

def gains(desc, web):  # evaluation gain from both ratings, track-consistent
    return min(desc, web)
def relevant(desc, web):  # TREC 2014 threshold: both >= 3
    return desc >= 3 and web >= 3

def metrics_at5(ranked, gain, rel):
    top = ranked[:5]
    p5 = sum(rel[i] for i in top) / 5
    mrr = next((1 / (r + 1) for r, i in enumerate(top) if rel[i]), 0.0)
    dcg = sum(gain[i] / math.log2(r + 2) for r, i in enumerate(top))
    ideal = sorted(gain.values(), reverse=True)[:5]
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
    return p5, mrr, (dcg / idcg if idcg else 0.0)

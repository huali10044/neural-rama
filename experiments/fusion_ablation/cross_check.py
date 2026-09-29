"""Cross-check: fit the same BPR objective with TensorFlow and compare to the exact convex solution.

The objective is convex, so there is one right answer. This verifies that gradient-based fitting --
the mechanism the neural package's learned fusion mode uses -- converges to it. A mismatch points at
the optimiser or the parameterisation in the neural package, not at this experiment.

Run in the REPOSITORY'S environment (the one with TensorFlow), not the experiment venv:
    python run_ablation.py --users 299 --seeds 1 --dump-pairs   # experiment venv, writes pairs_seed0.npz
    python cross_check.py --seed 0                              # repo venv (TensorFlow)

Requires only numpy and tensorflow. Reads the exact answer from results/learned_weights.csv.
"""
import argparse, csv, numpy as np

def fit_tf(dg, ds, steps=3000, lr=0.05, seed=0, log_every=500):
    import tensorflow as tf
    tf.random.set_seed(seed)
    dg = tf.constant(dg, tf.float32); ds = tf.constant(ds, tf.float32)
    theta = tf.Variable(0.0)   # Wg = sigmoid(theta), so Wg in (0, 1) and Wg + Ws = 1 by construction
    rho = tf.Variable(0.0)     # alpha = softplus(rho) > 0, the free score scale
    opt = tf.keras.optimizers.Adam(lr)
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            wg = tf.sigmoid(theta); alpha = tf.nn.softplus(rho)
            d = alpha * (wg * dg + (1.0 - wg) * ds)
            loss = tf.reduce_mean(tf.math.softplus(-d))     # -log sigmoid(d), stable
        opt.apply_gradients(zip(tape.gradient(loss, [theta, rho]), [theta, rho]))
        return loss
    for i in range(steps):
        loss = step()
        if log_every and (i + 1) % log_every == 0:
            print(f"  step {i+1:>5}  loss {float(loss):.6f}  Wg {float(tf.sigmoid(theta)):.4f}")
    return float(tf.sigmoid(theta)), float(tf.nn.softplus(rho)), float(loss)

def main(seed=0, results="results", tol=0.005, **kw):
    z = np.load(f"{results}/pairs_seed{seed}.npz")
    dg, ds = z["dg"], z["ds"]
    exact = {int(r["seed"]): r for r in csv.DictReader(open(f"{results}/learned_weights.csv"))}[seed]
    wg_exact, loss_exact = float(exact["Wg"]), float(exact["bpr_loss"])
    print(f"pairs: {len(dg):,}   exact solution: Wg = {wg_exact:.4f}  loss {loss_exact:.6f}")
    wg_tf, alpha_tf, loss_tf = fit_tf(dg, ds, seed=seed, **kw)
    gap = abs(wg_tf - wg_exact)
    print(f"\nTensorFlow: Wg = {wg_tf:.4f}  alpha = {alpha_tf:.3f}  loss {loss_tf:.6f}")
    print(f"gap = {gap:.4f}  ->  {'PASS' if gap <= tol else 'FAIL'} (tolerance {tol})")
    if gap > tol:
        print("  Not a failure of the experiment: the convex optimum is the reference. Check the\n"
              "  neural package's fusion parameterisation (is Wg + Ws = 1 enforced?), its learning\n"
              "  rate and step count, and whether a free score scale is present.")
    return gap <= tol

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0); p.add_argument("--results", default="results")
    p.add_argument("--steps", type=int, default=3000); p.add_argument("--lr", type=float, default=0.05)
    a = p.parse_args(); main(a.seed, a.results, steps=a.steps, lr=a.lr)

"""Non-PDE reference baselines for regime classification.

1. static   - logistic regression on the causal inputs (P, Ip, nebar) per
              sample: does the task need dynamics at all?
2. lag      - first-order relaxation tau*dz/dt = alpha(r) - z with a
              sigmoid readout: does a single relaxation time suffice
              without cubic structure?

Both are trained on the grouped TRAIN sessions with the same weak labels
(clean L/H samples only) and evaluated per-shot on the requested role.
Writes experiments/baselines_<role>.json.
"""

import argparse
import glob
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from fusion_ode_identification.regime_metrics import regime_classification_metrics

SCALES = np.array([1e6, 1e6, 1e19])


def load_shot(shot):
    d = np.load(f"data/{shot}_torax_training.npz", allow_pickle=True)
    t = d["t"]
    X = np.stack([d["P_nbi"], np.abs(d["Ip"]), d["nebar"]], -1) / SCALES
    reg = d["regime"].astype(float)
    mask = ((reg > 0.5) & (reg < 1.5)) | ((reg > 2.5) & (reg < 3.5))
    y = (reg > 2.0).astype(float)
    return t, X, y, mask, reg


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def fit_logistic(X, y, iters=3000, lr=0.3, l2=1e-4):
    w = np.zeros(X.shape[1] + 1)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        g = Xb.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w


def lag_rollout(t, X, theta):
    """theta = [w(3), b, log_tau, k, k0]; explicit Euler on the shot grid."""
    w, b = theta[:3], theta[3]
    tau = np.exp(theta[4])
    alpha = X @ w + b
    z = np.empty(len(t))
    z[0] = alpha[0]
    for i in range(1, len(t)):
        dt = max(t[i] - t[i - 1], 1e-4)
        z[i] = z[i - 1] + dt * (alpha[i - 1] - z[i - 1]) / tau
    return theta[5] * z + theta[6]


def fit_lag(shots_data, iters=400):
    theta = np.array([1.0, 0.0, 0.0, -1.0, np.log(0.025), 2.0, -1.0])
    eps = 1e-4

    def loss(th):
        tot, n = 0.0, 0
        for t, X, y, mask, _ in shots_data:
            logit = lag_rollout(t, X, th)
            p = sigmoid(logit[mask])
            yy = y[mask]
            if len(yy) == 0:
                continue
            tot += -np.mean(yy * np.log(p + 1e-9) + (1 - yy) * np.log(1 - p + 1e-9))
            n += 1
        return tot / max(n, 1)

    # SPSA (cheap, derivative-free; adequate for 7 parameters)
    rng = np.random.default_rng(0)
    a0 = 0.15
    best, best_l = theta.copy(), loss(theta)
    for k in range(iters):
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        c = 0.05 / (1 + k) ** 0.2
        g = (loss(theta + c * delta) - loss(theta - c * delta)) / (2 * c) * delta
        theta = theta - a0 / (1 + k) ** 0.4 * g
        l = loss(theta)
        if l < best_l:
            best, best_l = theta.copy(), l
    return best


def per_shot_metrics(shots, model_logit_fn):
    out = {}
    for shot in shots:
        t, X, y, mask, reg = load_shot(shot)
        logits = model_logit_fn(t, X)
        out[str(shot)] = regime_classification_metrics(logits, reg, mask.astype(float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="val", choices=["val", "test"])
    args = ap.parse_args()

    with open("data/split.json") as f:
        split = json.load(f)
    train_shots = split["train"]
    eval_shots = split[args.role]

    train_data = [load_shot(s) for s in train_shots]

    # static logistic
    Xtr = np.concatenate([X[m] for _, X, y, m, _ in train_data])
    ytr = np.concatenate([y[m] for _, X, y, m, _ in train_data])
    w = fit_logistic(Xtr, ytr)
    static = per_shot_metrics(eval_shots, lambda t, X: np.hstack([X, np.ones((len(X), 1))]) @ w)

    # first-order lag
    th = fit_lag(train_data)
    lag = per_shot_metrics(eval_shots, lambda t, X: lag_rollout(t, X, th))

    def agg(per_shot):
        aucs = [v["auc"] for v in per_shot.values() if v.get("n_H", 0) > 0 and np.isfinite(v.get("auc", np.nan))]
        accs = [v["accuracy"] for v in per_shot.values() if v.get("n_scored", 0) > 1]
        return {"median_auc": float(np.median(aucs)) if aucs else float("nan"),
                "median_acc": float(np.median(accs)) if accs else float("nan"),
                "n_auc_shots": len(aucs)}

    report = {
        "role": args.role,
        "static_logistic": {"weights": w.tolist(), "aggregate": agg(static), "per_shot": static},
        "first_order_lag": {"theta": th.tolist(), "aggregate": agg(lag), "per_shot": lag},
    }
    os.makedirs("experiments", exist_ok=True)
    with open(f"experiments/baselines_{args.role}.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v["aggregate"] for k, v in report.items() if isinstance(v, dict) and "aggregate" in v}, indent=1))


if __name__ == "__main__":
    main()

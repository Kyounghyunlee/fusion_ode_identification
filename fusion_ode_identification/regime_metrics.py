"""Quantitative L/H regime assessment.

All metrics operate on the model's calibrated regime probability
p_H(t) = sigmoid(regime_logit(z(t))) and, for the cusp latent, on the
closed-form bifurcation structure of the normal form
    tau * dz/dt = a(u) + b z - z^3.

Nothing here depends on JAX; inputs are NumPy arrays.
"""

from typing import Dict, Optional

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def regime_classification_metrics(
    regime_logits: np.ndarray,
    regime_ts: np.ndarray,
    regime_mask: np.ndarray,
) -> Dict[str, float]:
    """Classification quality of p_H against the L/H labels.

    Labels follow the pack convention: 1 = L, 3 = H; only samples with
    regime_mask > 0.5 (clean L or H, inside the plasma window) are scored.
    """
    m = np.asarray(regime_mask) > 0.5
    if m.sum() < 2:
        return {"n_scored": int(m.sum())}
    y = (np.asarray(regime_ts)[m] > 2.0).astype(float)
    p = _sigmoid(np.asarray(regime_logits)[m])
    yhat = (p > 0.5).astype(float)

    tp = float(np.sum((yhat == 1) & (y == 1)))
    tn = float(np.sum((yhat == 0) & (y == 0)))
    fp = float(np.sum((yhat == 1) & (y == 0)))
    fn = float(np.sum((yhat == 0) & (y == 1)))
    acc = (tp + tn) / max(len(y), 1)
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    # AUC via rank statistic (Mann-Whitney U); defined only with both classes.
    auc = float("nan")
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(p) + 1)
        # midranks for ties
        sorted_p = p[order]
        i = 0
        while i < len(sorted_p):
            j = i
            while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
                j += 1
            if j > i:
                ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
            i = j + 1
        auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    # Brier score: calibration-sensitive proper scoring rule.
    brier = float(np.mean((p - y) ** 2))

    return {
        "n_scored": int(len(y)),
        "n_H": n_pos,
        "n_L": n_neg,
        "accuracy": float(acc),
        "f1": float(f1),
        "auc": auc,
        "brier": brier,
    }


def transition_time_error(
    ts: np.ndarray,
    regime_logits: np.ndarray,
    labeled_transition_time: float,
) -> Dict[str, float]:
    """Time of the model's first sustained p_H = 0.5 upcrossing vs the label."""
    p = _sigmoid(np.asarray(regime_logits))
    above = p > 0.5
    t_model = float("nan")
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            # require it to stay above for at least 5 samples (sustained)
            j_end = min(len(above), i + 5)
            if np.all(above[i:j_end]):
                t_model = float(ts[i])
                break
    err = float("nan")
    if np.isfinite(t_model) and np.isfinite(labeled_transition_time):
        err = float(t_model - labeled_transition_time)
    return {
        "t_transition_model": t_model,
        "t_transition_label": float(labeled_transition_time) if np.isfinite(labeled_transition_time) else float("nan"),
        "transition_time_error_s": err,
    }


def cusp_bifurcation_diagnostics(
    latent,
    latent_features_ts: np.ndarray,
    zs: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """Closed-form bifurcation diagnostics for the cusp latent.

    Returns time series of the drive a(t), the fold amplitude a_fold, the
    normalized bifurcation margin, and which attractor basin the state
    occupies. Returns None for non-cusp latents.
    """
    if not hasattr(latent, "a_fold"):
        return None
    import jax
    import jax.numpy as jnp

    feats = jnp.asarray(latent_features_ts)
    a_t = np.asarray(jax.vmap(latent.drive)(feats))
    b = float(latent.b_eff())
    a_fold = float(latent.a_fold())
    tau = float(latent.tau_eff())

    # Equilibria of a + b z - z^3 = 0 for each t; classify basin of z(t).
    zs = np.asarray(zs)
    basin = np.zeros_like(zs)  # +1 upper (H), -1 lower (L)
    z_saddle = np.full_like(zs, np.nan)
    for i, a in enumerate(a_t):
        roots = np.roots([-1.0, 0.0, b, float(a)])
        real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
        if len(real) == 3:
            z_saddle[i] = real[1]
            basin[i] = 1.0 if zs[i] > real[1] else -1.0
        elif len(real) >= 1:
            # Monostable: the single attractor's sign defines the regime.
            basin[i] = 1.0 if real[np.argmax(np.abs(real))] > 0 else -1.0

    return {
        "a_t": a_t,
        "a_fold": a_fold,
        "b": b,
        "tau": tau,
        # margin > 0: bistable; margin_LH < 0: L branch destroyed (forced H).
        "margin_LH": a_fold - a_t,   # distance of drive below the L->H fold
        "margin_HL": a_t + a_fold,   # distance of drive above the H->L fold
        "z_saddle": z_saddle,
        "basin": basin,
        "bistable_fraction": float(np.mean(np.abs(a_t) < a_fold)),
    }

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


def normal_form_diagnostics(
    latent,
    latent_features_ts: np.ndarray,
    zs: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """Closed-form diagnostics for the unfolded cubic latent.

    For tau * dz/dt = a(u) + c1*z + c2*z^2 - z^3, the equilibrium drive is
    g(z) = z^3 - c2*z^2 - c1*z. If c2^2 + 3*c1 > 0, g has two critical
    points and the identified vector field is bistable for
    a in (a_fold_low, a_fold_high); otherwise it is monostable and the folds
    are reported as NaN. Whether the data selects a bistable field is an
    RESULT of identification, not an assumption.
    """
    if not hasattr(latent, "beta_raw"):
        return None
    import jax
    import jax.numpy as jnp

    feats = jnp.asarray(latent_features_ts)
    a_t = np.asarray(jax.vmap(latent.drive)(feats))
    beta = float(latent.beta())
    c1, c2 = beta, 0.0  # depressed cubic
    tau = float(latent.tau_eff())

    if beta > 0:
        a_fold_high = 2.0 * (beta / 3.0) ** 1.5
        a_fold_low = -a_fold_high
        bistable = True
    else:
        a_fold_low = a_fold_high = float("nan")
        bistable = False

    # Equilibria of a + c1 z + c2 z^2 - z^3 = 0 per sample; basin of z(t).
    zs = np.asarray(zs)
    basin = np.zeros_like(zs)  # +1 upper (H), -1 lower (L)
    z_saddle = np.full_like(zs, np.nan)
    for i, a in enumerate(a_t):
        roots = np.roots([-1.0, c2, c1, float(a)])
        real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
        if len(real) == 3:
            z_saddle[i] = real[1]
            basin[i] = 1.0 if zs[i] > real[1] else -1.0
        elif len(real) >= 1:
            basin[i] = 1.0 if real[0] > 0 else -1.0

    return {
        "a_t": a_t,
        "beta": beta,
        "c1": c1,
        "c2": c2,
        "tau": tau,
        "bistable": bistable,
        "a_fold_low": a_fold_low,    # H branch lost below this drive
        "a_fold_high": a_fold_high,  # L branch lost above this drive
        # margins are NaN when the identified field is monostable
        "margin_LH": a_fold_high - a_t,
        "margin_HL": a_t - a_fold_low,
        "z_saddle": z_saddle,
        "basin": basin,
        "bistable_fraction": float(np.mean((a_t > a_fold_low) & (a_t < a_fold_high))) if bistable else 0.0,
    }


def _label_events(ts, regime_ts):
    """Label switch times: list of (time, +1 for L->H, -1 for H->L).

    Works on the 1/2/3 label code; transition windows (2) are bridged by
    looking at the nearest clean labels on each side.
    """
    reg = np.asarray(regime_ts)
    clean = np.where((reg > 0.5) & (reg < 1.5), 1, np.where((reg > 2.5) & (reg < 3.5), 3, 0))
    idx = np.where(clean > 0)[0]
    events = []
    for a, b in zip(idx[:-1], idx[1:]):
        if clean[a] == 1 and clean[b] == 3:
            events.append((float(0.5 * (ts[a] + ts[b])), +1))
        elif clean[a] == 3 and clean[b] == 1:
            events.append((float(0.5 * (ts[a] + ts[b])), -1))
    return events


def _model_events(ts, regime_logits, dwell_n=5):
    """Sustained p_H = 0.5 crossings, both directions."""
    p = _sigmoid(np.asarray(regime_logits))
    above = p > 0.5
    events = []
    for i in range(1, len(above)):
        if above[i] != above[i - 1]:
            j = min(len(above), i + dwell_n)
            if np.all(above[i:j] == above[i]):
                events.append((float(ts[i]), +1 if above[i] else -1))
    return events


def event_metrics(ts, regime_logits, regime_ts, tol_s=0.04, dwell_n=5):
    """One-to-one event matching within a tolerance window.

    Returns per-direction true/false positives, misses, and signed timing
    errors of the matched pairs.
    """
    lab = _label_events(ts, regime_ts)
    mod = _model_events(ts, regime_logits, dwell_n=dwell_n)
    out = {}
    for direction, tag in ((+1, "LH"), (-1, "HL")):
        L = [t for t, d in lab if d == direction]
        M = [t for t, d in mod if d == direction]
        used = set()
        matches = []
        for tl in L:
            best, best_j = None, None
            for j, tm in enumerate(M):
                if j in used or abs(tm - tl) > tol_s:
                    continue
                if best is None or abs(tm - tl) < abs(best - tl):
                    best, best_j = tm, j
            if best is not None:
                used.add(best_j)
                matches.append(best - tl)
        out[tag] = {
            "n_label": len(L),
            "n_model": len(M),
            "n_matched": len(matches),
            "misses": len(L) - len(matches),
            "false_alarms": len(M) - len(matches),
            "timing_errors_s": matches,
        }
    return out


def reliability_curve(p, y, n_bins=10):
    """Reliability diagram data + expected calibration error."""
    p = np.asarray(p)
    y = np.asarray(y)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "confidence": conf, "frequency": acc, "n": int(m.sum())})
        ece += m.mean() * abs(conf - acc)
    return {"bins": rows, "ece": float(ece)}

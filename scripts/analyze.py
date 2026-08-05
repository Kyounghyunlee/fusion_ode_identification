"""Primary results generator: calibration, classification, events, cohorts.

Consumes the per-shot artifacts written by evaluate_model.py
(bifurcation_shot_*.npz: ts, regime_logits, regime_ts, z, a_t, ...).

Calibration: Platt scaling p = sigmoid(A * logit + B) with (A, B) fitted by
BCE on the VALIDATION role only, then applied unchanged to any other role.
Raw (uncalibrated) metrics are reported alongside, and AUC is invariant to
this monotone map by construction.

Usage:
  python scripts/analyze.py --model-id v2_free_s0 --role val        # fits + saves calibration
  python scripts/analyze.py --model-id v2_free_s0 --role test       # applies saved calibration
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

from fusion_ode_identification.regime_metrics import (
    _sigmoid,
    event_metrics,
    regime_classification_metrics,
    reliability_curve,
)


def load_shots(model_id, shots):
    out = {}
    for s in shots:
        p = f"logs/{model_id}/evaluation/bifurcation_shot_{s}.npz"
        if os.path.exists(p):
            out[s] = np.load(p)
    return out


def clean_mask(reg):
    return ((reg > 0.5) & (reg < 1.5)) | ((reg > 2.5) & (reg < 3.5))


def fit_platt(logits, y, iters=4000, lr=0.05):
    """p = sigmoid(A*logit + B); minimize BCE (A initialized at 1)."""
    A, B = 1.0, 0.0
    for _ in range(iters):
        p = _sigmoid(A * logits + B)
        gA = np.mean((p - y) * logits)
        gB = np.mean(p - y)
        A -= lr * gA
        B -= lr * gB
    return float(A), float(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--role", default="val")
    ap.add_argument("--tol-s", type=float, default=0.04)
    args = ap.parse_args()

    with open("data/split.json") as f:
        split = json.load(f)
    audit = json.load(open("data/label_audit.json"))
    cohort_of = {int(k): v["cohort"] for k, v in audit["shots"].items()}

    shots = split[args.role]
    data = load_shots(args.model_id, shots)
    if not data:
        raise SystemExit(f"no artifacts for {args.model_id} role={args.role}")

    # pooled clean-label samples
    L, Y = [], []
    for s, d in data.items():
        m = clean_mask(d["regime_ts"])
        L.append(d["regime_logits"][m])
        Y.append((d["regime_ts"][m] > 2).astype(float))
    Lp, Yp = np.concatenate(L), np.concatenate(Y)

    calib_path = f"experiments/calibration_{args.model_id}.json"
    if args.role == "val":
        A, B = fit_platt(Lp, Yp)
        json.dump({"A": A, "B": B, "fitted_on": "val", "n_samples": int(len(Yp))},
                  open(calib_path, "w"), indent=1)
    else:
        if not os.path.exists(calib_path):
            raise SystemExit("calibration must be fitted on --role val first")
        c = json.load(open(calib_path))
        A, B = c["A"], c["B"]

    per_shot, ev_rows = {}, []
    for s, d in data.items():
        reg, lg, ts = d["regime_ts"], d["regime_logits"], d["ts"]
        m = clean_mask(reg)
        raw = regime_classification_metrics(lg, reg, m.astype(float))
        cal = regime_classification_metrics(A * lg + B, reg, m.astype(float))
        ev = event_metrics(ts, A * lg + B, reg, tol_s=args.tol_s)
        per_shot[str(s)] = {"cohort": cohort_of.get(s, "unknown"), "raw": raw, "calibrated": cal,
                            "events": ev, "max_pH": float(_sigmoid(A * lg + B).max())}
        ev_rows.append(ev)

    def med(key, sub="calibrated", cond=lambda v: True):
        vals = [v[sub][key] for v in per_shot.values()
                if cond(v) and key in v[sub] and np.isfinite(v[sub][key])]
        return float(np.median(vals)) if vals else float("nan"), len(vals)

    both = lambda v: v["raw"].get("n_H", 0) > 0 and v["raw"].get("n_L", 0) > 0
    auc_med, n_auc = med("auc", "raw", both)
    aucs = [v["raw"]["auc"] for v in per_shot.values() if both(v) and np.isfinite(v["raw"].get("auc", np.nan))]
    rng = np.random.default_rng(0)
    boot = [float(np.median(rng.choice(aucs, len(aucs)))) for _ in range(4000)] if len(aucs) > 2 else []

    # event aggregates
    agg_ev = {}
    for tag in ("LH", "HL"):
        n_lab = sum(e[tag]["n_label"] for e in ev_rows)
        n_mod = sum(e[tag]["n_model"] for e in ev_rows)
        n_match = sum(e[tag]["n_matched"] for e in ev_rows)
        errs = np.array([x for e in ev_rows for x in e[tag]["timing_errors_s"]])
        agg_ev[tag] = {
            "n_label": n_lab, "n_model": n_mod, "n_matched": n_match,
            "recall": n_match / n_lab if n_lab else float("nan"),
            "precision": n_match / n_mod if n_mod else float("nan"),
            "timing_median_ms": float(np.median(np.abs(errs)) * 1e3) if errs.size else float("nan"),
            "timing_iqr_ms": [float(np.percentile(np.abs(errs), 25) * 1e3),
                              float(np.percentile(np.abs(errs), 75) * 1e3)] if errs.size else None,
            "timing_signed_median_ms": float(np.median(errs) * 1e3) if errs.size else float("nan"),
        }

    # negative control on CONFIRMED negatives only
    negs = {s: v for s, v in per_shot.items() if v["cohort"] == "confirmed_negative"}
    neg_stats = {
        "n_confirmed_negative": len(negs),
        "n_with_false_H_event": sum(1 for v in negs.values() if v["events"]["LH"]["n_model"] > 0),
        "max_pH_values": {s: round(v["max_pH"], 3) for s, v in negs.items()},
    }
    amb = {s: v for s, v in per_shot.items() if v["cohort"] == "ambiguous"}

    rel = reliability_curve(_sigmoid(A * Lp + B), Yp)
    rel_raw = reliability_curve(_sigmoid(Lp), Yp)

    out = {
        "model_id": args.model_id, "role": args.role, "n_shots": len(per_shot),
        "calibration": {"A": A, "B": B, "ece_raw": rel_raw["ece"], "ece_calibrated": rel["ece"]},
        "reliability_calibrated": rel["bins"],
        "aggregate": {
            "median_auc": auc_med, "n_auc_shots": n_auc,
            "auc_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
            "median_f1_raw": med("f1", "raw", both)[0],
            "median_f1_calibrated": med("f1", "calibrated", both)[0],
            "median_acc_calibrated": med("accuracy", "calibrated")[0],
            "median_brier_calibrated": med("brier", "calibrated")[0],
        },
        "events": agg_ev,
        "negative_control": neg_stats,
        "n_ambiguous_shots": len(amb),
        "per_shot": per_shot,
    }
    os.makedirs("experiments", exist_ok=True)
    with open(f"experiments/analysis_{args.model_id}_{args.role}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_shot" and k != "reliability_calibrated"}, indent=1))


if __name__ == "__main__":
    main()

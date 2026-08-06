"""Uncertainty quantification for the identified low-dimensional parameters.

Three complementary sources, with their assumptions stated:

1. Laplace (local curvature): Gaussian approximation from the Hessian of the
   mean training loss with respect to the 8 low-dimensional latent
   parameters (beta_raw, tau_raw, drive weights/bias, readout gain/offset),
   holding the network blocks fixed at the optimum. Local and approximate.
2. Seed spread: the same quantities across independently trained seeds.
3. Shot bootstrap of per-shot evaluation metrics (resampling shots).

Derived quantities (beta, folds alpha_f = +/- 2(beta/3)^{3/2}, equivalent
power threshold) get delta-method intervals through autodiff of the maps.

Usage: python scripts/uncertainty.py --model-ids v2_free_s0 v2_free_s1 v2_free_s2
"""

import argparse
import glob
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import yaml

from fusion_ode_identification.data import load_data
from fusion_ode_identification.loss import shot_loss_imex
from fusion_ode_identification.model import build_hybrid_model
from fusion_ode_identification.types import IMEXConfig, LossCfg


def _theta_from_model(model):
    lat = model.latent
    return jnp.array([
        lat.beta_raw, lat.tau_raw,
        lat.drive_weights[0], lat.drive_weights[1], lat.drive_weights[2],
        lat.drive_bias, lat.kb_raw, lat.bb,
    ])


def _model_with_theta(model, th):
    lat = model.latent
    lat = eqx.tree_at(lambda l: l.beta_raw, lat, th[0])
    lat = eqx.tree_at(lambda l: l.tau_raw, lat, th[1])
    lat = eqx.tree_at(lambda l: l.drive_weights, lat, th[2:5])
    lat = eqx.tree_at(lambda l: l.drive_bias, lat, th[5])
    lat = eqx.tree_at(lambda l: l.kb_raw, lat, th[6])
    lat = eqx.tree_at(lambda l: l.bb, lat, th[7])
    return eqx.tree_at(lambda m: m.latent, model, lat)


def derived(th, ip_ma=0.75, ne19=2.5):
    """Map raw params -> (beta, alpha_fold_high, P_threshold_MW)."""
    beta = th[0]
    kP = jax.nn.softplus(th[2])
    alpha_f = jnp.where(beta > 0, 2.0 * (jnp.clip(beta, 1e-9) / 3.0) ** 1.5, jnp.nan)
    p_thr = (alpha_f - th[3] * ip_ma - th[4] * ne19 - th[5]) / kP
    return jnp.array([beta, alpha_f, p_thr])


def laplace(model_id, cfg, n_shots_max=40):
    model = build_hybrid_model(cfg, jax.random.PRNGKey(0))
    ckpt = os.path.join(cfg["output"]["save_dir"], model_id, "model_best.eqx")
    model = eqx.tree_deserialise_leaves(ckpt, model)

    with open(cfg["data"].get("split", "data/split.json")) as f:
        split = json.load(f)
    cfg2 = dict(cfg)
    cfg2["data"] = dict(cfg["data"])
    cfg2["data"]["shots"] = split["train"][:n_shots_max]
    bundles, _, _, _ = load_data(cfg2)

    tr = cfg["training"]
    loss_cfg = LossCfg(
        float(tr["huber_delta"]), float(tr["lambda_src"]), float(tr["src_delta"]),
        float(tr["lambda_z"]), float(tr["lambda_zreg"]), float(tr["lambda_regime"]),
        float(tr["lambda_dalpha"]), float(tr.get("lambda_pH", 0.0)), False,
    )
    imx = tr.get("imex", {})
    imex_cfg = IMEXConfig(float(imx.get("theta", 0.7)), 1e-3, 50000, 1e-4, 1e-6, int(imx.get("substeps", 5)))

    th0 = _theta_from_model(model)

    def mean_loss(th):
        m = _model_with_theta(model, th)
        losses, _, _ = jax.vmap(lambda b: shot_loss_imex(m, b, loss_cfg, imex_cfg))(bundles)
        return jnp.mean(losses)

    n_train = int(bundles.ts_t.shape[0])
    H = jax.hessian(mean_loss)(th0)
    # Covariance ~ inv(n * H) for a mean loss (quasi-likelihood; approximate).
    Hn = np.asarray(H) * n_train
    # regularize tiny negative curvature directions
    w, V = np.linalg.eigh(0.5 * (Hn + Hn.T))
    w = np.clip(w, 1e-6, None)
    cov = (V / w) @ V.T

    J = np.asarray(jax.jacobian(derived)(th0))
    d0 = np.asarray(derived(th0))
    dcov = J @ cov @ J.T
    dse = np.sqrt(np.clip(np.diag(dcov), 0, None))

    beta_se = float(np.sqrt(cov[0, 0]))
    beta0 = float(th0[0])
    from math import erf, sqrt
    p_beta_pos = 0.5 * (1 + erf(beta0 / (beta_se * sqrt(2)))) if beta_se > 0 else float(beta0 > 0)

    return {
        "n_train_shots": n_train,
        "theta_hat": np.asarray(th0).tolist(),
        "beta": beta0, "beta_se": beta_se, "P(beta>0)_laplace": p_beta_pos,
        "derived": {"beta": float(d0[0]), "alpha_fold": float(d0[1]), "P_thr_MW": float(d0[2])},
        "derived_se": {"beta": float(dse[0]), "alpha_fold": float(dse[1]), "P_thr_MW": float(dse[2])},
    }


def shot_bootstrap(report_path, n_boot=2000, seed=0):
    with open(report_path) as f:
        rep = json.load(f)
    rows = []
    for sid, m in rep["shot_metrics"].items():
        rc = m.get("regime_classification", {})
        if rc.get("n_H", 0) > 0 and np.isfinite(rc.get("auc", np.nan)):
            rows.append(rc["auc"])
    rows = np.array(rows)
    if len(rows) < 3:
        return {"n": int(len(rows))}
    rng = np.random.default_rng(seed)
    meds = [float(np.median(rng.choice(rows, len(rows)))) for _ in range(n_boot)]
    return {"n": int(len(rows)), "median_auc": float(np.median(rows)),
            "ci95": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-ids", nargs="+", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--role", default="val")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = {"models": {}}
    for mid in args.model_ids:
        entry = {}
        try:
            entry["laplace"] = laplace(mid, cfg)
        except Exception as e:
            entry["laplace_error"] = str(e)
        rep = os.path.join("logs", mid, "evaluation", f"evaluation_report_{args.role}.json")
        if os.path.exists(rep):
            entry["auc_bootstrap"] = shot_bootstrap(rep)
        out["models"][mid] = entry

    betas = [v["laplace"]["beta"] for v in out["models"].values() if "laplace" in v]
    if betas:
        out["seed_spread_beta"] = {"values": betas, "min": min(betas), "max": max(betas)}
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/uncertainty.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()

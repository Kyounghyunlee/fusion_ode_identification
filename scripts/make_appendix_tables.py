"""Generate the manifest, cohort, and configuration appendix tables.

Writes paper/tables/{manifest,cohorts,hyperparams,ablations}.tex from
data/split.json, data/label_audit.json, data/sanity_summary.csv,
config/config.yaml, and experiments/*.json. Nothing is transcribed by hand.
"""

import csv
import glob
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

OUT = "paper/tables"


def esc(x):
    return str(x).replace("_", r"\_").replace("%", r"\%")


def manifest():
    split = json.load(open("data/split.json"))
    audit = json.load(open("data/label_audit.json"))
    role = {}
    for r in ("train", "val", "test"):
        for s in split[r]:
            role[s] = r
    rows = []
    for p in sorted(glob.glob("data/*_torax_training.npz")):
        shot = int(os.path.basename(p).split("_")[0])
        d = np.load(p, allow_pickle=True)
        reg = d["regime"]
        n_h = int(np.sum(reg == 3))
        tt = float(d["transition_time"])
        cols = int((d["Te_mask"].mean(axis=0) > 0.05).sum())
        gm = str(d["geom_method"]) if "geom_method" in d.files else "?"
        ratio = float(d["geom_V_total_ratio"]) if "geom_V_total_ratio" in d.files else float("nan")
        rows.append((shot, role.get(shot, "--"),
                     audit["shots"].get(str(shot), {}).get("cohort", "--"),
                     cols, int(d["t_ts"].size), n_h,
                     f"{tt:.3f}" if np.isfinite(tt) else "--",
                     f"{ratio:.3f}" if np.isfinite(ratio) else "--",
                     "psi" if gm == "psi_axis_to_lcfs" else esc(gm)))
    lines = [r"\begin{longtable}{rlllrrrrl}", r"\toprule",
             r"shot & split & cohort & $n_\rho$ & $n_t$ & $n_H$ & $t_{\mathrm{LH}}$ [s] & $V$ ratio & geom \\",
             r"\midrule", r"\endhead"]
    for r in rows:
        lines.append(" & ".join(esc(x) for x in r) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    open(f"{OUT}/manifest.tex", "w").write("\n".join(lines))
    return len(rows)


def cohorts():
    audit = json.load(open("data/label_audit.json"))
    split = json.load(open("data/split.json"))
    c = audit["cohorts"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"cohort & all & train & validation & test \\", r"\midrule"]
    for key, name in (("confirmed_transition", "confirmed transition"),
                      ("confirmed_negative", "confirmed negative"),
                      ("ambiguous", "ambiguous")):
        ids = set(c.get(key, []))
        lines.append(f"{name} & {len(ids)} & "
                     f"{len(ids & set(split['train']))} & "
                     f"{len(ids & set(split['val']))} & "
                     f"{len(ids & set(split['test']))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/cohorts.tex", "w").write("\n".join(lines))


def hyperparams():
    import yaml
    cfg = yaml.safe_load(open("config/config.yaml"))
    tr, mo = cfg["training"], cfg["model"]
    try:
        import equinox as eqx, jax
        jax.config.update("jax_enable_x64", True)
        from fusion_ode_identification.model import build_hybrid_model
        m = build_hybrid_model(cfg, jax.random.PRNGKey(0))
        cnt = lambda t: sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array)))
        n_src, n_head = cnt(m.nn), cnt(m.latent.dalpha_head)
        n_lat = cnt(m.latent) - n_head
        n_chi = cnt(m) - cnt(m.nn) - cnt(m.latent)
    except Exception:
        n_src = n_head = n_lat = n_chi = "--"
    rows = [
        ("grid points $N$", 65), ("substeps per observation interval", tr["imex"]["substeps"]),
        (r"$\thIMEX$", tr["imex"]["theta"]),
        ("optimiser", "AdamW"), ("peak learning rate", tr["learning_rate"]),
        ("weight decay", tr["weight_decay"]), ("gradient clip (global norm)", tr["grad_clip"]),
        ("batch size (discharges)", tr["batch_size"]),
        ("step ceiling", tr["total_steps"]),
        ("early-stop patience (evaluations)", tr["early_stop_patience"]),
        ("early-stop relative threshold", tr["early_stop_min_delta"]),
        (r"$\lambda_s$ (source)", tr["lambda_src"]),
        (r"$\lambda_1,\lambda_2$ (latent)", f"{tr['lambda_zreg']}, {tr['lambda_z']}"),
        (r"$\lambda_r$ (regime)", tr["lambda_regime"]),
        (r"$\lambda_y$ (observation)", tr["lambda_dalpha"]),
        ("source network", f"{mo['depth']} hidden layers, width {mo['layers']}, tanh"),
        ("source parameters", f"{n_src:,}" if isinstance(n_src, int) else n_src),
        ("observation-head parameters", f"{n_head:,}" if isinstance(n_head, int) else n_head),
        ("latent vector field + readout parameters", n_lat),
        ("transport-coefficient parameters", n_chi),
        ("precision", "float64"),
    ]
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"setting & value \\", r"\midrule"]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/hyperparams.tex", "w").write("\n".join(lines))


def ablations():
    names = {"e_ext_free_s0": "full model",
             "e_ext_nochi_s0": r"$\Delta\chi = 0$ (no regime$\to$transport coupling)",
             "e_ext_nosrc_s0": "no residual source",
             "e_ext_nodalpha_s0": "no observation loss",
             "e_ext_noregime_s0": "no label loss"}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"configuration & AUC & Brier & $r_{\mathrm{LH}}$ & MAE [eV] \\", r"\midrule"]
    for mid, label in names.items():
        ap = f"experiments/analysis_{mid}_val.json"
        rp = f"logs/{mid}/evaluation/evaluation_report_val.json"
        if not (os.path.exists(ap) and os.path.exists(rp)):
            continue
        a = json.load(open(ap)); r = json.load(open(rp))
        g, e = a["aggregate"], a["events"]
        lines.append(f"{label} & {g['median_auc']:.3f} & {g['median_brier_calibrated']:.3f} & "
                     f"{e['LH']['recall']:.2f} & {r['overall_metrics']['annulus_mean_mae_eV']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/ablations.tex", "w").write("\n".join(lines))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n = manifest(); cohorts(); hyperparams(); ablations()
    print(f"wrote manifest ({n} discharges), cohorts, hyperparams, ablations")

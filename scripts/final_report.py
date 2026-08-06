"""Consolidate all experiment artifacts into paper-ready outputs.

Produces:
  experiments/model_comparison.json  - every run: training, classification,
                                       events, calibration, latent parameters
  paper/tables/comparison.tex        - main model-comparison table
  paper/figures/fig_aggregate.pdf    - held-out distributions (AUC, timing)
  paper/figures/fig_reliability.pdf  - calibration curves
  paper/figures/fig_threshold.pdf    - conditional power-threshold surface
  paper/results.tex                  - macros consumed by main.tex

Usage: python scripts/final_report.py --primary e_ext_free_s0 --role val
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GRAY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b6b", "#c9c9c9"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "figure.dpi": 200,
    "savefig.bbox": "tight",
})

LABELS = {
    "e_ext_free_s": ("extended", "free"), "e_bas_free_s": ("basic", "free"),
    "e_ext_mono_s": ("extended", r"$\beta\leq 0$"),
    "e_ext_nochi_s": ("extended", r"free, $\Delta\chi=0$"),
    "e_ext_nosrc_s": ("extended", "free, no source"),
    "e_ext_nodalpha_s": ("extended", r"free, no $y$ loss"),
    "e_ext_noregime_s": ("extended", "free, no label loss"),
}


def collect(role="val"):
    runs = {}
    for path in sorted(glob.glob(f"experiments/analysis_*_{role}.json")):
        mid = os.path.basename(path)[len("analysis_"):-len(f"_{role}.json")]
        a = json.load(open(path))
        t = json.load(open(f"models/{mid}/train_summary.json")) if os.path.exists(f"models/{mid}/train_summary.json") else {}
        bif = {}
        tau_s = None
        rep_p = f"logs/{mid}/evaluation/evaluation_report_{role}.json"
        if os.path.exists(rep_p):
            rep = json.load(open(rep_p))
            first = next(iter(rep.get("shot_metrics", {}).values()), {})
            tau_s = (first.get("bifurcation") or {}).get("tau_s")
        bfiles = sorted(glob.glob(f"logs/{mid}/evaluation/bifurcation_shot_*.npz"))
        if bfiles:
            z = np.load(bfiles[0])
            bif = {"beta": float(z["c1"]), "tau_s": tau_s,
                   "a_fold_high": float(z["a_fold_high"]), "bistable": bool(np.isfinite(z["a_fold_high"]))}
            sw, fr = [], []
            for f in bfiles:
                zz = np.load(f)
                sw.append(float(zz["z"].max() - zz["z"].min()))
                fr.append(float(np.mean(zz["basin"] > 0)))
            bif["z_swing_median"] = float(np.median(sw))
            bif["frac_shots_reaching_H"] = float(np.mean(np.array(fr) > 0.05))
        runs[mid] = {"train": t, "analysis": a, "latent": bif}
    return runs


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def write_table(runs, out="paper/tables/comparison.tex"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    order = sorted(runs, key=lambda m: (0 if "ext_free" in m else 1 if "bas_free" in m else 2 if "mono" in m else 3, m))
    lines = [
        r"\begin{tabular}{llrrrrrrr}", r"\toprule",
        r"drive & latent & val loss & AUC & F1 & Brier & $r_{\mathrm{LH}}$ & $\beta$ & H-basin \\",
        r"\midrule",
    ]
    for m in order:
        r = runs[m]
        a, e, lt = r["analysis"]["aggregate"], r["analysis"]["events"], r["latent"]
        key = m.rsplit("_s", 1)[0] + "_s"
        drive, lat = LABELS.get(key, ("?", "?"))
        seed = m.rsplit("_s", 1)[-1]
        lines.append(
            f"{drive} & {lat} (s{seed}) & {fmt(r['train'].get('best_val'), 3)} & "
            f"{fmt(a.get('median_auc'))} & {fmt(a.get('median_f1_calibrated'), 2)} & "
            f"{fmt(a.get('median_brier_calibrated'))} & {fmt(e['LH'].get('recall'), 2)} & "
            f"{fmt(lt.get('beta'), 2)} & {fmt(lt.get('frac_shots_reaching_H'), 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(out, "w").write("\n".join(lines))
    return out


def fig_aggregate(runs, primary, out="paper/figures/fig_aggregate.pdf"):
    free = [m for m in runs if "ext_free" in m]
    mono = [m for m in runs if "ext_mono" in m]
    bas = [m for m in runs if "bas_free" in m]

    def aucs(mids):
        v = []
        for m in mids:
            for s, d in runs[m]["analysis"]["per_shot"].items():
                if d["raw"].get("n_H", 0) > 0 and np.isfinite(d["raw"].get("auc", np.nan)):
                    v.append(d["raw"]["auc"])
        return np.array(v)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    groups = [("extended,\nfree", aucs(free), BLUE), ("extended,\n" + r"$\beta\leq0$", aucs(mono), ORANGE),
              ("basic,\nfree", aucs(bas), AQUA)]
    for i, (name, v, col) in enumerate(groups):
        if v.size:
            axes[0].scatter(np.full(v.size, i) + np.linspace(-0.13, 0.13, v.size), v, s=7, color=col, alpha=0.65, lw=0)
            axes[0].hlines(np.median(v), i - 0.25, i + 0.25, color=col, lw=2)
    axes[0].axhline(0.5, color=LIGHT, lw=0.7)
    axes[0].set_xticks(range(3), [g[0] for g in groups])
    axes[0].set_ylabel("per-discharge AUC")
    axes[0].set_ylim(0, 1.05)

    errs = []
    for m in ([primary] if primary in runs else free[:1]):
        for s, d in runs[m]["analysis"]["per_shot"].items():
            errs += [abs(x) * 1e3 for x in d["events"]["LH"]["timing_errors_s"]]
    if errs:
        axes[1].hist(errs, bins=np.arange(0, 45, 5), color=BLUE, alpha=0.85)
    axes[1].set_xlabel("|L$\\to$H timing error| [ms]")
    axes[1].set_ylabel("matched events")

    names, rec = [], []
    for m in sorted(runs):
        key = m.rsplit("_s", 1)[0] + "_s"
        if key in LABELS:
            names.append(LABELS[key][1] + f" ({LABELS[key][0][:3]})")
            rec.append(runs[m]["analysis"]["events"]["LH"]["recall"])
    idx = np.argsort(rec)
    axes[2].barh(np.arange(len(rec)), np.array(rec)[idx], color=BLUE, height=0.7)
    axes[2].set_yticks(np.arange(len(rec)), [names[i] for i in idx], fontsize=6)
    axes[2].set_xlabel(r"L$\to$H recall")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_reliability(runs, primary, out="paper/figures/fig_reliability.pdf"):
    a = runs[primary]["analysis"]
    bins = a.get("reliability_calibrated", [])
    if not bins:
        return
    conf = [b["confidence"] for b in bins]
    freq = [b["frequency"] for b in bins]
    n = [b["n"] for b in bins]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=0.8)
    ax.plot(conf, freq, "o-", color=BLUE, ms=4)
    for c, f, k in zip(conf, freq, n):
        ax.annotate(str(k), (c, f), fontsize=5, color=GRAY, xytext=(2, -6), textcoords="offset points")
    ax.set_xlabel(r"predicted $p_H$ (calibrated)")
    ax.set_ylabel("observed H frequency")
    ax.set_title(f"ECE {a['calibration']['ece_calibrated']:.3f} (raw {a['calibration']['ece_raw']:.3f})", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_threshold(primary, out="paper/figures/fig_threshold.pdf"):
    """Conditional loss-power threshold surface from the identified drive."""
    import equinox as eqx, jax, yaml
    jax.config.update("jax_enable_x64", True)
    from fusion_ode_identification.model import build_hybrid_model

    cfg = yaml.safe_load(open(f"logs/{primary}/config.yaml"))
    m = build_hybrid_model(cfg, jax.random.PRNGKey(0))
    m = eqx.tree_deserialise_leaves(f"models/{primary}/model_best.eqx", m)
    lat = m.latent
    beta = float(lat.beta())
    if beta <= 0:
        return
    af = 2.0 * (beta / 3.0) ** 1.5
    w = np.asarray(jax.nn.softplus(lat.drive_weights[0])), np.asarray(lat.drive_weights[1:])
    kP = float(w[0]); rest = np.asarray(w[1], dtype=float); k0 = float(lat.drive_bias)

    ip = np.linspace(0.4, 0.9, 60)      # MA
    ne = np.linspace(1.0, 4.0, 60)      # 1e19 m^-3
    IP, NE = np.meshgrid(ip, ne)
    other = rest[0] * IP + rest[1] * NE + (rest[2] * 0.2 if rest.size > 2 else 0.0)
    Pthr = (af - other - k0) / max(kP, 1e-9)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    cs = ax.contourf(IP, NE, np.clip(Pthr, 0, 8), levels=12, cmap="Blues")
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label(r"$P_{\mathrm{loss}}$ threshold [MW]")
    ax.set_xlabel(r"$|I_p|$ [MA]")
    ax.set_ylabel(r"$\bar n_e$ [$10^{19}\,$m$^{-3}$]")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_results_tex(runs, primary, role, split, audit, out="paper/results.tex"):
    r = runs[primary]
    a, e, lt = r["analysis"]["aggregate"], r["analysis"]["events"], r["latent"]
    free = [m for m in runs if "ext_free" in m]
    mono = [m for m in runs if "ext_mono" in m]
    betas = [runs[m]["latent"].get("beta") for m in free if runs[m]["latent"].get("beta") is not None]
    v = {
        "nShots": sum(len(split.get(k, [])) for k in ("train", "val", "test")),
        "nTrain": len(split.get("train", [])), "nVal": len(split.get("val", [])),
        "nTest": len(split.get("test", [])),
        "nTrans": len(audit["cohorts"].get("confirmed_transition", [])),
        "nNoTrans": len(audit["cohorts"].get("confirmed_negative", [])),
        "nAmbig": len(audit["cohorts"].get("ambiguous", [])),
        "resBeta": fmt(lt.get("beta"), 2),
        "resBetaRange": f"{min(betas):.2f}--{max(betas):.2f}" if len(betas) > 1 else fmt(lt.get("beta"), 2),
        "resTau": fmt((lt.get("tau_s") or 0) * 1e3, 0),
        "resFoldHigh": fmt(lt.get("a_fold_high"), 3),
        "resAUCmed": fmt(a.get("median_auc"), 2),
        "resAUCci": f"[{a['auc_ci95'][0]:.2f}, {a['auc_ci95'][1]:.2f}]" if a.get("auc_ci95") else "--",
        "resFone": fmt(a.get("median_f1_calibrated"), 2),
        "resBrier": fmt(a.get("median_brier_calibrated"), 3),
        "resECEraw": fmt(r["analysis"]["calibration"]["ece_raw"], 3),
        "resECEcal": fmt(r["analysis"]["calibration"]["ece_calibrated"], 3),
        "resRecallLH": fmt(e["LH"].get("recall"), 2),
        "resPrecLH": fmt(e["LH"].get("precision"), 2),
        "resTTmed": fmt(e["LH"].get("timing_median_ms"), 0),
        "resRecallHL": fmt(e["HL"].get("recall"), 2),
        "resZswing": fmt(lt.get("z_swing_median"), 2),
        "resHbasin": fmt(lt.get("frac_shots_reaching_H"), 2),
        "resAUCmono": fmt(np.median([runs[m]["analysis"]["aggregate"]["median_auc"] for m in mono]), 2) if mono else "--",
        "resValFree": fmt(np.median([runs[m]["train"].get("best_val", np.nan) for m in free]), 3) if free else "--",
        "resValMono": fmt(np.median([runs[m]["train"].get("best_val", np.nan) for m in mono]), 3) if mono else "--",
        "resMAE": "--",
        "resPbetaPos": "--",
        "resRole": role,
        "resPrimary": primary.replace("_", r"\_"),
    }
    rep_p = f"logs/{primary}/evaluation/evaluation_report_{role}.json"
    if os.path.exists(rep_p):
        om = json.load(open(rep_p)).get("overall_metrics", {})
        if "annulus_mean_mae_eV" in om:
            v["resMAE"] = f"\\SI{{{om['annulus_mean_mae_eV']:.0f}}}{{eV}}"
    unc_p = "experiments/uncertainty.json"
    if os.path.exists(unc_p):
        lap = json.load(open(unc_p)).get("models", {}).get(primary, {}).get("laplace", {})
        if "P(beta>0)_laplace" in lap:
            v["resPbetaPos"] = f"{lap['P(beta>0)_laplace']:.2f}"
            v["resBetaSE"] = f"{lap['beta_se']:.2f}"
    try:
        import equinox as eqx, jax, yaml
        jax.config.update("jax_enable_x64", True)
        from fusion_ode_identification.model import build_hybrid_model
        cfg = yaml.safe_load(open("config/config.yaml"))
        mm = build_hybrid_model(cfg, jax.random.PRNGKey(0))
        cnt = lambda tr: sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(tr, eqx.is_inexact_array)))
        v["resNsource"] = f"{cnt(mm.nn):,}"
        v["resNhead"] = f"{cnt(mm.latent.dalpha_head):,}"
    except Exception:
        v["resNsource"] = v["resNhead"] = "--"

    with open(out, "w") as f:
        f.write(f"% Generated by scripts/final_report.py (primary={primary}, role={role}). Do not edit.\n")
        for k, val in v.items():
            f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--role", default="val")
    args = ap.parse_args()

    runs = collect(args.role)
    if args.primary not in runs:
        raise SystemExit(f"{args.primary} has no analysis for role={args.role}")
    json.dump(runs, open("experiments/model_comparison.json", "w"), indent=1, default=float)
    write_table(runs)
    fig_aggregate(runs, args.primary)
    fig_reliability(runs, args.primary)
    try:
        fig_threshold(args.primary)
    except Exception as exc:
        print(f"[warn] threshold figure skipped: {exc}")
    v = write_results_tex(runs, args.primary, args.role,
                          json.load(open("data/split.json")), json.load(open("data/label_audit.json")))
    print(json.dumps(v, indent=1))
    print(f"runs summarized: {len(runs)}")


if __name__ == "__main__":
    main()

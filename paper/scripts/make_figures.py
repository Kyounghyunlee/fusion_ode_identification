"""Publication figures for the paper.

Reads evaluation artifacts (logs/<model_id>/evaluation/bifurcation_shot_*.npz
and evaluation_report.json) and training packs, writes PDF figures into
paper/figures/. Neutral dynamical-systems presentation: the emission proxy is
y(t), the latent is zeta, regimes are L/H.

Usage:
    python paper/scripts/make_figures.py --model-id nf_run_v1 [--shot 27574]
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (light mode), fixed slot order.
BLUE = "#2a78d6"   # slot 1: model / latent
ORANGE = "#eb6834" # slot 2: data / emission proxy
AQUA = "#1baf7a"   # slot 3: secondary series
YELLOW = "#eda100" # slot 4
GRAY = "#6b6b6b"
LIGHT = "#c9c9c9"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _shade_regime(ax, ts, regime, y0=0.0, y1=1.0):
    """Light background bands for labeled H segments."""
    reg = np.asarray(regime)
    in_h = reg > 2.5
    start = None
    for i in range(len(in_h)):
        if in_h[i] and start is None:
            start = ts[i]
        if start is not None and (not in_h[i] or i == len(in_h) - 1):
            ax.axvspan(start, ts[i], color=BLUE, alpha=0.08, lw=0)
            start = None


def load_bif(eval_dir, shot):
    p = os.path.join(eval_dir, f"bifurcation_shot_{shot}.npz")
    return np.load(p) if os.path.exists(p) else None


def fig_data_example(pack_path, out):
    """Example discharge: emission proxy with lower envelope, inputs, labels."""
    d = np.load(pack_path, allow_pickle=True)
    t, y = d["t"], d["D_alpha"]
    fig, axes = plt.subplots(3, 1, figsize=(5.2, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 0.8]})
    _shade_regime(axes[0], t, d["regime"])
    axes[0].plot(t, y, color=ORANGE, lw=0.4)
    axes[0].set_ylabel(r"$y(t)$")

    axes[1].plot(t, d["P_nbi"] / 1e6, color=BLUE, label=r"$w_1$ (power)")
    axes[1].plot(t, np.abs(d["Ip"]) / 1e6, color=AQUA, label=r"$w_2$ (current)")
    axes[1].plot(t, d["nebar"] / 1e19, color=YELLOW, label=r"$w_3$ (density)")
    axes[1].set_ylabel(r"inputs $\mathbf{w}$ (scaled)")
    axes[1].legend(frameon=False, ncol=3, loc="upper left", handlelength=1.2,
                   columnspacing=0.9, borderaxespad=0.0)

    axes[2].step(t, d["regime"], color=GRAY, lw=0.9, where="post")
    axes[2].set_yticks([0, 1, 2, 3], ["--", "L", "T", "H"])
    axes[2].set_ylabel("label")
    axes[2].set_xlabel(r"$t$ [s]")
    tt = float(d["transition_time"])
    if np.isfinite(tt):
        for ax in axes:
            ax.axvline(tt, color=GRAY, lw=0.6, ls=":")
    fig.align_ylabels(axes)
    fig.savefig(out)
    plt.close(fig)


def fig_bifurcation(bif, out):
    """Identified normal form: equilibrium manifold vs drive, trajectory overlaid."""
    c1, c2 = float(bif["c1"]), float(bif["c2"])
    a_fold_low, a_fold_high = float(bif["a_fold_low"]), float(bif["a_fold_high"])
    a_t = bif["a_t"]
    span = max(np.max(np.abs(a_t)), abs(a_fold_high) if np.isfinite(a_fold_high) else 0, 0.5)
    a_grid = np.linspace(-1.4 * span, 1.4 * span, 601)
    stable_lo, stable_hi, unstable = [], [], []
    for a in a_grid:
        roots = np.roots([-1.0, c2, c1, a])
        real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
        if len(real) == 3:
            stable_lo.append((a, real[0])); unstable.append((a, real[1])); stable_hi.append((a, real[2]))
        elif len(real) >= 1:
            r = real[0]
            (stable_hi if r > 0 else stable_lo).append((a, r))

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for arr, ls, lbl in ((stable_lo, "-", "stable branches"), (stable_hi, "-", None), (unstable, "--", "saddle")):
        arr = np.array(arr)
        if arr.size:
            ax.plot(arr[:, 0], arr[:, 1], ls, color=GRAY if ls == "--" else BLUE,
                    lw=1.1 if ls == "-" else 0.9, label=lbl)
    if np.isfinite(a_fold_low) and np.isfinite(a_fold_high):
        for af, lbl in ((a_fold_high, r"$a_{\mathrm{f}}^{+}$"), (a_fold_low, r"$a_{\mathrm{f}}^{-}$")):
            ax.axvline(af, color=LIGHT, lw=0.7)
            ax.annotate(lbl, (af, ax.get_ylim()[0]), textcoords="offset points",
                        xytext=(3, 4), color=GRAY)

    # overlay identified trajectory (a(t), zeta(t)) colored by time
    a_t, z_t, ts = bif["a_t"], bif["z"], bif["ts"]
    sc = ax.scatter(a_t, z_t, c=ts, cmap="Oranges", s=4, lw=0, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, aspect=28)
    cbar.set_label(r"$t$ [s]")
    ax.set_xlabel(r"drive $a(\mathbf{w})$")
    ax.set_ylabel(r"latent $\zeta$")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(out)
    plt.close(fig)


def fig_classification(eval_dir, shots, out):
    """p_H(t) against weak labels for several discharges."""
    rows = len(shots)
    fig, axes = plt.subplots(rows, 1, figsize=(5.2, 1.25 * rows), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, shot in zip(axes, shots):
        bif = load_bif(eval_dir, shot)
        if bif is None:
            ax.set_visible(False)
            continue
        ts, logits, reg = bif["ts"], bif["regime_logits"], bif["regime_ts"]
        _shade_regime(ax, ts, reg)
        ax.plot(ts, _sigmoid(logits), color=BLUE)
        ax.axhline(0.5, color=LIGHT, lw=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"$p_H$\n{shot}")
    axes[-1].set_xlabel(r"$t$ [s]")
    fig.align_ylabels(axes)
    fig.savefig(out)
    plt.close(fig)


def fig_margins(bif, shot, out):
    """Fold margins along a discharge."""
    ts = bif["ts"]
    a_t = bif["a_t"]
    a_fold_low, a_fold_high = float(bif["a_fold_low"]), float(bif["a_fold_high"])
    if not (np.isfinite(a_fold_low) and np.isfinite(a_fold_high)):
        return  # identified field is monostable: no fold margins to plot
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    ax.plot(ts, a_fold_high - a_t, color=BLUE, label=r"$\mu_{\mathrm{LH}}$")
    ax.plot(ts, a_t - a_fold_low, color=ORANGE, label=r"$\mu_{\mathrm{HL}}$")
    ax.axhline(0.0, color=GRAY, lw=0.6, ls=":")
    _shade_regime(ax, ts, bif["regime_ts"])
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel("fold margin")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out)
    plt.close(fig)


def fig_fit(eval_dir, shot, out):
    """Measured vs modeled temperature at three radii + barrier trajectory."""
    p = os.path.join(eval_dir, f"fit_shot_{shot}.npz")
    if not os.path.exists(p):
        return
    d = np.load(p)
    ts, rho, Tm, To, mask = d["ts"], d["rho"], d["Te_model"], d["Te_obs"], d["mask"]
    # the three best-covered interior radii, displayed inner to outer
    cov = mask[:, :-1].mean(axis=0)
    picks = sorted(np.argsort(cov)[::-1][:3].tolist())

    fig, axes = plt.subplots(len(picks) + 1, 1, figsize=(5.2, 1.15 * (len(picks) + 1)), sharex=True)
    for ax, j in zip(axes[:-1], picks):
        obs = np.where(mask[:, j] > 0.5, To[:, j], np.nan)
        ax.plot(ts, obs, ".", color=ORANGE, ms=2.5, label="measured")
        ax.plot(ts, Tm[:, j], color=BLUE, lw=1.0, label="model")
        ax.set_ylabel(rf"$u(\rho={rho[j]:.2f})$")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[-1].plot(ts, d["z_barrier"], color=BLUE)
    axes[-1].set_ylabel(r"$p_H$")
    axes[-1].set_ylim(-0.05, 1.05)
    axes[-1].set_xlabel(r"$t$ [s]")
    fig.align_ylabels(axes)
    fig.savefig(out)
    plt.close(fig)


def fig_chi(out, chi_core=0.6, chi_edge_base=2.0, chi_edge_drop=1.0, ped_center=0.85, ped_width=0.08):
    """Transport-coefficient family chi(rho, p_H)."""
    rho = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    for pH, shade in ((0.0, 0.25), (0.5, 0.55), (1.0, 1.0)):
        chi_edge = np.clip(chi_edge_base - chi_edge_drop * pH, 0.1, 5.0)
        w = 1.0 / (1.0 + np.exp(-(rho - ped_center) / ped_width))
        chi = chi_core + w * (chi_edge - chi_core)
        ax.plot(rho, chi, color=BLUE, alpha=shade, label=rf"$p_H={pH:.1f}$")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\chi(\rho, \zeta)$")
    ax.legend(frameon=False)
    fig.savefig(out)
    plt.close(fig)


def fig_labeler(pack_path, out):
    """Anatomy of the weak labels: proxy, lower envelope, threshold, segments."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from preprocessing.build_training_pack import _rolling_quantile, _otsu_threshold, interp_fill_1d

    d = np.load(pack_path, allow_pickle=True)
    t, y, ip, ne = d["t"], d["D_alpha"].astype(float), d["Ip"], d["nebar"]
    base = _rolling_quantile(interp_fill_1d(t, y), t, 0.015, 0.15)
    gate = (np.abs(ip) > 0.6 * np.nanpercentile(np.abs(ip), 95)) & (ne > 0.2 * np.nanpercentile(ne, 95))
    lo, hi = np.nanpercentile(base[gate], [1, 99])
    norm = np.clip((base - lo) / (hi - lo), 0, 1)
    thr = _otsu_threshold(norm[gate])
    thr_abs = lo + thr * (hi - lo)

    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    _shade_regime(ax, t, d["regime"])
    ax.plot(t, y, lw=0.3, color=ORANGE, alpha=0.55, label=r"proxy $y(t)$")
    ax.plot(t, base, lw=1.1, color=AQUA, label="lower envelope")
    tg = np.where(gate, thr_abs, np.nan)
    ax.plot(t, tg, lw=0.9, color=GRAY, ls="--", label="threshold (gated)")
    tt = float(d["transition_time"])
    if np.isfinite(tt):
        ax.axvline(tt, color=GRAY, lw=0.6, ls=":")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$y(t)$")
    ax.legend(frameon=False, ncol=3, fontsize=7)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="nf_run_v1")
    ap.add_argument("--shot", type=int, default=None, help="example shot for data/bifurcation figures")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    eval_dir = os.path.join("logs", args.model_id, "evaluation")
    os.makedirs(FIGDIR, exist_ok=True)

    bif_files = sorted(glob.glob(os.path.join(eval_dir, "bifurcation_shot_*.npz")))
    shots = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in bif_files]
    # showcase = transitioning shots ranked by classification F1
    report_path = os.path.join(eval_dir, "evaluation_report.json")
    f1_by_shot = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            rep_all = json.load(f)
        for sid, m in rep_all.get("shot_metrics", {}).items():
            rc = m.get("regime_classification", {}) or {}
            if rc.get("n_H", 0) > 0:
                f1_by_shot[int(sid)] = rc.get("f1", 0.0)
    showcase = []
    for s in shots:
        bif = load_bif(eval_dir, s)
        if bif is not None and np.any(bif["regime_ts"] > 2.5):
            showcase.append(s)
    showcase.sort(key=lambda s: -(f1_by_shot.get(s, 0.0)))
    example = args.shot or (showcase[0] if showcase else (shots[0] if shots else None))
    if example is None:
        raise SystemExit("No evaluation artifacts found; run scripts/evaluate_model.py first.")

    fig_data_example(os.path.join(args.data_dir, f"{example}_torax_training.npz"),
                     os.path.join(FIGDIR, "fig_data.pdf"))
    bif = load_bif(eval_dir, example)
    if bif is not None:
        fig_bifurcation(bif, os.path.join(FIGDIR, "fig_bifurcation.pdf"))
        fig_margins(bif, example, os.path.join(FIGDIR, "fig_margins.pdf"))
    fig_classification(eval_dir, showcase[:4] if showcase else shots[:4],
                       os.path.join(FIGDIR, "fig_classification.pdf"))
    fig_fit(eval_dir, example, os.path.join(FIGDIR, "fig_fit.pdf"))
    fig_chi(os.path.join(FIGDIR, "fig_chi.pdf"))
    fig_labeler(os.path.join(args.data_dir, f"{example}_torax_training.npz"),
                os.path.join(FIGDIR, "fig_labeler.pdf"))

    report_path = os.path.join(eval_dir, "evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            rep = json.load(f)
        summ = rep.get("regime_classification_summary", {})
        print("classification summary:", json.dumps(summ, indent=2))
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()

"""Publication figures for the paper.

Reads evaluation artifacts (logs/<model_id>/evaluation/bifurcation_shot_*.npz
and evaluation_report.json) and training packs, writes PDF figures into
paper/figures/. Neutral dynamical-systems presentation: the emission proxy is
y(t), the latent is zeta, regimes are L/H.

Usage:
    python paper/scripts/make_figures.py --model-id cusp_run_v1 [--shot 27574]
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
    """Identified cusp: equilibrium manifold vs drive, trajectory overlaid."""
    b, a_fold = float(bif["b"]), float(bif["a_fold"])
    a_grid = np.linspace(-1.9 * a_fold, 1.9 * a_fold, 601)
    stable_lo, stable_hi, unstable = [], [], []
    for a in a_grid:
        roots = np.roots([-1.0, 0.0, b, a])
        real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
        if len(real) == 3:
            stable_lo.append((a, real[0])); unstable.append((a, real[1])); stable_hi.append((a, real[2]))
        else:
            r = real[0]
            (stable_hi if r > 0 else stable_lo).append((a, r))

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for arr, ls, lbl in ((stable_lo, "-", "stable branches"), (stable_hi, "-", None), (unstable, "--", "saddle")):
        arr = np.array(arr)
        if arr.size:
            ax.plot(arr[:, 0], arr[:, 1], ls, color=GRAY if ls == "--" else BLUE,
                    lw=1.1 if ls == "-" else 0.9, label=lbl)
    ax.annotate("H", (0, np.sqrt(b)), textcoords="offset points", xytext=(-10, 4), color=BLUE)
    ax.annotate("L", (0, -np.sqrt(b)), textcoords="offset points", xytext=(6, -10), color=BLUE)
    for af in (+a_fold, -a_fold):
        ax.axvline(af, color=LIGHT, lw=0.7)
    ax.annotate(r"$a_{\mathrm{f}}$", (a_fold, ax.get_ylim()[0]), textcoords="offset points",
                xytext=(3, 4), color=GRAY)
    ax.annotate(r"$-a_{\mathrm{f}}$", (-a_fold, ax.get_ylim()[0]), textcoords="offset points",
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
    a_t, a_fold = bif["a_t"], float(bif["a_fold"])
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    ax.plot(ts, a_fold - a_t, color=BLUE, label=r"$\mu_{\mathrm{LH}}$")
    ax.plot(ts, a_t + a_fold, color=ORANGE, label=r"$\mu_{\mathrm{HL}}$")
    ax.axhline(0.0, color=GRAY, lw=0.6, ls=":")
    _shade_regime(ax, ts, bif["regime_ts"])
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel("fold margin")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="cusp_run_v1")
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

    report_path = os.path.join(eval_dir, "evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            rep = json.load(f)
        summ = rep.get("regime_classification_summary", {})
        print("classification summary:", json.dumps(summ, indent=2))
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()

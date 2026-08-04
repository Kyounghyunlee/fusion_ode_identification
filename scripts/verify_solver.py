"""Numerical verification of the IMEX theta-method transport solver.

Tests (all must pass before identified parameters are interpreted):
 1. analytic decay      - Neumann(0)/Dirichlet(1) cosine mode, V'=const:
                          T(rho,t) = cos(k rho) exp(-chi k^2 t), k = pi/2.
 2. grid convergence    - spatial error vs N on the decay problem.
 3. substep convergence - temporal error vs substeps (theta = 0.7 -> order 1).
 4. flux balance        - discrete energy change equals boundary flux integral.
 5. positivity          - nonnegative initial data stays nonnegative.
 6. gradient check      - autodiff through the rollout vs central finite
                          differences for physical scalars and an NN weight.

Writes a JSON report and a convergence figure used by the paper appendix.

Usage: JAX_PLATFORMS=cpu python scripts/verify_solver.py
"""

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

from fusion_ode_identification.imex_solver import (
    IMEXIntegrator,
    apply_diffusion_explicit,
    build_diffusion_solve_tridiag_implicit,
    cell_volumes,
)

OUT_DIR = "logs/solver_verification"
CHI0 = 1.0
THETA = 0.7


class ConstChiModel(eqx.Module):
    """Minimal model exposing the IMEX interface with constant chi and
    zero source/latent dynamics, for solver verification."""

    chi0: float = CHI0
    Te_scale: float = 1.0

    def build_diffusion_matrix_imex(self, t, z, args, dt, theta=1.0):
        rho, Vprime = args[0], args[1]
        chi = jnp.full_like(rho, self.chi0)
        a, b, c, b_bc = build_diffusion_solve_tridiag_implicit(
            rho, Vprime, chi, dt, theta,
            dr=args[5], Vprime_face=args[6], Vprime_cell=args[7], denom=args[8],
        )
        return a, b, c, b_bc, chi

    def compute_source_imex(self, t, Te_total, z, args):
        return jnp.zeros(Te_total.shape[0] - 1)

    def compute_latent_rhs_imex(self, t, z, args):
        return 0.0


def _geom_args(rho):
    Vprime = jnp.ones_like(rho)
    dr = jnp.diff(rho)
    from fusion_ode_identification.imex_solver import cell_volumes as _cv
    Vf = 0.5 * (Vprime[:-1] + Vprime[1:])
    denom = jnp.maximum(_cv(rho, Vprime, dr), 1e-10)
    return (rho, Vprime, None, None, None, dr, Vf, Vf, denom)


def run_decay(N, substeps, t_end=0.05, n_save=26):
    """Integrate the analytic cosine decay mode; return L2 error at t_end."""
    rho = jnp.linspace(0.0, 1.0, N)
    k = np.pi / 2.0
    # Physical magnitude (eV): the integrator's smooth positivity safeguard
    # is exact to ~0.01 eV, so the test must run at operating scale.
    T0 = 1000.0 * jnp.cos(k * rho)
    saveat = jnp.linspace(0.0, t_end, n_save)
    integ = IMEXIntegrator(theta=THETA, substeps=substeps)
    model = ConstChiModel()
    ny = N - 1
    y0 = jnp.concatenate([T0[:-1], jnp.array([0.0])])
    zeros = jnp.zeros((n_save,))
    sol = integ.integrate(
        t_span=(0.0, t_end), y0=y0, saveat=saveat, model=model,
        Te_edge_ts=zeros,                       # Dirichlet T(1,t) = 0
        ctrl_norm_ts=jnp.zeros((n_save, 4)),
        ne_ts=jnp.zeros((n_save, N)),
        latent_features_ts=jnp.zeros((n_save, 4)),
        args=_geom_args(rho),
    )
    T_num = np.asarray(sol.ys[-1, :ny])
    T_exact = 1000.0 * np.cos(k * np.asarray(rho[:-1])) * np.exp(-CHI0 * k**2 * t_end)
    err = float(np.sqrt(np.mean((T_num - T_exact) ** 2))) / 1000.0
    return err


def flux_balance(N=65):
    """Discrete conservation: sum_i vol_i * div_i equals net boundary flux."""
    rho = jnp.linspace(0.0, 1.0, N)
    Vprime = jnp.ones_like(rho)
    T = 100.0 * (1.0 - rho**2) + 10.0
    chi = jnp.full_like(rho, CHI0)
    div = apply_diffusion_explicit(rho, Vprime, chi, T)
    dr = jnp.diff(rho)
    vol = cell_volumes(rho, Vprime, dr)
    total = float(jnp.sum(vol * div))
    # boundary flux at the last face (axis face flux is zero by construction)
    grad_edge = (T[-1] - T[-2]) / dr[-1]
    F_edge = float(-0.5 * (Vprime[-1] + Vprime[-2]) * CHI0 * grad_edge)
    residual = abs(total - (-F_edge)) / max(abs(F_edge), 1e-12)
    return {"volume_integral": total, "boundary_flux": -F_edge, "relative_residual": residual}


def positivity(N=65, substeps=5):
    rho = jnp.linspace(0.0, 1.0, N)
    T0 = jnp.where(rho < 0.5, 0.0, 100.0)  # nonnegative, sharp
    n_save = 21
    saveat = jnp.linspace(0.0, 0.05, n_save)
    integ = IMEXIntegrator(theta=THETA, substeps=substeps)
    sol = integ.integrate(
        t_span=(0.0, 0.05), y0=jnp.concatenate([T0[:-1], jnp.array([0.0])]),
        saveat=saveat, model=ConstChiModel(),
        Te_edge_ts=jnp.full((n_save,), 100.0),
        ctrl_norm_ts=jnp.zeros((n_save, 4)), ne_ts=jnp.zeros((n_save, N)),
        latent_features_ts=jnp.zeros((n_save, 4)), args=_geom_args(rho),
    )
    return {"min_value": float(jnp.min(sol.ys[:, : N - 1])), "pass": bool(jnp.min(sol.ys[:, : N - 1]) > -1e-8)}


def gradient_check():
    """Autodiff through one real training rollout vs central differences."""
    import yaml
    from fusion_ode_identification.data import load_data
    from fusion_ode_identification.model import build_hybrid_model
    from fusion_ode_identification.loss import shot_loss_imex
    from fusion_ode_identification.types import LossCfg, IMEXConfig

    cfg = yaml.safe_load(open("config/config.yaml"))
    cfg["data"]["shots"] = [27574]
    bundles, _, _, _ = load_data(cfg)
    bundle = jax.tree_util.tree_map(lambda x: x[0], bundles)
    model = build_hybrid_model(cfg, jax.random.PRNGKey(0))
    loss_cfg = LossCfg(1.0, 0.1, 1.0, 1e-3, 1e-3, 0.5, 2.0, 0.0, False)
    imex_cfg = IMEXConfig(THETA, 1e-3, 50000, 1e-4, 1e-6, 5)

    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(p):
        m = eqx.combine(p, static)
        loss, _, _ = shot_loss_imex(m, bundle, loss_cfg, imex_cfg)
        return loss

    grads = jax.grad(loss_fn)(params)

    checks = {}
    probes = [
        ("latent.beta_raw", lambda p: p.latent.beta_raw, 1e-5),
        ("latent.tau_raw", lambda p: p.latent.tau_raw, 1e-5),
        ("latent.drive_bias", lambda p: p.latent.drive_bias, 1e-5),
        ("chi_edge_gap_raw", lambda p: p.chi_edge_gap_raw, 1e-5),
    ]
    for name, get, eps in probes:
        g_ad = float(get(grads))
        p_plus = eqx.tree_at(get, params, get(params) + eps)
        p_minus = eqx.tree_at(get, params, get(params) - eps)
        g_fd = float((loss_fn(p_plus) - loss_fn(p_minus)) / (2 * eps))
        denom = max(abs(g_ad), abs(g_fd), 1e-12)
        checks[name] = {"autodiff": g_ad, "finite_diff": g_fd, "rel_err": abs(g_ad - g_fd) / denom}

    # one NN weight
    w = params.nn.mlp.layers[-1].weight
    idx = (0, 0)
    eps = 1e-5
    g_ad = float(grads.nn.mlp.layers[-1].weight[idx])
    for sgn, tag in ((+1, "p"), (-1, "m")):
        pass
    wp = eqx.tree_at(lambda p: p.nn.mlp.layers[-1].weight, params, w.at[idx].add(eps))
    wm = eqx.tree_at(lambda p: p.nn.mlp.layers[-1].weight, params, w.at[idx].add(-eps))
    g_fd = float((loss_fn(wp) - loss_fn(wm)) / (2 * eps))
    denom = max(abs(g_ad), abs(g_fd), 1e-12)
    checks["nn.last_layer.w[0,0]"] = {"autodiff": g_ad, "finite_diff": g_fd, "rel_err": abs(g_ad - g_fd) / denom}
    return checks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"theta": THETA}

    grids = [17, 33, 65, 129]
    report["grid_convergence"] = {str(N): run_decay(N, substeps=40) for N in grids}
    errs = [report["grid_convergence"][str(N)] for N in grids]
    report["spatial_order_estimate"] = float(np.polyfit(np.log([1 / (N - 1) for N in grids]), np.log(errs), 1)[0])

    subs = [1, 2, 5, 10, 20]
    report["substep_convergence"] = {str(m): run_decay(129, substeps=m) for m in subs}

    report["flux_balance"] = flux_balance()
    report["positivity"] = positivity()
    report["gradient_checks"] = gradient_check()

    with open(os.path.join(OUT_DIR, "verification_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))
    h = [1 / (N - 1) for N in grids]
    axes[0].loglog(h, errs, "o-", color="#2a78d6")
    axes[0].loglog(h, [errs[-1] * (x / h[-1]) ** 2 for x in h], "--", color="#6b6b6b", label=r"$O(h^2)$")
    axes[0].set_xlabel(r"$\Delta\rho$"); axes[0].set_ylabel("L2 error"); axes[0].legend(frameon=False)
    sub_errs = [report["substep_convergence"][str(m)] for m in subs]
    dt = [1.0 / m for m in subs]
    axes[1].loglog(dt, sub_errs, "o-", color="#2a78d6")
    axes[1].loglog(dt, [sub_errs[0] * (x / dt[0]) for x in dt], "--", color="#6b6b6b", label=r"$O(\Delta t)$")
    axes[1].set_xlabel("substep size (rel.)"); axes[1].set_ylabel("L2 error"); axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "convergence.pdf"))
    fig.savefig("paper/figures/fig_convergence.pdf", bbox_inches="tight")

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

"""Model definitions: transport backbone + canonical regime latent.

The latent regime coordinate z(t) obeys the depressed cubic

    tau * dz/dt = alpha(r) + beta * z - z^3,

the canonical (translation-reduced) form of the scalar cubic family: a
quadratic term can always be removed by shifting z, with the shift absorbed
into the drive intercept and observation offsets, so only beta is an
independently meaningful topology parameter. beta is unconstrained: the
data may support beta <= 0 (single equilibrium branch, finite-rate lag but
no static multistability) or beta > 0 (two folds, static hysteresis).

The drive alpha(r) is affine in the causal input vector r = [P_nbi, |Ip|,
nebar] in fixed physical units (MW, MA, 1e19 m^-3), monotone non-decreasing
in power. No per-shot statistics enter the model, so inference is causal.
The D-alpha proxy never enters the dynamics; it supervises a learned
observation head. The residual source network deliberately does NOT see z,
so regime dependence of the profiles can only arise through the transport
coefficient chi(rho, b) - removing the interpretability bypass.
"""

from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx


CONTROL_NAMES = ["P_nbi", "Ip", "nebar", "D_alpha"]

# Fixed physical scales for control inputs (W, A, m^-3, a.u.). Frozen;
# identical for every discharge and available online.
CONTROL_SCALES = (1.0e6, 1.0e6, 1.0e19, 2.0)

TAU_MIN = 5.0e-3  # s; numerical floor, below the ~4 ms observation cadence


def _as64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def softclip(x, limit):
    limit = jnp.asarray(limit, dtype=jnp.float64)
    return limit * (x / (limit + jnp.abs(x)))


def smooth_clamp(x, lo, hi, beta: float = 50.0):
    """Smoothly clamp x into [lo, hi] with nonzero gradients near the bounds."""
    x = jnp.asarray(x)
    lo = jnp.asarray(lo, dtype=x.dtype)
    hi = jnp.asarray(hi, dtype=x.dtype)
    beta = jnp.asarray(beta, dtype=x.dtype)
    x1 = lo + jax.nn.softplus(beta * (x - lo)) / beta
    x2 = hi - jax.nn.softplus(beta * (hi - x1)) / beta
    return x2


def normalize_observed_signal(x: jnp.ndarray, default: float = 0.5) -> jnp.ndarray:
    """Range-normalize a measured trace to [0, 1] (supervision targets only;
    never used on model inputs)."""
    x = _as64(x)
    x_min = jnp.min(x)
    span = jnp.max(x) - x_min
    scaled = (x - x_min) / (span + 1.0e-6)
    return jnp.where(span > 1.0e-6, scaled, jnp.full_like(x, float(default)))


class SourceNN(eqx.Module):
    """Residual source s(rho, Te, ne, r); zero-initialized; regime-blind.

    z is intentionally NOT an input: regime dependence of the profile must
    flow through chi(rho, b), keeping the transport interpretation testable.
    """

    mlp: eqx.nn.MLP
    source_scale: float

    def __init__(self, key, source_scale: float = 1.0, layers: int = 64, depth: int = 3):
        in_size = 1 + 1 + 1 + len(CONTROL_NAMES)  # rho, Te, ne, controls
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=layers,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].weight, self.mlp, jnp.zeros_like(self.mlp.layers[-1].weight))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].bias, self.mlp, jnp.zeros_like(self.mlp.layers[-1].bias))
        self.source_scale = float(source_scale)

    def __call__(self, rho, Te_val, ne_val, controls):
        x = jnp.concatenate(
            [
                jnp.atleast_1d(rho),
                jnp.atleast_1d(Te_val),
                jnp.atleast_1d(ne_val),
                jnp.asarray(controls),
            ],
            axis=0,
        )
        return self.mlp(x)[0] * self.source_scale


class NormalFormLatent(eqx.Module):
    """Depressed-cubic regime latent with monotone barrier readout.

    tau * dz/dt = alpha(r) + beta*z - z^3.

    beta_mode:
      "free"        - beta = beta_raw (unconstrained; sign decided by data)
      "nonpositive" - beta = -softplus(beta_raw) (constrained monostable)
    """

    drive_weights: jnp.ndarray
    drive_bias: jnp.ndarray
    beta_raw: jnp.ndarray
    tau_raw: jnp.ndarray
    kb_raw: jnp.ndarray
    bb: jnp.ndarray
    dalpha_head: eqx.nn.MLP
    beta_mode: str = eqx.field(static=True)

    N_DRIVE = 3  # P_nbi (command), Ip and nebar (measured context)

    def __init__(self, key, beta_mode: str = "free"):
        key_w, key_head = jax.random.split(key)
        self.drive_weights = jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float64) + (
            jax.random.normal(key_w, (self.N_DRIVE,), dtype=jnp.float64) * 0.01
        )
        self.drive_bias = jnp.array(-0.5, dtype=jnp.float64)
        self.beta_raw = jnp.array(0.3, dtype=jnp.float64)
        self.tau_raw = jnp.array(-3.9, dtype=jnp.float64)  # softplus + TAU_MIN ~ 25 ms
        self.kb_raw = jnp.array(1.5, dtype=jnp.float64)    # softplus + 0.5 -> k_b > 0.5
        self.bb = jnp.array(0.0, dtype=jnp.float64)
        self.beta_mode = str(beta_mode)
        self.dalpha_head = eqx.nn.MLP(
            in_size=1 + len(CONTROL_NAMES) + 2,
            out_size=1,
            width_size=16,
            depth=1,
            activation=jax.nn.tanh,
            key=key_head,
        )

    # -- canonical quantities --

    def beta(self) -> jnp.ndarray:
        if self.beta_mode == "nonpositive":
            return -jax.nn.softplus(self.beta_raw)
        return self.beta_raw

    def tau_eff(self) -> jnp.ndarray:
        return jax.nn.softplus(self.tau_raw) + TAU_MIN

    def drive(self, latent_features: jnp.ndarray) -> jnp.ndarray:
        """alpha(r): affine in physically scaled inputs; monotone in power."""
        feat = _as64(latent_features)
        w_power = jax.nn.softplus(self.drive_weights[0])
        raw = (
            w_power * feat[0]
            + self.drive_weights[1] * feat[1]
            + self.drive_weights[2] * feat[2]
            + self.drive_bias
        )
        return softclip(raw, 5.0)

    def __call__(self, z: float, latent_features: jnp.ndarray) -> float:
        zeta = _as64(z)
        alpha = self.drive(latent_features)
        rhs = (alpha + self.beta() * zeta - zeta**3) / self.tau_eff()
        return softclip(rhs, 1.0e3)

    def initial_state(self, latent_features_0: jnp.ndarray) -> jnp.ndarray:
        """Causal initialization: the lowest equilibrium of the vector field
        at the initial drive (every gated discharge starts in state L).

        Newton iteration on f(z) = alpha0 + beta*z - z^3 from a bracket left
        of all roots; static iteration count keeps this jit-differentiable.
        """
        alpha0 = self.drive(latent_features_0)
        beta = self.beta()
        z_start = -(1.0 + jnp.sqrt(jnp.abs(beta)) + jnp.abs(alpha0) ** (1.0 / 3.0))

        def newton(_, z):
            f = alpha0 + beta * z - z**3
            fp = beta - 3.0 * z**2
            fp = jnp.where(jnp.abs(fp) < 1e-8, -1e-8, fp)
            return z - f / fp

        return jax.lax.fori_loop(0, 60, newton, z_start)

    # -- readouts --

    def regime_logit(self, z: float) -> float:
        k_b = jax.nn.softplus(self.kb_raw) + 0.5
        return k_b * _as64(z) + self.bb

    def barrier_coordinate(self, z: float) -> float:
        """Soft barrier activation b(z) in [0, 1]; also the soft regime
        coordinate reported as p_H (its probabilistic calibration is
        measured, not assumed)."""
        return jax.nn.sigmoid(self.regime_logit(z))

    def aux_dalpha_hat(self, z: float, control_norm: jnp.ndarray, Te_edge: float, ne_edge: float) -> float:
        x = jnp.concatenate(
            [
                jnp.atleast_1d(self.barrier_coordinate(z)),
                jnp.asarray(control_norm, dtype=jnp.float64),
                jnp.atleast_1d(_as64(Te_edge) / 1000.0),
                jnp.atleast_1d(_as64(ne_edge) / 1e19),
            ]
        )
        return jax.nn.sigmoid(self.dalpha_head(x)[0])


def build_hybrid_model(cfg, key) -> "HybridField":
    model_cfg = cfg.get("model", {})
    layers = int(model_cfg.get("layers", 64))
    depth = int(model_cfg.get("depth", 3))
    source_scale = float(model_cfg.get("source_scale", 3.0e5))
    divergence_clip = float(model_cfg.get("divergence_clip", 1.0e6))
    beta_mode = str(model_cfg.get("beta_mode", "free"))
    delta_chi_off = bool(model_cfg.get("delta_chi_off", False))
    source_off = bool(model_cfg.get("source_off", False))

    key_nn, key_latent = jax.random.split(key)
    return HybridField(
        nn=SourceNN(key_nn, source_scale=0.0 if source_off else source_scale, layers=layers, depth=depth),
        latent=NormalFormLatent(key_latent, beta_mode=beta_mode),
        divergence_clip=divergence_clip,
        delta_chi_off=delta_chi_off,
    )


class HybridField(eqx.Module):
    """1D flux-coordinate temperature-diffusion backbone + regime latent.

    Phenomenological: the evolved quantity is the electron temperature
    directly; density, heat capacity, and coordinate-motion factors of a
    full energy balance are absorbed into chi and the residual source.
    """

    nn: SourceNN
    latent: Any

    Te_scale: float = 1000.0
    ne_scale: float = 1e19
    chi_core_raw: jnp.ndarray
    chi_edge_H_raw: jnp.ndarray
    chi_edge_gap_raw: jnp.ndarray
    divergence_clip: jnp.ndarray
    ped_center: float = 0.85
    ped_width: float = 0.08
    delta_chi_off: bool = eqx.field(static=True)

    def __init__(
        self,
        nn: SourceNN,
        latent: Any,
        chi_core: float = 0.6,
        chi_edge_L: float = 2.0,
        chi_edge_H: float = 1.0,
        divergence_clip: float = 1.0e6,
        delta_chi_off: bool = False,
    ):
        # softplus-inverse init so constraints hold by construction:
        # chi_core > 0, chi_edge_H > 0, chi_edge_L = chi_edge_H + gap > chi_edge_H.
        def _inv_softplus(y):
            import numpy as np
            return float(np.log(np.expm1(y)))

        self.nn = nn
        self.latent = latent
        self.chi_core_raw = jnp.array(_inv_softplus(chi_core), dtype=jnp.float64)
        self.chi_edge_H_raw = jnp.array(_inv_softplus(chi_edge_H), dtype=jnp.float64)
        self.chi_edge_gap_raw = jnp.array(_inv_softplus(chi_edge_L - chi_edge_H), dtype=jnp.float64)
        self.divergence_clip = jnp.array(divergence_clip, dtype=jnp.float64)
        self.delta_chi_off = bool(delta_chi_off)

    # -- chi parameters (positive, ordered by construction) --

    def chi_core(self):
        return jax.nn.softplus(self.chi_core_raw)

    def chi_edge_H(self):
        return jax.nn.softplus(self.chi_edge_H_raw)

    def chi_edge_L(self):
        return self.chi_edge_H() + jax.nn.softplus(self.chi_edge_gap_raw)

    def barrier_coordinate(self, z):
        return self.latent.barrier_coordinate(z)

    def compute_regime_logit(self, z):
        return self.latent.regime_logit(z)

    def compute_aux_dalpha_hat(self, z, control_norm, Te_edge, ne_edge):
        return self.latent.aux_dalpha_hat(z, control_norm, Te_edge, ne_edge)

    def _chi_profile(self, rho, z):
        b = jnp.where(self.delta_chi_off, 0.0, self.barrier_coordinate(z))
        chi_edge = (1.0 - b) * self.chi_edge_L() + b * self.chi_edge_H()
        w_ped = jax.nn.sigmoid((rho - self.ped_center) / self.ped_width)
        return self.chi_core() + w_ped * (chi_edge - self.chi_core())

    # -- IMEX interface --

    def build_diffusion_matrix_imex(self, t, z, args, dt, theta=1.0):
        """Tridiagonal coefficients of (I - theta*dt*L) plus boundary coupling.

        chi depends only on (rho, z) so the implicit operator is linear in T.
        args: (rho, Vprime, ctrl_norm, ne, latent_inputs, dr, Vprime_face,
        Vprime_cell, denom).
        """
        from .imex_solver import build_diffusion_solve_tridiag_implicit

        rho = _as64(args[0])
        Vprime = jnp.clip(_as64(args[1]), 1e-6, None)
        chi = self._chi_profile(rho, z)
        a, b, c, b_bc = build_diffusion_solve_tridiag_implicit(
            rho, Vprime, chi, dt, theta,
            dr=args[5], Vprime_face=args[6], Vprime_cell=args[7], denom=args[8],
        )
        return a, b, c, b_bc, chi

    def compute_source_from_values(self, rho, Te_total, z, ne_vals, control_norm):
        """Explicit NN source on interior nodes (z unused by design)."""
        del z
        rho = _as64(rho)
        ne_vals = jnp.clip(_as64(ne_vals), 1e17, 1e21)
        control_norm = jnp.clip(_as64(control_norm), -10.0, 10.0)
        S_nn = jax.vmap(
            lambda r, T, n: self.nn(r, T / self.Te_scale, n / self.ne_scale, control_norm)
        )(rho[:-1], Te_total[:-1], ne_vals[:-1])
        return S_nn

    def compute_divergence_from_values(self, rho, Vprime, Te_total, z):
        """Conservative-form diffusion divergence on interior nodes (diagnostics)."""
        from .imex_solver import apply_diffusion_explicit

        rho = _as64(rho)
        Vprime = jnp.clip(_as64(Vprime), 1e-6, None)
        chi = self._chi_profile(rho, z)
        divergence = apply_diffusion_explicit(rho, Vprime, chi, Te_total)
        return softclip(divergence, self.divergence_clip)

    def compute_source_imex(self, t, Te_total, z, args):
        return self.compute_source_from_values(args[0], Te_total, z, args[3], args[2])

    def compute_latent_rhs_imex(self, t, z, args):
        return self.latent(z, _as64(args[4]))

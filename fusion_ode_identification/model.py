"""Model definitions: transport backbone + normal-form regime latent.

The latent confinement state z(t) obeys the full unfolding of the cubic
normal form,

    tau * dz/dt = a(u) + c1 * z + c2 * z^2 - z^3,

with unconstrained c1, c2: the data decides whether the identified vector
field is bistable (c2^2 + 3*c1 > 0: folds, hysteresis) or monostable (a
smooth threshold response with no memory). The drive a(u) depends only on
actuators in fixed physical units, so the model is causal and the same
actuator setting means the same drive in every discharge.
"""

from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx


CONTROL_NAMES = ["P_nbi", "Ip", "nebar", "D_alpha"]

# Fixed physical scales for all control inputs (W, A, m^-3, a.u.). Used both
# for the source network and the latent drive; no per-shot statistics enter
# the model, which keeps inference causal / real-time capable.
CONTROL_SCALES = (1.0e6, 1.0e6, 1.0e19, 2.0)


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
    """Range-normalize a measured trace to [0, 1] (supervision targets only)."""
    x = _as64(x)
    x_min = jnp.min(x)
    span = jnp.max(x) - x_min
    scaled = (x - x_min) / (span + 1.0e-6)
    return jnp.where(span > 1.0e-6, scaled, jnp.full_like(x, float(default)))


class SourceNN(eqx.Module):
    """Residual source closure s(rho, Te, ne, u, z), zero-initialized."""

    mlp: eqx.nn.MLP
    source_scale: float

    def __init__(self, key, source_scale: float = 1.0, layers: int = 64, depth: int = 3):
        in_size = 1 + 1 + 1 + len(CONTROL_NAMES) + 1  # rho, Te, ne, controls, z
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

    def __call__(self, rho, Te_val, ne_val, controls, z):
        x = jnp.concatenate(
            [
                jnp.atleast_1d(rho),
                jnp.atleast_1d(Te_val),
                jnp.atleast_1d(ne_val),
                jnp.asarray(controls),
                jnp.atleast_1d(z),
            ],
            axis=0,
        )
        return self.mlp(x)[0] * self.source_scale


class NormalFormLatent(eqx.Module):
    """Regime latent: full unfolding of the cubic normal form.

    tau * dz/dt = a(u) + c1 * z + c2 * z^2 - z^3

    The drive a(u) is affine in the physically scaled actuators (P, Ip,
    nebar), monotone non-decreasing in power. The measured emission proxy
    (D-alpha) never enters the dynamics; it is predicted by a learned
    observation head, so the latent is identified rather than pinned to a
    proxy. The calibrated regime probability p_H = sigmoid(k*z + k0) also
    serves as the barrier coordinate that closes the transport loop.
    """

    drive_weights: jnp.ndarray
    drive_bias: jnp.ndarray
    c1: jnp.ndarray
    c2: jnp.ndarray
    tau_raw: jnp.ndarray
    logit_gain_raw: jnp.ndarray
    logit_bias: jnp.ndarray
    dalpha_head: eqx.nn.MLP

    N_DRIVE = 3  # P_nbi, Ip, nebar; D_alpha excluded by design

    def __init__(self, key):
        key_w, key_head = jax.random.split(key)
        # Features are physically scaled (P in MW, Ip in MA, ne in 1e19).
        # Init: unheated flat-top sits at a ~ -0.5; ~3 MW reaches a ~ +0.4.
        self.drive_weights = jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float64) + (
            jax.random.normal(key_w, (self.N_DRIVE,), dtype=jnp.float64) * 0.01
        )
        self.drive_bias = jnp.array(-0.5, dtype=jnp.float64)
        # Mildly bistable-capable start; sign and size are free to change.
        self.c1 = jnp.array(0.5, dtype=jnp.float64)
        self.c2 = jnp.array(0.0, dtype=jnp.float64)
        self.tau_raw = jnp.array(-3.9, dtype=jnp.float64)  # softplus + 5e-3 -> ~25 ms
        self.logit_gain_raw = jnp.array(1.5, dtype=jnp.float64)
        self.logit_bias = jnp.array(0.0, dtype=jnp.float64)
        self.dalpha_head = eqx.nn.MLP(
            in_size=1 + len(CONTROL_NAMES) + 2,
            out_size=1,
            width_size=16,
            depth=1,
            activation=jax.nn.tanh,
            key=key_head,
        )

    # -- normal-form quantities --

    def tau_eff(self) -> jnp.ndarray:
        return jax.nn.softplus(self.tau_raw) + 5.0e-3

    def drive(self, latent_features: jnp.ndarray) -> jnp.ndarray:
        """Affine drive over physically scaled actuators; monotone in power."""
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
        a = self.drive(latent_features)
        rhs = (a + self.c1 * zeta + self.c2 * zeta**2 - zeta**3) / self.tau_eff()
        return softclip(rhs, 1.0e3)

    # -- observation / classification heads --

    def regime_logit(self, z: float) -> float:
        k = jax.nn.softplus(self.logit_gain_raw) + 0.5
        return k * _as64(z) + self.logit_bias

    def barrier_coordinate(self, z: float) -> float:
        """p_H in [0, 1]; also modulates the edge transport coefficient."""
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

    key_nn, key_latent = jax.random.split(key)
    return HybridField(
        nn=SourceNN(key_nn, source_scale=source_scale, layers=layers, depth=depth),
        latent=NormalFormLatent(key_latent),
        divergence_clip=divergence_clip,
    )


class HybridField(eqx.Module):
    """1D conservative diffusion + residual source + normal-form latent."""

    nn: SourceNN
    latent: Any

    Te_scale: float = 1000.0
    ne_scale: float = 1e19
    chi_core: jnp.ndarray
    chi_edge_base: jnp.ndarray
    chi_edge_drop: jnp.ndarray
    divergence_clip: jnp.ndarray
    ped_center: float = 0.85
    ped_width: float = 0.08

    def __init__(
        self,
        nn: SourceNN,
        latent: Any,
        chi_core: float = 0.6,
        chi_edge_base: float = 2.0,
        chi_edge_drop: float = 1.0,
        divergence_clip: float = 1.0e6,
    ):
        self.nn = nn
        self.latent = latent
        self.chi_core = jnp.array(chi_core, dtype=jnp.float64)
        self.chi_edge_base = jnp.array(chi_edge_base, dtype=jnp.float64)
        self.chi_edge_drop = jnp.array(chi_edge_drop, dtype=jnp.float64)
        self.divergence_clip = jnp.array(divergence_clip, dtype=jnp.float64)

    def barrier_coordinate(self, z):
        return self.latent.barrier_coordinate(z)

    def compute_regime_logit(self, z):
        return self.latent.regime_logit(z)

    def compute_aux_dalpha_hat(self, z, control_norm, Te_edge, ne_edge):
        return self.latent.aux_dalpha_hat(z, control_norm, Te_edge, ne_edge)

    def _chi_profile(self, rho, z):
        chi_edge = self.chi_edge_base - self.chi_edge_drop * self.barrier_coordinate(z)
        chi_edge = jnp.clip(chi_edge, 0.1, 5.0)
        w_ped = jax.nn.sigmoid((rho - self.ped_center) / self.ped_width)
        return self.chi_core + w_ped * (chi_edge - self.chi_core)

    # -- IMEX interface --

    def build_diffusion_matrix_imex(self, t, z, args, dt, theta=1.0):
        """Tridiagonal coefficients of (I - theta*dt*L) plus boundary coupling.

        The implicit operator must be linear in T, so chi depends only on
        (rho, z). args carries precomputed geometry:
        (rho, Vprime, ctrl_norm, ne, latent_inputs, dr, Vprime_face,
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
        """Explicit NN source on interior nodes from already-sampled inputs."""
        rho = _as64(rho)
        ne_vals = jnp.clip(_as64(ne_vals), 1e17, 1e21)
        control_norm = jnp.clip(_as64(control_norm), -10.0, 10.0)
        S_nn = jax.vmap(
            lambda r, T, n: self.nn(r, T / self.Te_scale, n / self.ne_scale, control_norm, z)
        )(rho[:-1], Te_total[:-1], ne_vals[:-1])
        return S_nn

    def compute_divergence_from_values(self, rho, Vprime, Te_total, z):
        """Conservative diffusion divergence on interior nodes (diagnostics)."""
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

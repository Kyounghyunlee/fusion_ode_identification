"""Model definitions for tokamak electron temperature ODE."""
# fusion_ode_identification/model.py

from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx


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

CONTROL_NAMES = ["P_nbi", "Ip", "nebar", "D_alpha"]
LATENT_FEATURE_SIZE = 6

# Fixed physical scales for the cusp drive features (W, A, m^-3, a.u.).
# The drive map must see the SAME value for the same actuator setting in
# every shot, otherwise a shared fold threshold cannot exist; per-shot
# z-scoring (used for the source NN) breaks that, so the cusp latent uses
# globally scaled raw controls instead.
CONTROL_SCALES = (1.0e6, 1.0e6, 1.0e19, 1.0)


def build_cusp_drive_features(ctrl_vals_ts: jnp.ndarray) -> jnp.ndarray:
    """Raw controls on the profile time base, in fixed physical units."""
    scales = jnp.asarray(CONTROL_SCALES, dtype=jnp.float64)
    return _as64(ctrl_vals_ts) / scales


def _as64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _moving_average_same(x: jnp.ndarray, width: int) -> jnp.ndarray:
    width = max(1, int(width))
    if width <= 1 or x.shape[0] < 3:
        return _as64(x)
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    x_pad = jnp.concatenate([jnp.repeat(x[:1], pad_left), x, jnp.repeat(x[-1:], pad_right)], axis=0)
    kernel = jnp.ones((width,), dtype=jnp.float64) / float(width)
    return jnp.convolve(_as64(x_pad), kernel, mode="valid")


def _normalize_range(x: jnp.ndarray, default: float = 0.5) -> jnp.ndarray:
    x = _as64(x)
    x_min = jnp.min(x)
    span = jnp.max(x) - x_min
    scaled = (x - x_min) / (span + 1.0e-6)
    return jnp.where(span > 1.0e-6, scaled, jnp.full_like(x, float(default)))


def normalize_observed_signal(x: jnp.ndarray) -> jnp.ndarray:
    return _normalize_range(x)


def _time_derivative(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    x = _as64(x)
    t = _as64(t)
    if x.shape[0] < 2:
        return jnp.zeros_like(x)
    dx = jnp.diff(x)
    dt = jnp.maximum(jnp.diff(t), 1.0e-6)
    if x.shape[0] == 2:
        slope = dx[0] / dt[0]
        return jnp.array([slope, slope], dtype=jnp.float64)
    center = (x[2:] - x[:-2]) / jnp.maximum(t[2:] - t[:-2], 1.0e-6)
    first = dx[0] / dt[0]
    last = dx[-1] / dt[-1]
    return jnp.concatenate([jnp.array([first], dtype=jnp.float64), center, jnp.array([last], dtype=jnp.float64)])


def build_latent_feature_series(
    ts: jnp.ndarray,
    ctrl_norm_ts: jnp.ndarray,
    dalpha_ts: jnp.ndarray,
    Te_edge_ts: jnp.ndarray,
    ne_edge_ts: jnp.ndarray,
) -> jnp.ndarray:
    ts = _as64(ts)
    ctrl_norm_ts = _as64(ctrl_norm_ts)
    dalpha_norm = normalize_observed_signal(dalpha_ts)
    Te_edge_norm = normalize_observed_signal(Te_edge_ts)
    ne_edge_norm = normalize_observed_signal(ne_edge_ts)

    dalpha_s = _moving_average_same(dalpha_norm, 11)
    Te_edge_s = _moving_average_same(Te_edge_norm, 11)
    ne_edge_s = _moving_average_same(ne_edge_norm, 11)

    dalpha_hmode_evidence = 1.0 - dalpha_s
    d_dalpha = -_time_derivative(dalpha_s, ts)
    d_Te_edge = _time_derivative(Te_edge_s, ts)
    d_ne_edge = _time_derivative(ne_edge_s, ts)
    P_nbi = ctrl_norm_ts[:, 0]
    Ip = ctrl_norm_ts[:, 1]
    return jnp.stack([dalpha_hmode_evidence, d_dalpha, d_Te_edge, d_ne_edge, P_nbi, Ip], axis=-1)


class SourceNN(eqx.Module):
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
        # zero-init final layer for stability
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].weight, self.mlp, jnp.zeros_like(self.mlp.layers[-1].weight))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].bias, self.mlp, jnp.zeros_like(self.mlp.layers[-1].bias))
        self.source_scale = float(source_scale)

    def __call__(self, rho, Te_val, ne_val, controls, z):
        # Do not force float64 here; let dtype follow model/inputs.
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


class LatentDynamics(eqx.Module):
    alpha: jnp.ndarray
    beta: jnp.ndarray
    gamma: jnp.ndarray
    mu_weights: jnp.ndarray
    mu_bias: jnp.ndarray
    mu_ref: jnp.ndarray

    def __call__(self, z: float, controls: jnp.ndarray) -> float:
        mu = jnp.dot(controls[:3], self.mu_weights) + self.mu_bias
        alpha_eff = jax.nn.softplus(self.alpha)
        beta_eff = jax.nn.softplus(self.beta)
        gamma_eff = jax.nn.softplus(self.gamma)
        return alpha_eff * (mu - self.mu_ref) - beta_eff * z - gamma_eff * z**3

    def barrier_coordinate(self, z: float, latent_gain: float = 1.0) -> float:
        return jax.nn.sigmoid(_as64(latent_gain) * _as64(z))

    def regime_logit(self, z: float, latent_gain: float = 1.0) -> float:
        return _as64(latent_gain) * _as64(z)

    def aux_dalpha_hat(self, z: float, control_norm: jnp.ndarray, Te_edge: float, ne_edge: float, latent_gain: float = 1.0) -> float:
        del control_norm, Te_edge, ne_edge
        return self.barrier_coordinate(z, latent_gain=latent_gain)


class BarrierLatentDynamics(eqx.Module):
    target_extra_weights_raw: jnp.ndarray
    target_bias: jnp.ndarray
    evidence_gain_raw: jnp.ndarray
    tau_lh_raw: jnp.ndarray
    tau_hl_raw: jnp.ndarray
    regime_gain: jnp.ndarray

    def __init__(self, key):
        del key
        self.target_extra_weights_raw = jnp.full((LATENT_FEATURE_SIZE - 1,), -5.0, dtype=jnp.float64)
        self.target_bias = jnp.array(0.0, dtype=jnp.float64)
        self.evidence_gain_raw = jnp.array(5.5, dtype=jnp.float64)
        self.tau_lh_raw = jnp.array(-3.0, dtype=jnp.float64)
        self.tau_hl_raw = jnp.array(-2.2, dtype=jnp.float64)
        self.regime_gain = jnp.array(1.0, dtype=jnp.float64)

    def __call__(self, z: float, latent_features: jnp.ndarray) -> float:
        feat = _as64(latent_features)
        zeta = _as64(z)
        z_b = self.barrier_coordinate(zeta)
        dalpha_evidence = jnp.clip(feat[0], 0.0, 1.0)
        extra = feat[1:]
        extra_drive = jnp.dot(jax.nn.softplus(self.target_extra_weights_raw), extra)
        evidence_gain = jax.nn.softplus(self.evidence_gain_raw) + 1.0
        target_logit = evidence_gain * (dalpha_evidence - 0.5) + 0.1 * extra_drive + self.target_bias
        target_logit = jnp.clip(target_logit, -8.0, 8.0)
        target_barrier = jax.nn.sigmoid(target_logit)
        tau_lh = jax.nn.softplus(self.tau_lh_raw) + 5.0e-3
        tau_hl = jax.nn.softplus(self.tau_hl_raw) + 5.0e-3
        tau = jnp.where(target_barrier >= z_b, tau_lh, tau_hl)
        return (target_logit - zeta) / tau

    def barrier_coordinate(self, z: float, latent_gain: float = 1.0) -> float:
        del latent_gain
        return jax.nn.sigmoid(_as64(z))

    def regime_logit(self, z: float, latent_gain: float = 1.0) -> float:
        del latent_gain
        return self.regime_gain * _as64(z)

    def aux_dalpha_hat(self, z: float, control_norm: jnp.ndarray, Te_edge: float, ne_edge: float, latent_gain: float = 1.0) -> float:
        del control_norm, Te_edge, ne_edge, latent_gain
        return 1.0 - self.barrier_coordinate(z)


class CuspLatentDynamics(eqx.Module):
    """Bistable cusp normal-form latent for the confinement state.

    tau * dz/dt = a(u) + b * z - z**3,   b > 0

    For |a| < a_fold = 2*(b/3)**1.5 the system is bistable: a lower stable
    branch (L regime, z < 0) and an upper stable branch (H regime, z > 0)
    separated by an unstable middle equilibrium. Sweeping the actuator drive
    a(u) through +a_fold destroys the L branch (forced L->H transition,
    saddle-node bifurcation); sweeping back through -a_fold destroys the H
    branch (H->L back-transition). Hysteresis is intrinsic to the normal form.

    The drive a(u) is an affine map of normalized actuator controls only
    (P_nbi, Ip, nebar). The D-alpha measurement never enters the dynamics;
    it is predicted by a separate observation head, so the latent is
    identified from data rather than pinned to a proxy signal.
    """

    drive_weights: jnp.ndarray
    drive_bias: jnp.ndarray
    b_raw: jnp.ndarray
    tau_raw: jnp.ndarray
    regime_gain_raw: jnp.ndarray
    dalpha_head: eqx.nn.MLP

    N_DRIVE = 3  # P_nbi, Ip, nebar (normalized); D_alpha excluded by design

    def __init__(self, key):
        key_w, key_head = jax.random.split(key)
        # Features are physically scaled (P in MW, Ip in MA, ne in 1e19).
        # Init so that an unheated flat-top sits below the fold (a ~ -0.5)
        # and full beam power (~3 MW) reaches it: softplus(-1) * 3 ~ 0.9.
        self.drive_weights = jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float64) + (
            jax.random.normal(key_w, (self.N_DRIVE,), dtype=jnp.float64) * 0.01
        )
        self.drive_bias = jnp.array(-0.5, dtype=jnp.float64)
        self.b_raw = jnp.array(0.55, dtype=jnp.float64)      # softplus -> b ~ 1.0
        self.tau_raw = jnp.array(-3.9, dtype=jnp.float64)    # softplus + 5e-3 -> tau ~ 25 ms
        self.regime_gain_raw = jnp.array(1.5, dtype=jnp.float64)
        # Small observation head: D_alpha_hat = sigmoid(MLP(z_b, controls, Te_edge, ne_edge))
        self.dalpha_head = eqx.nn.MLP(
            in_size=1 + len(CONTROL_NAMES) + 2,
            out_size=1,
            width_size=16,
            depth=1,
            activation=jax.nn.tanh,
            key=key_head,
        )

    # -- normal-form quantities (all closed-form) --

    def b_eff(self) -> jnp.ndarray:
        return jax.nn.softplus(self.b_raw) + 1e-3

    def tau_eff(self) -> jnp.ndarray:
        return jax.nn.softplus(self.tau_raw) + 5.0e-3

    def z_scale(self) -> jnp.ndarray:
        """Amplitude of the stable branches at a=0: z* = +/- sqrt(b)."""
        return jnp.sqrt(self.b_eff())

    def a_fold(self) -> jnp.ndarray:
        """Fold amplitude: bistability holds for |a| < a_fold."""
        b = self.b_eff()
        return 2.0 * (b / 3.0) ** 1.5

    def drive(self, latent_features: jnp.ndarray) -> jnp.ndarray:
        """Affine drive over physically scaled actuators (P, Ip, nebar).

        Monotonicity prior: the drive is non-decreasing in injected power
        (softplus on the power weight); the current and density weights are
        unconstrained in sign.
        """
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
        rhs = (a + self.b_eff() * zeta - zeta**3) / self.tau_eff()
        return softclip(rhs, 1.0e3)

    # -- observation / classification heads --

    def barrier_coordinate(self, z: float, latent_gain: float = 1.0) -> float:
        # Scale-invariant barrier coordinate: branches map to ~0.05 / ~0.95.
        return jax.nn.sigmoid(_as64(latent_gain) * 3.0 * _as64(z) / self.z_scale())

    def regime_logit(self, z: float, latent_gain: float = 1.0) -> float:
        del latent_gain
        k = jax.nn.softplus(self.regime_gain_raw) + 0.5
        return k * _as64(z) / self.z_scale()

    def aux_dalpha_hat(self, z: float, control_norm: jnp.ndarray, Te_edge: float, ne_edge: float, latent_gain: float = 1.0) -> float:
        z_b = self.barrier_coordinate(z, latent_gain=latent_gain)
        x = jnp.concatenate(
            [
                jnp.atleast_1d(z_b),
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
    latent_gain = float(model_cfg.get("latent_gain", 1.0))
    source_scale = float(model_cfg.get("source_scale", 3.0e5))
    divergence_clip = float(model_cfg.get("divergence_clip", 1.0e6))
    latent_design = str(model_cfg.get("latent_design", "cubic")).lower()

    key_nn, key_latent = jax.random.split(key)
    if latent_design == "cubic":
        latent = LatentDynamics(
            alpha=jnp.array(1.0, dtype=jnp.float64),
            beta=jnp.array(1.0, dtype=jnp.float64),
            gamma=jnp.array(1.0, dtype=jnp.float64),
            mu_weights=jax.random.normal(key_latent, (3,), dtype=jnp.float64) * 0.01,
            mu_bias=jnp.array(0.0, dtype=jnp.float64),
            mu_ref=jnp.array(0.0, dtype=jnp.float64),
        )
    elif latent_design == "barrier_v1":
        latent = BarrierLatentDynamics(key_latent)
    elif latent_design == "cusp":
        latent = CuspLatentDynamics(key_latent)
    else:
        raise ValueError(f"Unknown model.latent_design={latent_design!r}")

    return HybridField(
        nn=SourceNN(key_nn, source_scale=source_scale, layers=layers, depth=depth),
        latent=latent,
        latent_gain=latent_gain,
        divergence_clip=divergence_clip,
    )


class HybridField(eqx.Module):
    nn: SourceNN
    latent: Any
    latent_gain: jnp.ndarray

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
        latent_gain: float = 1.0,
        chi_core: float = 0.6,
        chi_edge_base: float = 2.0,
        chi_edge_drop: float = 1.0,
        divergence_clip: float = 1.0e6,
    ):
        self.nn = nn
        self.latent = latent
        self.latent_gain = jnp.array(latent_gain, dtype=jnp.float64)
        self.chi_core = jnp.array(chi_core, dtype=jnp.float64)
        self.chi_edge_base = jnp.array(chi_edge_base, dtype=jnp.float64)
        self.chi_edge_drop = jnp.array(chi_edge_drop, dtype=jnp.float64)
        self.divergence_clip = jnp.array(divergence_clip, dtype=jnp.float64)

    def __call__(self, t, y, args):
        dTe_hat_dt, z_dot, _div_raw, _src_raw = self.compute_rhs_components(t, y, args)

        rhs = jnp.concatenate([dTe_hat_dt, jnp.array([z_dot], dtype=jnp.float64)])
        rhs = jnp.where(jnp.isfinite(rhs), rhs, 0.0)
        return rhs

    def uses_barrier_latent(self) -> bool:
        return isinstance(self.latent, BarrierLatentDynamics)

    def uses_cusp_latent(self) -> bool:
        return isinstance(self.latent, CuspLatentDynamics)

    def barrier_coordinate(self, z):
        return self.latent.barrier_coordinate(z, latent_gain=self.latent_gain)

    def compute_regime_logit(self, z):
        return self.latent.regime_logit(z, latent_gain=self.latent_gain)

    def compute_aux_dalpha_hat(self, z, control_norm, Te_edge, ne_edge):
        return self.latent.aux_dalpha_hat(z, control_norm, Te_edge, ne_edge, latent_gain=self.latent_gain)

    def _chi_profile(self, rho, z):
        chi_edge = self.chi_edge_base - self.chi_edge_drop * self.barrier_coordinate(z)
        chi_edge = jnp.clip(chi_edge, 0.1, 5.0)
        w_ped = jax.nn.sigmoid((rho - self.ped_center) / self.ped_width)
        return self.chi_core + w_ped * (chi_edge - self.chi_core)

    def _conservative_divergence(self, rho, Vprime, chi, Te_total):
        # Expect rho, Vprime, chi, Te_total length N (including boundary).
        dr = jnp.diff(rho)
        dr = jnp.clip(dr, 1e-6 * jnp.max(dr) + 1e-12, None)

        grad_T = jnp.diff(Te_total) / dr
        chi_face = 0.5 * (chi[:-1] + chi[1:])
        Vprime_face = 0.5 * (Vprime[:-1] + Vprime[1:])
        flux_face = -Vprime_face * chi_face * grad_T

        flux_in = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64), flux_face[:-1]])
        flux_out = flux_face

        Vprime_cell = 0.5 * (Vprime[:-1] + Vprime[1:])
        vol = Vprime_cell * dr
        vol_floor = jnp.maximum(1e-4 * jnp.max(vol), 1e-10)
        denom = jnp.maximum(vol, vol_floor)

        divergence = -(flux_out - flux_in) / denom
        divergence = softclip(divergence, self.divergence_clip)
        return divergence, vol, dr

    def _control_norm(self, t, ctrl_interp, control_means, control_stds):
        control_vals = ctrl_interp.evaluate(t)
        control_norm = (control_vals - control_means) / (control_stds + 1e-6)
        return jnp.clip(control_norm, -10.0, 10.0)

    def compute_physics_tendency(self, t, Te_total, z, args):
        (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_interp) = args
        rho = _as64(rho_vals)
        Vprime = jnp.clip(_as64(Vprime_vals), 1e-6, None)
        chi = self._chi_profile(rho, z)

        divergence, _vol, _dr = self._conservative_divergence(rho, Vprime, chi, Te_total)

        control_norm = self._control_norm(t, ctrl_interp, control_means, control_stds)
        ne_vals = jnp.clip(ne_interp.evaluate(t), 1e17, 1e21)

        S_nn = jax.vmap(
            lambda r, T, n: self.nn(r, T / self.Te_scale, n / self.ne_scale, control_norm, z)
        )(rho[:-1], Te_total[:-1], ne_vals[:-1])

        return divergence + S_nn

    def compute_divergence_only(self, t, Te_total, z, args):
        (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_interp) = args
        rho = _as64(rho_vals)
        Vprime = jnp.clip(_as64(Vprime_vals), 1e-6, None)
        chi = self._chi_profile(rho, z)
        divergence, _vol, _dr = self._conservative_divergence(rho, Vprime, chi, Te_total)
        return divergence

    def compute_source(self, t, Te_total, z, args):
        (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_interp) = args
        rho = _as64(rho_vals)

        control_norm = self._control_norm(t, ctrl_interp, control_means, control_stds)
        ne_vals = jnp.clip(ne_interp.evaluate(t), 1e17, 1e21)

        S_nn = jax.vmap(
            lambda r, T, n: self.nn(r, T / self.Te_scale, n / self.ne_scale, control_norm, z)
        )(rho[:-1], Te_total[:-1], ne_vals[:-1])
        return S_nn

    # -------- Fast (interp-free) helpers for IMEX --------

    def compute_source_from_values(self, rho, Te_total, z, ne_vals, control_norm):
        """Compute explicit NN source on interior nodes from already-sampled inputs."""
        rho = _as64(rho)
        ne_vals = jnp.clip(_as64(ne_vals), 1e17, 1e21)
        control_norm = jnp.clip(_as64(control_norm), -10.0, 10.0)
        S_nn = jax.vmap(
            lambda r, T, n: self.nn(r, T / self.Te_scale, n / self.ne_scale, control_norm, z)
        )(rho[:-1], Te_total[:-1], ne_vals[:-1])
        return S_nn

    def compute_divergence_from_values(self, rho, Vprime, Te_total, z):
        """Compute conservative diffusion divergence on interior nodes."""
        rho = _as64(rho)
        Vprime = jnp.clip(_as64(Vprime), 1e-6, None)
        chi = self._chi_profile(rho, z)
        divergence, _vol, _dr = self._conservative_divergence(rho, Vprime, chi, Te_total)
        return divergence

    def compute_rhs_components(self, t, y, args):
        (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_interp) = args

        Te_hat = smooth_clamp(y[:-1], 0.0, 5000.0 / self.Te_scale)
        z = smooth_clamp(y[-1], -10.0, 10.0)

        Te_bc = Te_bc_interp.evaluate(t)
        Te_total = jnp.append(Te_hat * self.Te_scale, Te_bc)
        Te_total = smooth_clamp(Te_total, 0.0, 5000.0)

        div = self.compute_divergence_only(t, Te_total, z, args)
        src = self.compute_source(t, Te_total, z, args)

        limit = 1e4
        total_clip = softclip(div + src, limit)

        control_norm = self._control_norm(t, ctrl_interp, control_means, control_stds)
        z_dot = self.latent(z, control_norm)

        dTe_hat_dt = total_clip / self.Te_scale
        return dTe_hat_dt, z_dot, div, src
    
    # ========== IMEX Interface Methods ==========
    
    def build_diffusion_matrix_imex(self, t, z, args, dt, theta=1.0):
        """Build implicit diffusion solve coefficients for IMEX.

        Clean split requirement: the implicit operator must be linear in T.
        Therefore chi must not depend on T (only on rho and latent z).

        Returns:
            a,b,c: (N-1,) tridiagonal coefficients for (I - theta*dt*L)
            b_bc: (N-1,) boundary coupling vector (multiplied by T_edge)
            chi: (N,) diffusivity profile
        """
        from .imex_solver import build_diffusion_solve_tridiag_implicit

        # Support both legacy args (with interpolants) and new fast args.
        rho_vals = args[0]
        Vprime_vals = args[1]
        rho = _as64(rho_vals)
        Vprime = jnp.clip(_as64(Vprime_vals), 1e-6, None)
        chi = self._chi_profile(rho, z)

        # Optional precomputed geometry.
        # Legacy fast args: (rho, Vprime, ctrl_norm, ne, dr, Vprime_face, Vprime_cell, denom)
        # Barrier-latent fast args: (rho, Vprime, ctrl_norm, ne, latent_inputs, dr, Vprime_face, Vprime_cell, denom)
        if len(args) >= 9:
            dr = args[5]
            Vprime_face = args[6]
            Vprime_cell = args[7]
            denom = args[8]
            a, b, c, b_bc = build_diffusion_solve_tridiag_implicit(
                rho,
                Vprime,
                chi,
                dt,
                theta,
                dr=dr,
                Vprime_face=Vprime_face,
                Vprime_cell=Vprime_cell,
                denom=denom,
            )
        elif len(args) >= 8:
            dr = args[4]
            Vprime_face = args[5]
            Vprime_cell = args[6]
            denom = args[7]
            a, b, c, b_bc = build_diffusion_solve_tridiag_implicit(
                rho,
                Vprime,
                chi,
                dt,
                theta,
                dr=dr,
                Vprime_face=Vprime_face,
                Vprime_cell=Vprime_cell,
                denom=denom,
            )
        else:
            a, b, c, b_bc = build_diffusion_solve_tridiag_implicit(rho, Vprime, chi, dt, theta)
        return a, b, c, b_bc, chi
    
    def compute_source_imex(self, t, Te_total, z, args):
        """
        Compute explicit source term for IMEX (S_net on interior nodes).
        """
        # New fast args: (rho, Vprime, control_norm, ne_vals)
        if len(args) >= 4:
            rho_vals = args[0]
            control_norm = args[2]
            ne_vals = args[3]
            return self.compute_source_from_values(rho_vals, Te_total, z, ne_vals, control_norm)
        return self.compute_source(t, Te_total, z, args)
    
    def compute_latent_rhs_imex(self, t, z, args):
        """
        Compute dz/dt for explicit latent evolution in IMEX.
        """
        # New fast args: (rho, Vprime, control_norm, ne_vals, latent_inputs)
        if len(args) >= 5:
            return self.latent(z, _as64(args[4]))
        if len(args) >= 4:
            control_norm = args[2]
            control_norm = jnp.clip(_as64(control_norm), -10.0, 10.0)
            return self.latent(z, control_norm)

        (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_interp) = args
        control_norm = self._control_norm(t, ctrl_interp, control_means, control_stds)
        return self.latent(z, control_norm)

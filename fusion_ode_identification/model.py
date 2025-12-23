"""Model definitions for tokamak electron temperature ODE."""
# fusion_ode_identification/model.py

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

CONTROL_NAMES = ["P_nbi", "Ip", "nebar", "S_gas", "S_rec", "S_nbi"]


def _as64(x):
    return jnp.asarray(x, dtype=jnp.float64)


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


class HybridField(eqx.Module):
    nn: SourceNN
    latent: LatentDynamics
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
        latent: LatentDynamics,
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

    def _chi_profile(self, rho, z):
        chi_edge = self.chi_edge_base - self.chi_edge_drop * jax.nn.sigmoid(self.latent_gain * z)
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
    
    def build_diffusion_matrix_imex(self, t, Te_total, z, args, dt, theta=1.0):
        """
        Build implicit diffusion matrix and boundary coupling for IMEX.
        
        Returns:
            A: (N-1, N-1) matrix for implicit solve
            b_bc: (N-1,) boundary coupling vector
            chi: (N,) diffusivity profile
        """
        from .imex_solver import build_diffusion_matrix_implicit
        
        (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_interp) = args
        rho = _as64(rho_vals)
        Vprime = jnp.clip(_as64(Vprime_vals), 1e-6, None)
        chi = self._chi_profile(rho, z)
        
        A, b_bc = build_diffusion_matrix_implicit(rho, Vprime, chi, dt, theta)
        return A, b_bc, chi
    
    def compute_source_imex(self, t, Te_total, z, args):
        """
        Compute explicit source term for IMEX (S_net on interior nodes).
        """
        return self.compute_source(t, Te_total, z, args)
    
    def compute_latent_rhs_imex(self, t, z, args):
        """
        Compute dz/dt for explicit latent evolution in IMEX.
        """
        (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_interp) = args
        control_norm = self._control_norm(t, ctrl_interp, control_means, control_stds)
        return self.latent(z, control_norm)

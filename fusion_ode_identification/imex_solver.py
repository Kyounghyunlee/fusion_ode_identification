"""
IMEX (Implicit-Explicit) time-stepping for stiff transport equations.

Implements theta-method for diffusion (implicit) + explicit source terms.
Properly handles Neumann BC at axis and Dirichlet BC at edge.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Callable, Tuple, NamedTuple, Optional
import equinox as eqx


def smooth_clamp(x, lo, hi, beta: float = 50.0):
    x = jnp.asarray(x)
    lo = jnp.asarray(lo, dtype=x.dtype)
    hi = jnp.asarray(hi, dtype=x.dtype)
    beta = jnp.asarray(beta, dtype=x.dtype)
    x1 = lo + jax.nn.softplus(beta * (x - lo)) / beta
    x2 = hi - jax.nn.softplus(beta * (hi - x1)) / beta
    return x2


class IMEXState(NamedTuple):
    """State during IMEX integration."""
    t: float
    y: Float[Array, "n_state"]  # type: ignore
    step: int


class IMEXSolution(NamedTuple):
    """Solution from IMEX integration."""
    ts: Float[Array, "n_times"]  # type: ignore
    ys: Float[Array, "n_times n_state"]  # type: ignore
    success: bool
    num_steps: int
    message: str


def build_diffusion_matrix_implicit(
    rho: Float[Array, "N"],  # type: ignore
    Vprime: Float[Array, "N"],  # type: ignore
    chi: Float[Array, "N"],  # type: ignore
    dt: float,
    theta: float = 1.0,
) -> Tuple[Float[Array, "N-1 N-1"], Float[Array, "N-1"]]:  # type: ignore
    """
    Build the implicit diffusion matrix for (I - theta*dt*L)*T^{n+1} = rhs.
    
    Returns (A, b_bc) where:
    - A is the (N-1) x (N-1) tridiagonal matrix (I - theta*dt*L)
    - b_bc is the boundary contribution vector for Dirichlet at edge
    
    Boundary conditions:
    - Axis (i=0): Neumann (zero flux) via F_{-1/2}=0
    - Edge (i=N-1): Dirichlet (prescribed T_edge, not part of solve)
    
    The linear system is: A @ T_interior^{n+1} = rhs + dt*theta*b_bc*T_edge
    
    Args:
        rho: Radial grid (N nodes including boundary)
        Vprime: Volume derivative V'(rho) (N values)
        chi: Diffusivity profile (N values)
        dt: Timestep
        theta: Implicitness parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit Euler)
    
    Returns:
        A: Implicit operator matrix (N-1, N-1)
        b_bc: Boundary coupling vector (N-1,) to multiply by T_edge
    """
    N = rho.size
    assert Vprime.size == N and chi.size == N, "Grid size mismatch"
    assert N >= 3, "Need at least 3 nodes (core + interior + boundary)"
    
    # IMPORTANT: this function must match apply_diffusion_explicit() exactly.
    # apply_diffusion_explicit uses:
    #   flux_face = -k_face * (T_{i+1} - T_i),  k_face = V'_{i+1/2} * chi_{i+1/2} / dr_i
    #   divergence_i = -(flux_out - flux_in)/denom_i
    # which expands to:
    #   divergence_i = (k_i*T_{i+1} - (k_i+k_{i-1})*T_i + k_{i-1}*T_{i-1}) / denom_i
    # So L has negative diagonal entries and positive off-diagonals.

    # Clip safety (keep identical to apply_diffusion_explicit)
    Vprime = jnp.clip(Vprime, 1e-6, None)
    dr = jnp.diff(rho)
    dr = jnp.clip(dr, 1e-6 * jnp.max(dr) + 1e-12, None)

    chi_face = 0.5 * (chi[:-1] + chi[1:])
    Vprime_face = 0.5 * (Vprime[:-1] + Vprime[1:])

    # k_face[i] multiplies (T_{i+1} - T_i)
    k_face = Vprime_face * chi_face / dr  # (N-1,)

    # denom (cell volumes) exactly as in apply_diffusion_explicit
    Vprime_cell = 0.5 * (Vprime[:-1] + Vprime[1:])
    vol = Vprime_cell * dr
    vol_floor = jnp.maximum(1e-4 * jnp.max(vol), 1e-10)
    denom = jnp.maximum(vol, vol_floor)  # (N-1,)

    # Build tri-diagonals for L_int on unknown nodes i=0..N-2 (size N-1)
    lower = jnp.zeros((N - 1,))
    diag = jnp.zeros((N - 1,))
    upper = jnp.zeros((N - 1,))

    # i = 0 (axis): flux_in = 0
    diag = diag.at[0].set(-k_face[0] / denom[0])
    upper = upper.at[0].set(+k_face[0] / denom[0])

    # i = 1..N-3
    idx = jnp.arange(1, N - 2)
    lower = lower.at[idx].set(+k_face[idx - 1] / denom[idx])
    diag = diag.at[idx].set(-(k_face[idx - 1] + k_face[idx]) / denom[idx])
    upper = upper.at[idx].set(+k_face[idx] / denom[idx])

    # i = N-2 (last interior): couples to boundary via k_face[N-2] * (T_edge - T_{N-2})
    i = N - 2
    lower = lower.at[i].set(+k_face[i - 1] / denom[i])
    diag = diag.at[i].set(-(k_face[i - 1] + k_face[i]) / denom[i])
    # upper[i] is zero because T_{N-1} is not an unknown

    # boundary coupling term: + k_face[N-2]/denom[N-2] * T_edge
    b_bc = jnp.zeros((N - 1,))
    b_bc = b_bc.at[i].set(+k_face[i] / denom[i])

    # A = I - theta*dt*L_int
    A_diag = 1.0 - theta * dt * diag
    A_lower = -theta * dt * lower
    A_upper = -theta * dt * upper

    A = jnp.diag(A_diag) + jnp.diag(A_lower[1:], k=-1) + jnp.diag(A_upper[:-1], k=1)
    return A, b_bc


def apply_diffusion_explicit(
    rho: Float[Array, "N"],  # type: ignore
    Vprime: Float[Array, "N"],  # type: ignore
    chi: Float[Array, "N"],  # type: ignore
    Te_total: Float[Array, "N"],  # type: ignore
) -> Float[Array, "N-1"]:  # type: ignore
    """
    Compute explicit diffusion term: D(T) for all interior nodes.
    
    Uses conservative finite-volume discretization with:
    - Neumann BC at axis (zero flux)
    - Dirichlet BC at edge (T_edge is prescribed, not computed)
    
    Args:
        rho: Radial grid (N nodes)
        Vprime: Volume derivative (N values)
        chi: Diffusivity (N values)
        Te_total: Temperature profile including boundary (N values)
    
    Returns:
        divergence: Diffusion term for interior nodes (N-1,)
    """
    N = rho.size
    Vprime = jnp.clip(Vprime, 1e-6, None)
    dr = jnp.diff(rho)
    dr = jnp.clip(dr, 1e-6 * jnp.max(dr) + 1e-12, None)
    
    # Face gradients and fluxes
    grad_T = jnp.diff(Te_total) / dr
    chi_face = 0.5 * (chi[:-1] + chi[1:])
    Vprime_face = 0.5 * (Vprime[:-1] + Vprime[1:])
    flux_face = -Vprime_face * chi_face * grad_T
    
    # Flux into/out of interior cells
    # At axis: flux_in[0] = 0 (Neumann BC)
    flux_in = jnp.concatenate([jnp.array([0.0]), flux_face[:-1]])
    flux_out = flux_face
    
    # Cell volumes
    Vprime_cell = 0.5 * (Vprime[:-1] + Vprime[1:])
    vol = Vprime_cell * dr
    vol_floor = jnp.maximum(1e-4 * jnp.max(vol), 1e-10)
    denom = jnp.maximum(vol, vol_floor)
    
    divergence = -(flux_out - flux_in) / denom
    return divergence


class IMEXIntegrator(eqx.Module):
    """
    IMEX time integrator for tokamak transport with implicit diffusion.
    
    Solves:
        dT/dt = D(T) + S(T, t)
        dz/dt = f_z(z, u(t))
    
    Where D is diffusion (treated implicitly) and S is source (explicit).
    
    Uses theta-method:
        T^{n+1} = T^n + dt * [theta*D(T^{n+1}) + (1-theta)*D(T^n)] + dt*S(T^n)
    """
    
    theta: float = 1.0  # Implicitness (0=explicit, 0.5=Crank-Nicolson, 1=implicit Euler)
    dt_base: float = 0.001  # Base timestep (seconds). NOTE: currently not used in fixed-substep mode.
    max_steps: int = 50000
    rtol: float = 1e-4  # Relative tolerance for adaptive stepping (future)
    atol: float = 1e-6  # Absolute tolerance
    substeps: int = 1  # Fixed substeps per save interval (enables reverse-mode autodiff)
    
    def __init__(
        self,
        theta: float = 1.0,
        dt_base: float = 0.001,
        max_steps: int = 50000,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        substeps: int = 1,
    ):
        self.theta = theta
        self.dt_base = dt_base
        self.max_steps = max_steps
        self.rtol = rtol
        self.atol = atol
        self.substeps = int(substeps)
    
    def step(
        self,
        t: float,
        y: Float[Array, "n_state"],  # type: ignore
        dt: float,
        model,  # HybridField model
        Te_edge_curr: float,
        Te_edge_next: float,
        args: tuple,
    ) -> Float[Array, "n_state"]:  # type: ignore
        """
        Take one IMEX step.
        
        Args:
            t: Current time
            y: Current state [T_interior_scaled, z]
            dt: Timestep
            model: HybridField model with IMEX methods
            Te_edge_curr: Boundary temperature at time t (used for explicit terms)
            Te_edge_next: Boundary temperature at time t+dt (used for implicit BC coupling)
            args: Additional arguments (rho, Vprime, interpolants, etc.)
        
        Returns:
            y_new: State at t+dt
        """
        # Unpack state
        Te_hat_interior = y[:-1]  # Scaled interior temperatures
        z = y[-1]
        
        # Get model parameters
        Te_scale = model.Te_scale
        
        # Reconstruct full temperature profile
        Te_interior = Te_hat_interior * Te_scale
        Te_total = jnp.append(Te_interior, Te_edge_curr)
        
        # Build implicit diffusion matrix
        A_matrix, b_bc, chi_profile = model.build_diffusion_matrix_imex(
            t, Te_total, z, args, dt, self.theta
        )
        
        # Compute explicit source
        S_explicit = model.compute_source_imex(t, Te_total, z, args)
        
        # Compute explicit diffusion (for theta < 1, Crank-Nicolson)
        if self.theta < 1.0:
            rho_vals, Vprime_vals = args[0], args[1]
            rho = rho_vals
            Vprime = jnp.clip(Vprime_vals, 1e-6, None)
            D_explicit = apply_diffusion_explicit(rho, Vprime, chi_profile, Te_total)
        else:
            D_explicit = jnp.zeros_like(Te_interior)
        
        # Build RHS for interior temperatures
        # RHS = T^n + dt*[(1-theta)*D(T^n) + S(T^n)] + dt*theta*b_bc*T_edge(t^{n+1})
        rhs = Te_interior + dt * ((1.0 - self.theta) * D_explicit + S_explicit)
        rhs = rhs + dt * self.theta * b_bc * Te_edge_next
        
        # Solve linear system: A @ T^{n+1} = rhs
        Te_interior_new = jnp.linalg.solve(A_matrix, rhs)
        Te_interior_new = smooth_clamp(Te_interior_new, 0.0, 5000.0)
        
        # Update latent variable (explicit)
        z_dot = model.compute_latent_rhs_imex(t, z, args)
        z_new = z + dt * z_dot
        z_new = smooth_clamp(z_new, -10.0, 10.0)
        
        # Pack state
        Te_hat_interior_new = Te_interior_new / Te_scale
        y_new = jnp.concatenate([Te_hat_interior_new, jnp.array([z_new])])
        
        return y_new
    
    def integrate(
        self,
        t_span: Tuple[float, float],
        y0: Float[Array, "n_state"],  # type: ignore
        saveat: Float[Array, "n_save"],  # type: ignore
        model,  # HybridField model
        Te_edge_interp: Callable,
        args: tuple,
    ) -> IMEXSolution:
        """
        Integrate from t_span[0] to t_span[1], saving at specified times.
        
        Args:
            t_span: (t_start, t_end)
            y0: Initial state
            saveat: Times to save solution
            model: HybridField model
            Te_edge_interp: Interpolant for edge temperature T_edge(t)
            args: Model arguments
        
        Returns:
            IMEXSolution with (ts, ys, success, num_steps, message)
        """
        # IMPORTANT: reverse-mode autodiff does not support lax.while_loop with dynamic
        # termination (used previously for variable dt stepping). For training, use a
        # fixed number of substeps per save interval. This keeps loop bounds static and
        # works under jit/vmap/pmap.
        #
        # Consequence: dt_base is currently not used. The effective per-substep dt is
        #   dt = (t_next - t_curr) / substeps
        # and the main sensitivity knobs are theta and substeps.

        substeps_i32 = jnp.asarray(self.substeps, dtype=jnp.int32)

        def interval_scan(carry, t_next):
            y_curr, t_curr = carry
            dt = (t_next - t_curr) / substeps_i32

            def one_substep(i, state):
                y, t = state
                t_new = t + dt
                Te_edge_curr = Te_edge_interp(t)
                Te_edge_next = Te_edge_interp(t_new)
                y_new = self.step(t, y, dt, model, Te_edge_curr, Te_edge_next, args)
                return (y_new, t_new)

            y_final, t_final = jax.lax.fori_loop(0, self.substeps, one_substep, (y_curr, t_curr))
            return (y_final, t_final), y_final

        # Initial state
        # We assume saveat[0] corresponds to y0
        init_carry = (y0, saveat[0])
        
        # Scan over remaining save points
        # If saveat has length 1, this returns empty arrays, which is handled correctly
        (y_end, t_end), ys_rest = jax.lax.scan(interval_scan, init_carry, saveat[1:])
        
        # Concatenate initial state with results
        ys = jnp.concatenate([y0[None, :], ys_rest], axis=0)
        
        total_steps = jnp.asarray((saveat.shape[0] - 1) * int(self.substeps), dtype=jnp.int32)
        success = jnp.asarray(total_steps <= int(self.max_steps))
        success = jnp.logical_and(success, jnp.all(jnp.isfinite(ys)))
        return IMEXSolution(
            ts=saveat,
            ys=ys,
            success=success,
            num_steps=total_steps,
            message="Success (fixed-substep scan)",
        )

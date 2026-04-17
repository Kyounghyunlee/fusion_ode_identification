"""Small JAX-friendly interpolation utilities.

This branch is IMEX-only and avoids depending on full ODE-solver libraries for
basic interpolation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LinearInterpolation(NamedTuple):
    """Piecewise-linear interpolation with a Diffrax-like `.evaluate(t)` API.

    - `ts`: shape (T,)
    - `ys`: shape (T,) or (T, D)

    `evaluate(t)` supports scalar `t` or vector `t`.
    """

    ts: jnp.ndarray
    ys: jnp.ndarray

    def evaluate(self, t: jnp.ndarray) -> jnp.ndarray:
        ts = self.ts
        ys = self.ys

        t = jnp.asarray(t, dtype=ts.dtype)

        def eval_scalar(t_scalar):
            # Right side so that exact knot points use the interval ending at that knot.
            idx = jnp.searchsorted(ts, t_scalar, side="right") - 1
            idx = jnp.clip(idx, 0, ts.shape[0] - 2)

            t0 = ts[idx]
            t1 = ts[idx + 1]
            y0 = ys[idx]
            y1 = ys[idx + 1]

            denom = jnp.where(t1 > t0, t1 - t0, jnp.array(1.0, dtype=ts.dtype))
            w = (t_scalar - t0) / denom
            return y0 + w * (y1 - y0)

        # Important: avoid `lax.cond` here. Even though `t.ndim` is static,
        # `lax.cond` traces *both* branches; the vmap-branch fails when `t` is
        # a scalar (rank-0). A Python branch only traces the valid path.
        if t.ndim == 0:
            return eval_scalar(t)
        return jax.vmap(eval_scalar)(t)

# Physics-Informed Tokamak ODE

This document records the current mathematical model and the data-quality policy
used by the active v3 pipeline.

## Philosophy

The model should not learn from fabricated or obviously damaged inputs. Thomson
scattering support is sparse and edge-localized, so the current pipeline keeps
only measured, stable `T_e` radial columns and drops entire packs when fewer than
four stable edge columns survive. Data trust comes before pack count.

The learned model is a hybrid:

- Known stiff part: conservative radial diffusion in normalized flux radius
  `rho`.
- Unknown closure: a small residual source network.
- Regime coordinate: one scalar barrier latent tied directly to D-alpha.

## State and Inputs

The runtime state is

```text
y(t) = [T_0(t), ..., T_{N-2}(t), zeta(t)]
```

where `T_{N-1}(t)` is a Dirichlet edge boundary supplied from data and `zeta` is
the unconstrained barrier latent. The displayed barrier coordinate is

```text
z_b(t) = sigmoid(zeta(t)).
```

Active scalar controls are:

```text
u(t) = [P_nbi(t), Ip(t), nebar(t), D_alpha(t)]
```

`nebar` is the representative profile-density scalar, not a radial density
profile. It is broadcast over rho inside `ShotBundle.ne_vals` only to preserve
the existing source-network API.

## Transport Equation

The temperature profile is evolved with

```text
dT/dt = div_rho( chi(rho, z_b) grad_rho T ) + S_theta(rho, T, ne, u, zeta)
```

in finite-volume form. `chi(rho,z_b)` is structured and positive; the residual
source `S_theta` is bounded by architecture, scale, and an explicit source
penalty. This makes the stiff part interpretable and dissipative while leaving
room for missing heating, radiation, and closure effects.

## Boundary Conditions

Axis boundary:

```text
F_{-1/2} = 0
```

This is the symmetry/zero-flux condition at `rho=0`.

Edge boundary:

```text
T(rho=1,t) = T_edge(t)
```

`T_edge(t)` is built per shot from the outermost masked-valid `T_e` value at each
time, then time-filled and clipped to a finite positive range. The edge node is
not fitted as data; it is imposed as the boundary value.

## IMEX Time Integration

The custom solver uses a fixed-step theta method:

- Diffusion is implicit.
- Residual source and latent dynamics are explicit.
- Padded tail steps are frozen with `active_mask` so padding cannot contribute
  dynamics or loss.

The solver is written in JAX control flow, so training differentiates through the
fixed rollout.

## Barrier Latent and D-Alpha

D-alpha is the central observed transition proxy. The model uses

```text
h_D(t) = 1 - norm(D_alpha(t))
```

as H-mode/barrier evidence: a drop in D-alpha corresponds to stronger barrier
evidence. The latent feature vector is

```text
[h_D,
 -d/dt norm(D_alpha),
  d/dt norm(T_edge),
  d/dt norm(ne_edge),
  norm(P_nbi),
  norm(Ip)]
```

The barrier latent follows a target logit dominated by `h_D`:

```text
target_logit = evidence_gain * (h_D - 0.5) + weak_extra_drive + bias
dzeta/dt = (target_logit - zeta) / tau
```

The auxiliary D-alpha prediction is

```text
D_alpha_hat_norm = 1 - z_b.
```

Thus the same scalar latent that modulates edge transport is also directly
supervised to track normalized D-alpha. In the v3 configs, `lambda_dalpha = 10.0`
so this supervision is not decorative.

## Data Supervision Policy

The pack builder writes only strict `T_e` masks. A radial column must satisfy
coverage, median-temperature, jump, span, and edge-support checks before it can
contribute to training.

Current validated thresholds:

```text
rho_min retained by final packs: 0.859375
min robust Te median: 50 eV
max Te relative jump p95: 0.60
max robust Te relative span: 1.80
min stable edge columns per written shot: 4
```

The final strict rebuild contains 18 packs and 110 retained `T_e` columns. Regions
without measured support are not scored as ground truth.

## Loss Terms

The training objective is the sum of:

| Term | Purpose |
|---|---|
| Masked pseudo-Huber `T_e` error | Fit measured temperature where support is trustworthy |
| Source magnitude penalty | Keep residual closure small unless data demands it |
| Latent smoothness | Avoid high-frequency latent noise |
| Latent magnitude | Keep the scalar state bounded |
| Regime BCE | Use heuristic L/H windows when available |
| D-alpha auxiliary loss | Force the barrier coordinate to track D-alpha evidence |

The data term is restricted by both the per-shot mask and the corpus reliable
annulus. The model may still roll out over the full uniform grid, but only trusted
measured regions supervise it.

## Interpretation

After the latest smoke run, a 5-step CPU debug checkpoint is not a physics-quality
model, but it verifies the algorithmic behavior:

```text
all solves ok
no NaNs
18 strict packs evaluated
barrier latent visibly moves with 1 - norm(D_alpha)
```

Production-quality conclusions require a full GPU training run from the new packs
and architecture. Old checkpoints from before the scalar-control and barrier
latent changes should be treated as incompatible.

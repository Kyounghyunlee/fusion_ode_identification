# Experiment Ledger

## EXP-000 (2026-08-04) - Phase 1+3 restructure  [baseline tag: baseline-nf-v1]
Hypothesis: none (structural corrections, no performance claim).
Changes bundled deliberately as the Phase-3 restructure:
- depressed cubic latent (beta free; quadratic term removed as coordinate-redundant)
- z removed from residual source (interpretability bypass closed)
- chi positivity/ordering by construction (chi_edge_L > chi_edge_H > 0)
- causal z0 = lowest equilibrium at initial drive (Newton, in-graph)
- normalized per-task losses (S_TE=100 eV, S_SRC=1e4 eV/s)
- grouped session split (gap>20), locked test set; early stopping (patience 8 evals, min_delta 1e-3)
- FV cell-volume fix: axis half-cell + node-value V' (was face-averaged; spatial convergence was stalled)
Solver verification: order ~1 end-to-end, L2 err 4e-4..3.5e-3 over N=17..129; flux residual 0;
positivity pass; grad rel err <= 1e-9. Artifacts: logs/solver_verification/.
Conclusion: accepted as the v2 platform. All subsequent EXPs compare within this platform.

## EXP-001 (2026-08-04) - free vs monostable, seed 0 (grouped val)
Hypothesis: constrained beta<=0 fits held-out data as well as free beta.
- v2_free_s0: early stop @2251 (best 1850), best_val 1.1204, beta=+0.303,
  tau=30 ms, val median AUC 0.899, MAE 105 eV.
- v2_mono_s0: early stop @651 (best 250), best_val 1.1448, beta=-0.847,
  val median AUC 0.376 (ranking near-inverted; latent barely informative).
- Reference baselines (same split): static logistic AUC 0.58; first-order
  lag AUC 0.935 (classification only, no profiles).
Bug found & fixed mid-experiment: train.py persisted the raw config, not
the effective (override-applied) one; v2_mono_s0 was first evaluated under
the wrong beta parameterization and re-evaluated after the fix.
Conclusion (provisional, 1 seed): free-beta strongly preferred on val loss
and AUC; awaiting seeds 1-2.

## EXP-001 RESULT (superseded corpus) - free vs monostable, 3 seeds, 103 packs
free beta:  best_val 1.118/1.178/1.081, per-shot median val AUC 0.841/0.842/0.937, beta=+0.303/+0.296/+0.293
mono beta:  best_val 1.145/1.104/1.096, per-shot median val AUC 0.376/0.244/0.228, beta=-0.847/-0.764/-0.765
ablations (free): no-chi AUC 0.406; no-source MAE 255 eV (vs 104); no-dalpha AUC 0.932; no-regime AUC 0.944
baselines: static logistic AUC 0.579; first-order lag AUC 0.935
Calibration: p_H uncalibrated (pooled ECE 0.25); Platt (fit on val) halves ECE to ~0.11.
Event detection: LH recall 0.06-0.11 - LOW. Diagnosis: when the model does fire,
timing is accurate (median -12 ms vs label), but in most discharges the drive
never reaches the fold. Inspection of z(t) shows the latent correctly stays on
the L branch in those shots: the affine drive in (P_nbi, Ip, nebar) cannot
reach threshold for the X-point-height campaign shots, whose threshold depends
on shape (Meyer 2011). NOT a solver/optimizer failure - a missing-covariate one.
Conclusion: accept free>mono provisionally; act on the covariate diagnosis.

## PROTOCOL CHANGE (2026-08-05) - physics-directed drive extension
Pack builder now also stores: P_ohm_clean, P_loss = P_nbi + P_ohm - P_rad - dW/dt
(loss-power proxy; the quantity in which L-H thresholds are conventionally
expressed), W_mhd, kappa_ts, delta_ts, q95_ts (from equilibrium.nc).
x_point_z is present in the equilibrium file but not 1-D; not used.
Drive sets: basic = (P_nbi, Ip, nebar); extended = (P_loss, Ip, nebar, P_rad,
kappa, delta), all causal and in fixed physical units.
Corpus rebuilt: 107 packs. Split regenerated and re-locked (60/17/30).
All EXP-001 runs discarded (different corpus/split); EXP-002 retrains the full
comparison grid on one footing.

## EXP-002 (2026-08-05) - drive set x topology x seed, 107 packs, locked split
First completed run e_ext_free_s0 (extended drive, free beta):
  best_val 1.0790 @1850 (early stop @2651)
  val: median AUC 0.752 [0.53, 0.95], F1(cal) 0.72, Brier 0.199, ECE 0.138->0.095
  events: L->H recall 0.33, precision 0.43, |timing| median 30 ms
          H->L recall 0.00 (back-transitions not captured - limitation)
  latent: median z swing 1.81 (branch separation ~1.1); 100% of shots reach the
          H basin (basic drive: 0.32 swing, ~30% of shots) - the covariate
          diagnosis from EXP-001 is confirmed by the intervention it motivated.
Remaining grid in flight: e_bas_free_s{0,1,2} (controlled comparison on the
same corpus), e_ext_mono_s{0,1,2}, then ablations on extended/free.

## CRITICAL CORRECTION (2026-08-05) - equilibrium geometry was wrong
An independent audit of the equilibrium data path (prompted by a request to
document it in full) found three defects that invalidate the physical
interpretation of every profile result obtained so far:

1. INVERTED RADIAL COORDINATE. MAST Level-2 psi is stored in Wb/rad and is
   MAXIMAL on the magnetic axis. The code assumed psi_axis = min(psi) and
   psi_edge = max(psi) over the (R,Z) grid, so the normalization was
   inverted: measured rho = 1.000 AT THE MAGNETIC AXIS, falling to ~0.75 at
   both ends of the Thomson chord, non-monotonic in R, with every profile
   compressed into rho in [0.745, 1.0]. The LCFS-sampled psi was computed
   and then always discarded by a `psi_edge < psi_max` guard. Consequence:
   the "reliable edge annulus rho >= 0.80" that all supervision and the
   chi(rho) barrier were built around was, in the intended convention,
   NEAR-AXIS - and interpolation interleaved inboard/outboard channels.
2. GEOMETRY NOT ACTUALLY FROM THE EQUILIBRIUM. `flux_surface_volume` does
   not exist in the Level-2 equilibrium group, so V' fell through to a
   sentinel V' == 1, which data.py then silently replaced with the analytic
   cylindrical V' = 2*rho. The claim "V' supplied by the equilibrium
   reconstruction" was false for all 107 shots.
3. DEAD/MIS-INDENTED CODE. extract_geom_params read R_axis/R_lcfs/Z_lcfs,
   which are absent (Level-2 uses magnetic_axis_r/z, lcfs_r/lcfs_z), so
   R_major/a_minor/kappa/delta were NaN in every pack; and the Thomson-read
   block was indented inside the else-arm of the V'-availability test, so
   supplying a real V' would have raised NameError.
Also: choose_itime took the record-index midpoint, including pre-breakdown
samples (t = 0.215 s of a -0.100..0.530 s record).

FIX (preprocessing/equilibrium_geometry.py):
  psi_axis     = psi interpolated at (magnetic_axis_r, magnetic_axis_z)
  psi_boundary = median of psi sampled on the (lcfs_r, lcfs_z) contour
  rho          = sqrt(clip((psi - psi_axis)/(psi_boundary - psi_axis), 0, 1))
  V(rho)       = integral of 2*pi*R dR dZ over {rho' <= rho}, confined to
                 psi_N <= 1 inside the LCFS bounding box; V' = dV/drho
  itime        = midpoint of the contiguous finite magnetic-axis block
VALIDATION: V(rho=1) reproduces the equilibrium's own `volume` scalar to
0.0-0.3% (25145: 7.20 vs 7.20 m^3; 27574: 7.01 vs 7.04 m^3). rho is now
monotonic outward, 0 on axis, 1 at the separatrix. Supervised columns move
to rho = 0.73-1.00 (true edge) and retention improves (25145: 14 kept
columns vs 7 before).
CONSEQUENCE: EXP-002 is void for profile/chi interpretation. All packs are
being rebuilt and the full grid rerun on the corrected geometry. Latent-only
quantities (drive, beta, event timing) are geometry-independent in their
inputs but were trained jointly, so they are retrained too rather than
mixed across geometries.

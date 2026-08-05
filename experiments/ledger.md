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

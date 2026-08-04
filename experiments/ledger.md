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

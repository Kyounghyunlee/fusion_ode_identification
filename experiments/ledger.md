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

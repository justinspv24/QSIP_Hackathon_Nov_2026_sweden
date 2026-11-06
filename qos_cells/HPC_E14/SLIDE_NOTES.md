# Backup slide (E14)

**Title:** Stronger classical streaming learners narrow, but do not close, the PBMC68k gap at S ≈ 44–64

- Best one-pass learner (hashing + SGD) reaches **0.86–0.87** at S = 44–64 (paper accounting; 0.85–0.86 counting every float). The paper's data-oblivious sketches reach ≈ 0.81 there. Full ridge on all genes = 0.901 (α = 200) / 0.913 (α tuned the same way).
- SGD's extra accuracy is an accuracy/balance trade-off, not a more memory-efficient fit:
  - exact ridge on the same hashed features, tuned the same way, stays at 0.811;
  - on balanced accuracy, SGD is not better (0.766 vs 0.784).
- Counting true floats (Gram included), covariance-based methods need 10³–10⁷ floats: sketch selection + ridge, Frequent Directions, exact hashed ridge. Only SGD-type learners reach 0.89 at ~500 floats.
- Not tested: nonlinear models, other datasets, any quantum run (the quantum size of 42–44 is the paper's formula).

Figure: `qos_cells/HPC_E14/figures/fig_e14_gap.png` (balanced-accuracy version: `fig_e14_gap_bacc.png`)

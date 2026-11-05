# E5: IQM Garnet vs IBM — summary

1. Same frozen sketch, 8 test cells, 5 calibration circuits, 2,000 shots: IQM Garnet vs IBM ibm_kingston (median 2q error 1.8e-03); IBM job dbmetst8mdtc73d7roj0.
2. 2-qubit gates · depth (test circuit 0): N=4 IQM 11 CZ · 24 vs IBM 11 · 44; N=16 IQM 53 CZ · 108 vs IBM 64 · 248.
3. Calibration alpha (signal kept): N=4 IQM 0.78 vs IBM 0.89; N=16 IQM 0.43 vs IBM 0.74.
4. Cells correct raw → calibrated: N=4 IQM 8/8 → 8/8 vs IBM 8/8 → 8/8; N=16 IQM 5/8 → 8/8 vs IBM 8/8 → 8/8.
5. Mean |error| raw → calibrated: N=4 IQM 0.127 → 0.046 vs IBM 0.059 → 0.026; N=16 IQM 0.258 → 0.094 vs IBM 0.069 → 0.055.

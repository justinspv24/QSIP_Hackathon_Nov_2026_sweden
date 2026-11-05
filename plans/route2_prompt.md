# Route 2 prompt — classical memory stress test (paste into a new chat in the "QSIP Hackathon Sweden" project)

---

I'm at the QSIP Hackathon (Stockholm, 5–6 Nov 2026, ExtremeData4QML challenge). Plan 3 is already executed: quantum oracle sketching for rare-cell classification on PBMC3k. Now run **Route 2: a classical memory stress test**. Work autonomously and only ask me if you're blocked.

## Context (read first)
- Project docs: `claude/plan_3.md` (the plan), `claude/plan_3_results.md` (results, deviations, speaker notes), `presentation/plan3.html` (deck).
- Code: `C:\Users\justi\Projects\QSIP\qos_cells\` on my laptop (request folder access to `C:\Users\justi\Projects\QSIP`). The repo is the `qos_cells` package: `load_data.py` (PBMC3k loader, train-only gene selection, amplitude features p = x²/‖x‖²), `stream.py` (record stream with ±1/π̂ importance weights), `sketch.py`, `readout.py` (balanced accuracy / AUPRC, threshold on train), `io.py` (results JSON with schema_version 1), `config.yaml`, scripts `01`–`08`. `data/` is not in the folder; the loader downloads PBMC3k itself (~25 MB from GitHub). Work on a copy in the cloud workspace, then write results back to my folder.
- Paper: Zhao, Zlokapa, Preskill, Huang 2026, arXiv:2604.07639. It claims an **exponential memory advantage**, proven only for its task families; on real data (scRNA-seq, movie reviews) it gives *evidence* only.

## The question
Our quantum sketch needs log₂N qubits for N features. The open question: **can a classical streaming method with memory S ≪ N already solve our task?** If yes, there's no memory advantage on this data. If accuracy collapses unless S ≈ N, that's evidence for the paper's claim on real data. Either answer is a result. Report it honestly, with no spin.

## Experiment
1. **Feature space large enough to matter:** gene-pair products (x_g·x_h, g ≤ h) of the top K genes, chosen on the train split only. Use K = 512 → ~1.3·10⁵ features and K = 2048 → ~2.1·10⁶ features (go bigger only if memory and time allow). Amplitude-normalise per cell as in `load_data.amplitude_features`. Positive classes: B cells, dendritic cells, megakaryocytes (one vs rest).
2. **Reference (memory = N):** exact centroid Δ = μ₁ − μ₀ (= our M → ∞ sketch), plus class-balanced logistic regression as a sanity check.
3. **Classical streaming methods with memory budget S** (S from ~10 up to N, log grid). All of them see the same stream of (cell, feature, label, value) records with the same ±1/π̂ weights, the same number of records M, and the same train/test split.
   - Count-sketch of Δ (S buckets, signed hashes, median over a few rows): classify with the decoded Δ
   - Feature hashing (hash features into S dims, then centroid and logreg)
   - Random projection / JL to S dims (dense Gaussian or sparse), then centroid
   - Heavy hitters (SpaceSaving / Misra–Gries) keeping the S largest |Δ| features
   - Reservoir sampling of S records (naive baseline)
4. **Control where compression should fail:** a synthetic task with no structure (random ± signal spread over all N features, as in the paper's worst-case families) at the same N. This shows whether the real data succeeds because of its structure.
5. **Metrics:** balanced accuracy and AUPRC only (never plain accuracy), 5 seeds. Key number: **S\*** = the smallest S within 0.01 balanced accuracy of the exact centroid, per method and class, and the ratio **S\*/N**. Also record bytes used (count every float, int and hash seed).
6. **Quantum column, for comparison only:** log₂N qubits (≈2·log₂N logical) from `resources.py`. Do NOT claim we ran quantum at this N. The sketch at this size is a classical simulation.

## Rules
- Gene and feature selection on train only. Fixed seeds. Every number in `results/*.json` (schema_version 1, with the config). Figures and slides are built only from JSON.
- Be explicit about limits: Louvain labels are circular, the rare-class test sets are tiny (DC ≈ 11 cells, Mk ≈ 5), and a finite experiment is evidence, not proof.
- If a method is clearly worse because I set it up badly (hash collisions, wrong scaling), fix it before concluding. A fair classical baseline is the whole point.

## Deliverables
1. `scripts/09_route2_classical_memory.py` and `results/e7_route2.json`
2. Figure `figures/fig_e7_memory.png`: balanced accuracy vs memory S (log axis) per method, with the exact centroid (S = N) and the quantum log₂N line marked; one panel for real data, one for the synthetic control
3. One new backup slide in `presentation/plan3_template.html` (rebuild via `scripts/08_slides.py`), titled with the honest verdict
4. A short section appended to `claude/plan_3_results.md`: verdict, S\*/N table, what it means for the advantage claim
5. Save everything **both** to the claude.ai project and to my local folder `C:\Users\justi\Projects\QSIP` (I can't delete files there; move superseded files into `_to_delete\`)

Answer style: short, direct, simple English; tables over prose; no sugarcoating.

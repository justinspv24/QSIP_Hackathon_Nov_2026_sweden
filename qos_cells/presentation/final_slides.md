# Final Slides (QSIP talk, 6 Nov 2026)

**Deck:** https://claude.ai/artifact/4mHEdxnn4BKdFEHSkWcuRu (Slides artifact, 7 slides, white theme, LaTeX equations as MathJax SVG). Export as PDF or PPTX from the deck.
**Story:** follows the 5 challenge steps (instructions.md: dataset → challenge → quantum technique → prototype → real world), plus one honest classical check.

| # | Slide | Key numbers (source) |
|---|---|---|
| 1 | Learning rare cells from a data stream too big to store | — |
| 2 | Blood cells: mostly zeros, and the key ones are rare | 32,738 genes, 93.8 % zeros, DC 37 (1 in 71), Mk 15 (1 in 176) (dataset in plan3 RESULTS) |
| 3 | Each record nudges a phase, then it is gone | $n=\lceil\log_2N\rceil$, $\varphi_g$, $\langle Z\rangle=\sum_g a_g^2\sin\varphi_g$ |
| 4 | Memory stays tiny; rare classes cost samples, not qubits | $\varepsilon^2\approx C N/M$; $M\propto 1/[\pi(1-\pi)]$; ×3.1/×13/×71 vs law ×2.8/×13/×50; DC ×72 (E2, E3) |
| 5 | It runs on real quantum hardware: 8 of 8 test cells right | Garnet N=4 8/8, α 0.78; N=16 5/8→8/8, α 0.43; IBM Kingston 8/8, α 0.89 / 0.74 (E5, e5_ibm_results.md) |
| 6 | Yes: on real data, a small classical sketch is enough | PBMC gene pairs N=2.1M → 256 counters; 1000 Genomes N=791k → 16,384; synthetic control → 262k (12 %) (E7, E8) |
| 7 | Where this matters, and what it would really cost | 30 / 60 logical qubits, ≈11 min / ≈1.5 years, 131 kB / 4.3 GB classical (E6) |

Speaker notes are in each slide. Not included: HPC runs E9–E13 (no results yet). Figures: `qos_cells/scripts/11_final_figs.py`; slide sources: `qos_cells/scripts/11b_final_slides_build.py`.

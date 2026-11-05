"""qos_cells — quantum oracle sketching for rare-cell classification (QSIP Hackathon 2026, Plan 3).

Modules
-------
load_data   : PBMC3k loader, stratified split, train-only gene selection, amplitude features
stream      : the record stream (cell, gene, label, value) -> (gene index, importance-weighted kick)
sketch      : the oracle sketch (sum of commuting phase kicks), safe rescaling, Hadamard-test score
circuits    : Qiskit circuits for the one-query readout (state prep + controlled-diagonal + Hadamard test)
readout     : metrics (balanced accuracy, AUPRC), threshold selection, hardware calibration
resources   : the resource ledger (qubits, samples, gates, time) for atlas-scale feature spaces
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

"""E6 — the resource ledger for atlas-scale feature spaces (all genes, gene pairs). Pure formulas (resources.py),
reported with the measured imbalance factors from E3. Writes results/e6_resources.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qos_cells import load_config
from qos_cells.resources import ledger, human
from qos_cells.io import save_json, load_json


def main():
    cfg = load_config()
    rc = cfg["resources"]
    e3 = load_json("e3_imbalance.json")
    pi_rare = rc["pi_rare"]
    rows = []
    for name, N in rc["feature_spaces"].items():
        L = ledger(N, rc["epsilon"], rc["queries"], pi_rare, rc["logical_gate_rate_hz"])
        L["feature_space"] = name
        rows.append(L)
        print(f"[E6] {name:10s} N=2^{L['index_qubits']}: {L['index_qubits']} index qubits, ≈{L['logical_qubits']} logical | "
              f"samples {human(L['samples_balanced'])} balanced, x{L['imbalance_factor']:.0f} -> {human(L['samples_rare'])} for pi={pi_rare} | "
              f"gates/sketch {human(L['gates_per_sketch'])} ≈ {human(L['seconds_per_sketch'])} s at {human(L['gate_rate_hz'])} Hz "
              f"({L['years_per_sketch']:.2g} years) | classical: {human(L['classical_bytes'])} bytes")
    measured = {s["positive"]: dict(pi=s["pi_hat"], law_factor=s["law_factor"], M_for_err_0p1=s["M_for_err_0p1"]) for s in e3["real"]}
    save_json("e6_resources.json", dict(ledger=rows, measured_imbalance=measured, assumptions=dict(
        eps=rc["epsilon"], Q=rc["queries"], pi_rare=pi_rare, logical_gate_rate_hz=rc["logical_gate_rate_hz"],
        gates_per_record="2n-1 (n-controlled phase with ancillas)", logical_qubits="2n (index + Hadamard ancilla + work)",
        samples="N Q^2 / eps (paper) x 1/(pi(1-pi)) (ours)",
        classical_lower_bound="M*S >= Omega(N*Q_C) for the paper's task families; not proven for the centroid task")), cfg)


if __name__ == "__main__":
    main()

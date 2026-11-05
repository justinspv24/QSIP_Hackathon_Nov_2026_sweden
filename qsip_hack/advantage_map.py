"""Advantage map: when does quantum oracle sketching (QOS) beat classical memory?

Uses the LS-SVM machine-size accounting of Zhao, Zlokapa, Preskill, Huang
(arXiv 2604.07639, Supplementary Information; repo: real_datasets/sweep_utils.py):

    Q(N, D, s) = 2 ceil(log2(N + 2D)) + ceil(log2(s + 1)) + 3 qubits + 1 float

and compares it with
    classical streaming : D numbers (one weight per feature)
    classical sparse    : nnz  ~ N * s_avg numbers (store the whole data)

Advantage factor = classical / quantum. The quantum machine wins only when
D grows faster than ~2 log2(N + 2D): high-dimensional data, not big-N data.

Run:  python advantage_map.py
Outputs: advantage_map.pdf / .png and a printed table.
NOTE: s (max nonzeros per row) and s_avg for the sparse datasets are rough
estimates; s only enters through ceil(log2(s+1)), i.e. +-1-2 qubits.
"""
import numpy as np
import matplotlib.pyplot as plt


def qos_machine_size(n_samples, n_features, sparsity):
    """LS-SVM QOS machine size (qubits + 1 classical float), as in sweep_utils."""
    return (2 * np.ceil(np.log2(n_samples + 2 * n_features))
            + np.ceil(np.log2(sparsity + 1)) + 4)


# name: (N samples, D features, s max nnz/row, s_avg mean nnz/row, property)
DATASETS = {
    "Credit-card fraud":        (284_807, 30, 30, 30, "imbalanced, dense, low-D"),
    "KDD Cup 99 (10%)":         (494_021, 41, 41, 41, "imbalanced, dense, low-D"),
    "Cora":                     (2_708, 1_433, 32, 18, "graph, sparse words"),
    "CiteSeer":                 (3_327, 3_703, 64, 32, "graph, sparse words"),
    "Tox21 (2048-bit FP)":      (7_831, 2_048, 128, 45, "imbalanced, sparse"),
    "Tox21 (2^20-bit FP)":      (7_831, 2**20, 128, 45, "imbalanced, sparse, high-D"),
    "Dorothea (paper)":         (1_150, 100_000, 2_048, 900, "imbalanced, sparse, high-D"),
    "PBMC68k scRNA (paper)":    (68_579, 32_738, 4_096, 800, "sparse, high-D"),
    "Billion-molecule library": (10**9, 2**20, 128, 45, "massive, sparse, high-D"),
}


def table():
    hdr = f"{'dataset':28s} {'N':>14s} {'D':>10s} {'Q qubits':>9s} {'D/Q':>10s} {'nnz/Q':>12s}  verdict"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, (N, D, s, s_avg, prop) in DATASETS.items():
        Q = qos_machine_size(N, D, s)
        r_stream = D / Q
        r_sparse = N * s_avg / Q
        verdict = "quantum smaller" if r_stream > 1 else "NO advantage (classical streaming smaller)"
        print(f"{name:28s} {N:14,d} {D:10,d} {Q:9.0f} {r_stream:10.1f} {r_sparse:12.2e}  {verdict}")
        rows.append((name, N, D, Q, r_stream, prop))
    return rows


def crossover_D(N, s=128):
    """Smallest D with D > Q(N, D, s): the dimension where QOS starts winning."""
    D = 1
    while D <= qos_machine_size(N, D, s):
        D += 1
    return D


def plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) machine size vs feature dimension
    ax = axes[0]
    Ds = np.logspace(0.5, 7, 400)
    ax.loglog(Ds, Ds, "k-", lw=2, label="Classical streaming (= D)")
    for N, ls in [(1e3, ":"), (1e6, "--"), (1e9, "-.")]:
        ax.loglog(Ds, [qos_machine_size(N, d, 128) for d in Ds], ls,
                  color="tab:purple", lw=2, label=f"Quantum QOS, N = {N:.0e}")
    for name, N, D, Q, r, _ in rows:
        c = "tab:green" if r > 1 else "tab:red"
        ax.scatter(D, Q, color=c, zorder=5, s=35)
    ax.axvspan(1, crossover_D(1e6), color="tab:red", alpha=0.07)
    ax.text(1.3, 5e2, "classical\nwins\n(low-D)", color="tab:red", fontsize=9)
    ax.set_xlabel("Feature dimension D")
    ax.set_ylabel("Machine size (memory units)")
    ax.set_title("(a) Quantum memory grows like log D, classical like D")
    ax.scatter([], [], color="tab:green", label="dataset: quantum smaller")
    ax.scatter([], [], color="tab:red", label="dataset: classical smaller")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.2, which="both")

    # (b) advantage factor per dataset
    ax = axes[1]
    names = [r[0].replace(" (paper)", "") for r in rows]
    ratios = [r[4] for r in rows]
    colors = ["tab:green" if r > 1 else "tab:red" for r in ratios]
    ax.barh(names, ratios, color=colors)
    ax.axvline(1, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Advantage factor  D / Q  (classical streaming / quantum)")
    ax.set_title("(b) Which data turns the advantage on?")
    ax.invert_yaxis()
    ax.grid(alpha=0.2, axis="x", which="both")
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    fig.savefig("advantage_map.pdf")
    fig.savefig("advantage_map.png", dpi=200)
    print("Saved advantage_map.pdf / advantage_map.png")


if __name__ == "__main__":
    rows = table()
    print()
    for N in [1e3, 1e6, 1e9]:
        print(f"Crossover: QOS beats classical streaming once D > {crossover_D(N)} (N = {N:.0e})")
    plot(rows)

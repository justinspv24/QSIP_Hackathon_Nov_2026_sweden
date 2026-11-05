import sys, json, numpy as np, importlib.util
sys.path.insert(0, "/home/claude/qos_cells")
from qos_cells.load_data import stratified_split, amplitude_features
from qos_cells.readout import best_threshold, evaluate
spec = importlib.util.spec_from_file_location("e7", "/home/claude/qos_cells/scripts/09_route2_classical_memory.py"); e7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e7)
d = np.load("eur_eas_chr22.npz"); G, pop, sp = d["G"], d["pop"], d["superpop"]
tasks = {"EAS vs EUR (easy contrast)": (sp == "EAS", np.ones(len(sp), bool)),
         "JPT vs CHB": (pop == "JPT", np.isin(pop, ["JPT", "CHB"])),
         "JPT vs rest of East Asia": (pop == "JPT", sp == "EAS"),
         "FIN vs rest of Europe": (pop == "FIN", sp == "EUR"),
         "TSI vs IBS": (pop == "TSI", np.isin(pop, ["TSI", "IBS"])),
         "CEU vs GBR": (pop == "CEU", np.isin(pop, ["CEU", "GBR"]))}
rows = []
for name, (pos, m) in tasks.items():
    y = pos[m].astype(int); P, ab, _ = amplitude_features(G[m].astype(np.float32))
    P = P.astype(np.float32)
    tr, te = stratified_split(y, 0.3, 7)
    Ptr, Pte, ytr, yte = P[tr], P[te], y[tr], y[te]; N = P.shape[1]
    D = Ptr[ytr == 1].mean(0).astype(float) - Ptr[ytr == 0].mean(0).astype(float)
    ex = e7.evaluate_columns(Ptr, ytr, Pte, yte, D[:, None])[0]
    d2 = D**2; s = np.sort(d2)[::-1]
    top1 = s[:max(1, N // 100)].sum() / s.sum(); prf = d2.sum()**2 / (d2**2).sum() / N
    tol = max(0.01, 0.5 / yte.sum(), 0.5 / (1 - yte).sum())
    curve = {}
    for S in [2**e for e in range(3, 17)] + [N]:
        cols = [e7.sk_hashing(D, S, np.random.default_rng([s_, S]).integers(0, 2**62, 4))[0] for s_ in range(3)]
        curve[S] = float(np.mean([r["balanced_accuracy"] for r in e7.evaluate_columns(Ptr, ytr, Pte, yte, np.stack(cols, 1))]))
    Ss = sorted(curve); ok = [curve[S] >= ex["balanced_accuracy"] - tol for S in Ss]
    Sstar = next((Ss[i] for i in range(len(Ss)) if all(ok[i:])), None)
    top = {}
    for S in [8, 64, 512, 4096, 32768]:
        keep = np.argsort(-np.abs(D))[:S]; Dt = np.zeros(N); Dt[keep] = D[keep]
        top[S] = e7.evaluate_columns(Ptr, ytr, Pte, yte, Dt[:, None])[0]["balanced_accuracy"]
    r = dict(task=name, n_pos=int(y.sum()), n_neg=int((1 - y).sum()), N=N, exact_bal_acc=round(ex["balanced_accuracy"], 3), exact_auprc=round(ex["auprc"], 3),
             tol=round(tol, 3), top1pct_share=round(float(top1), 3), PR_over_N=round(float(prf), 3), hashing_S_star=Sstar,
             S_star_over_N=(round(Sstar / N, 4) if Sstar else None), hashing_curve={k: round(v, 3) for k, v in curve.items()},
             oracle_topS={k: round(v, 3) for k, v in top.items()})
    rows.append(r); print(json.dumps({k: r[k] for k in ("task","n_pos","n_neg","exact_bal_acc","exact_auprc","tol","top1pct_share","PR_over_N","hashing_S_star","S_star_over_N")}), flush=True)
    print("   hashing:", r["hashing_curve"], "\n   oracle top-S:", r["oracle_topS"], flush=True)
json.dump(rows, open("g1k_density_screen.json", "w"), indent=1)

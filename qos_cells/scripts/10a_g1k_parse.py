import numpy as np, time
from cyvcf2 import VCF
pan = [l.rstrip("\n").split("\t") for l in open("panel.txt")][1:]
keep = [p[0] for p in pan if p[2] in ("EUR", "EAS")]
v = VCF("chr22.vcf.gz", samples=keep, gts012=True, threads=2)
samples = v.samples
pop = {p[0]: (p[1], p[2]) for p in pan}
cols, ids = [], []
t = time.time(); n = 0
for r in v:
    n += 1
    if not r.is_snp or len(r.ALT) != 1: continue
    g = r.gt_types  # gts012: 0 hom-ref, 1 het, 2 hom-alt, 3 unknown
    if (g == 3).any(): continue
    af = g.mean() / 2
    if af < 0.05 or af > 0.95: continue
    cols.append(g.astype(np.int8)); ids.append(r.ID if r.ID else f"22:{r.POS}")
G = np.stack(cols, 1)
np.savez_compressed("eur_eas_chr22.npz", G=G, samples=np.array(samples), pop=np.array([pop[s][0] for s in samples]),
                    superpop=np.array([pop[s][1] for s in samples]), snp=np.array(ids))
print(f"variants read {n:,}, kept SNPs {G.shape[1]:,}, samples {G.shape[0]}, {time.time()-t:.0f} s")

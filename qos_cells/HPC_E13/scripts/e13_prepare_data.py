"""E13 data: PBMC68k (Zheng et al. 2017, Nat. Commun. 8:14049, "Fresh 68k PBMCs (Donor A)", 10x Genomics).

Inputs (downloaded into HPC_E13/data/raw/, git-ignored):
  matrix      https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
  annotation  https://raw.githubusercontent.com/10XGenomics/single-cell-3prime-paper/master/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv
              (the paper's per-cell labels: each cell assigned to the most correlated of 11 FACS-purified reference
               populations; NOT a clustering of this data)

Output: HPC_E13/data/pbmc68k.h5ad  (raw counts csr; obs = celltype) and HPC_E13/results/e13_data.json (urls, sha256, counts).
Normalisation (applied at load time, same as the PBMC3k file used in E1-E7): counts per cell scaled to 1e4, log1p.
"""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "data" / "raw"
URL_M = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
URL_A = "https://raw.githubusercontent.com/10XGenomics/single-cell-3prime-paper/master/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import anndata as ad
    import pandas as pd

    tgz, ann = RAW / "pbmc68k_matrices.tar.gz", RAW / "68k_pbmc_barcodes_annotation.tsv"
    with tarfile.open(tgz) as t:
        rd = lambda n: t.extractfile(f"filtered_matrices_mex/hg19/{n}").read()
        bc = pd.read_csv(io.BytesIO(rd("barcodes.tsv")), header=None)[0].values
        gn = pd.read_csv(io.BytesIO(rd("genes.tsv")), header=None, sep="\t")
        M = scipy.io.mmread(io.BytesIO(rd("matrix.mtx")))          # genes x cells
    X = sp.csr_matrix(M.T, dtype=np.float32)
    a = pd.read_csv(ann, sep="\t")
    assert len(a) == X.shape[0] and (a["barcodes"].values == bc).all(), "annotation barcodes do not match the matrix"
    obs = pd.DataFrame({"celltype": a["celltype"].astype(str).values}, index=bc)
    var = pd.DataFrame({"gene_id": gn[0].values}, index=pd.Index(gn[1].values).astype(str))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    out = HERE / "data" / "pbmc68k.h5ad"
    adata.write_h5ad(out, compression="gzip")
    cnt = obs["celltype"].value_counts()
    info = dict(schema_version=1, experiment="e13_data", dataset="PBMC68k fresh donor A (Zheng et al. 2017)",
                sources=dict(matrix=dict(url=URL_M, sha256=sha(tgz), version="10x Cell Ranger 1.1.0 sample, hg19 filtered matrices"),
                             annotation=dict(url=URL_A, sha256=sha(ann), version="10XGenomics/single-cell-3prime-paper@master")),
                n_cells=int(X.shape[0]), n_genes=int(X.shape[1]), nnz=int(X.nnz),
                median_genes_per_cell=float(np.median(np.diff(X.indptr))), median_umi_per_cell=float(np.median(np.asarray(X.sum(1)).ravel())),
                class_counts={k: int(v) for k, v in cnt.items()},
                class_prevalence={k: float(v / X.shape[0]) for k, v in cnt.items()},
                label_source="per-cell annotation from the paper (correlation to 11 purified reference populations)",
                normalisation="per cell to 1e4 counts, then log1p (at load time)")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "e13_data.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

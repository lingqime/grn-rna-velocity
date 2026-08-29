# ============================================
# Cell 37. NT expression QC for the 426
# candidate response genes
# ============================================

import numpy as np
import pandas as pd
from scipy import sparse

# NT cells
nt_mask = (
    adata.obs["gene"].astype(str).values
    == "non-targeting"
)

# Gene indices in the full H5AD
gene_to_idx = {
    g: i
    for i, g in enumerate(adata.var_names.astype(str))
}

vc_gene_idx = np.array(
    [gene_to_idx[g] for g in vc_genes],
    dtype=int
)

# Sparse slices only; do not copy full adata
S_nt = adata.layers["spliced"][nt_mask, :][:, vc_gene_idx]
U_nt = adata.layers["unspliced"][nt_mask, :][:, vc_gene_idx]

# Mean counts
S_mean = np.asarray(S_nt.mean(axis=0)).ravel()
U_mean = np.asarray(U_nt.mean(axis=0)).ravel()

# Fraction of NT cells with nonzero counts
S_detect = np.asarray(
    (S_nt > 0).mean(axis=0)
).ravel()

U_detect = np.asarray(
    (U_nt > 0).mean(axis=0)
).ravel()

response_qc = pd.DataFrame(
    {
        "S_mean_NT": S_mean,
        "U_mean_NT": U_mean,
        "S_detect_NT": S_detect,
        "U_detect_NT": U_detect,
        "beta": beta_series.loc[vc_genes].values,
        "gamma": gamma_series.loc[vc_genes].values,
    },
    index=vc_genes,
)

print("Candidate response genes:", len(response_qc))

print("\nSpliced mean:")
print(response_qc["S_mean_NT"].describe())

print("\nUnspliced mean:")
print(response_qc["U_mean_NT"].describe())

print("\nSpliced detection fraction:")
print(response_qc["S_detect_NT"].describe())

print("\nUnspliced detection fraction:")
print(response_qc["U_detect_NT"].describe())

# Same thresholds we previously used for regulator eligibility
pass_basic = (
    (response_qc["S_mean_NT"] >= 0.1)
    & (response_qc["U_mean_NT"] >= 0.02)
)

print(
    "\nGenes passing mean S >= 0.1 and mean U >= 0.02:",
    int(pass_basic.sum()),
    "/",
    len(pass_basic),
)

print("\nLowest 15 genes by unspliced mean:")
print(
    response_qc
    .sort_values("U_mean_NT")
    .head(15)[
        [
            "S_mean_NT",
            "U_mean_NT",
            "S_detect_NT",
            "U_detect_NT",
        ]
    ]
)
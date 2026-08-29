# ============================================
# Cell 39. Verify how VeloCycle n_scounts /
# n_ucounts are defined on a filtered gene set
# ============================================

import numpy as np

# Raw totals within the 120-gene MedGene object
S_sum_120 = np.asarray(
    adata_med.layers["spliced"].sum(axis=1)
).ravel()

U_sum_120 = np.asarray(
    adata_med.layers["unspliced"].sum(axis=1)
).ravel()

n_s_saved = adata_med.obs["n_scounts"].to_numpy(dtype=float)
n_u_saved = adata_med.obs["n_ucounts"].to_numpy(dtype=float)

print(
    "n_scounts == raw spliced sum over 120 genes:",
    np.allclose(n_s_saved, S_sum_120)
)

print(
    "n_ucounts == raw unspliced sum over 120 genes:",
    np.allclose(n_u_saved, U_sum_120)
)

print("\nMax absolute difference:")
print(
    "spliced:",
    np.max(np.abs(n_s_saved - S_sum_120))
)
print(
    "unspliced:",
    np.max(np.abs(n_u_saved - U_sum_120))
)

# Verify stored S_sz against the source-code formula
S_factor = n_s_saved.mean() / n_s_saved

# check a small slice only
rows = np.arange(100)
cols = np.arange(min(20, adata_med.n_vars))

S_raw_small = adata_med.layers["spliced"][rows, :][:, cols]
if hasattr(S_raw_small, "toarray"):
    S_raw_small = S_raw_small.toarray()

S_expected = (
    S_raw_small
    * S_factor[rows, None]
)

S_stored = adata_med.layers["S_sz"][rows, :][:, cols]
if hasattr(S_stored, "toarray"):
    S_stored = S_stored.toarray()

print(
    "\nStored S_sz matches VeloCycle formula:",
    np.allclose(
        S_expected,
        S_stored,
        rtol=1e-5,
        atol=1e-7
    )
)

print("\nMean n_scounts:", n_s_saved.mean())
print("Mean n_ucounts:", n_u_saved.mean())
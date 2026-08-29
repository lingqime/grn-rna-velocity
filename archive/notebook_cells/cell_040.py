# ============================================
# Cell 40. Test whether VeloCycle n_scounts /
# n_ucounts are full-transcriptome totals
# ============================================

import numpy as np

# Check cell ordering first
print(
    "Cell order identical:",
    np.array_equal(
        adata.obs_names.astype(str).values,
        adata_med.obs_names.astype(str).values
    )
)

# Sparse row sums over the full H5AD
S_sum_full = np.asarray(
    adata.layers["spliced"].sum(axis=1)
).ravel()

U_sum_full = np.asarray(
    adata.layers["unspliced"].sum(axis=1)
).ravel()

n_s_saved = adata_med.obs["n_scounts"].to_numpy(dtype=float)
n_u_saved = adata_med.obs["n_ucounts"].to_numpy(dtype=float)

print(
    "\nn_scounts == full-H5AD spliced totals:",
    np.allclose(n_s_saved, S_sum_full)
)

print(
    "n_ucounts == full-H5AD unspliced totals:",
    np.allclose(n_u_saved, U_sum_full)
)

print("\nMax absolute difference:")
print(
    "spliced:",
    np.max(np.abs(n_s_saved - S_sum_full))
)
print(
    "unspliced:",
    np.max(np.abs(n_u_saved - U_sum_full))
)

print("\nMeans:")
print("saved n_scounts:", n_s_saved.mean())
print("full spliced total:", S_sum_full.mean())
print("saved n_ucounts:", n_u_saved.mean())
print("full unspliced total:", U_sum_full.mean())
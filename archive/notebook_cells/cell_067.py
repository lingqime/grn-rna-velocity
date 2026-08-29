# ============================================
# Cell 64. Reconstruct the exact VeloCycle
# spliced count_factor for all cells
# ============================================

import numpy as np

# Raw total spliced counts per cell
# Sparse-safe: does NOT densify the full matrix
n_scounts_full = np.asarray(
    adata.layers["spliced"].sum(axis=1)
).ravel().astype(np.float64)

mean_n_scounts = n_scounts_full.mean()

count_factor_full = np.log(
    n_scounts_full
    / mean_n_scounts
)

print(
    "n_scounts_full shape:",
    n_scounts_full.shape
)

print(
    "mean n_scounts:",
    mean_n_scounts
)

print(
    "\ncount_factor summary:"
)

print(
    "min   =", np.min(count_factor_full)
)

print(
    "median=",
    np.median(count_factor_full)
)

print(
    "mean  =",
    np.mean(count_factor_full)
)

print(
    "max   =",
    np.max(count_factor_full)
)

print(
    "\nAny NaN:",
    np.isnan(count_factor_full).any()
)

print(
    "Any inf:",
    np.isinf(count_factor_full).any()
)

# Check alignment with the official compact extract
cell_ids_latent = np.asarray(
    lat["cell_ids"],
    dtype=object
)

print(
    "\nCell order identical to adata:",
    np.array_equal(
        cell_ids_latent,
        np.asarray(
            adata.obs_names,
            dtype=object
        )
    )
)
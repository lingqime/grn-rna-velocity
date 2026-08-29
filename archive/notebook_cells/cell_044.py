# ============================================
# Cell 44. Build VeloCycle-normalized sparse
# expression slices for final analysis
# ============================================

import numpy as np
from scipy import sparse

# --------------------------------------------
# 1. Final cells
# --------------------------------------------

cond_all = adata.obs["gene"].astype(str).to_numpy()

final_mask_vc = np.isin(
    cond_all,
    final_conditions_vc
)

final_cell_idx = np.flatnonzero(final_mask_vc)

print("Final cells:", len(final_cell_idx))

# Conditions and official phase for these cells
condition_final_cells = cond_all[final_cell_idx]
cycle_final_cells = cycle_time_vc[final_cell_idx]

# --------------------------------------------
# 2. Gene indices
# --------------------------------------------

gene_to_idx = {
    g: i
    for i, g in enumerate(adata.var_names.astype(str))
}

response_idx = np.array(
    [gene_to_idx[g] for g in vc_genes],
    dtype=int
)

regulator_idx = np.array(
    [gene_to_idx[g] for g in regulator_genes_vc],
    dtype=int
)

print("Response genes:", len(response_idx))
print("Regulator genes:", len(regulator_idx))

# --------------------------------------------
# 3. Raw sparse expression slices
# --------------------------------------------

U_response_raw = adata.layers["unspliced"][
    final_cell_idx, :
][:, response_idx].tocsr()

S_regulator_raw = adata.layers["spliced"][
    final_cell_idx, :
][:, regulator_idx].tocsr()

print("\nRaw matrix shapes:")
print("U_response_raw:", U_response_raw.shape)
print("S_regulator_raw:", S_regulator_raw.shape)

# --------------------------------------------
# 4. VeloCycle size factors
#
# U uses full-transcriptome unspliced totals.
# S uses full-transcriptome spliced totals.
# --------------------------------------------

n_s_full = adata_med.obs["n_scounts"].to_numpy(dtype=float)
n_u_full = adata_med.obs["n_ucounts"].to_numpy(dtype=float)

s_factor_all = n_s_full.mean() / n_s_full
u_factor_all = n_u_full.mean() / n_u_full

s_factor = s_factor_all[final_cell_idx]
u_factor = u_factor_all[final_cell_idx]

print("\nSize-factor ranges:")
print("S:", s_factor.min(), s_factor.max())
print("U:", u_factor.min(), u_factor.max())

# Sparse row-wise scaling
S_regulator_sz = sparse.diags(
    s_factor
) @ S_regulator_raw

U_response_sz = sparse.diags(
    u_factor
) @ U_response_raw

S_regulator_sz = S_regulator_sz.tocsr()
U_response_sz = U_response_sz.tocsr()

print("\nNormalized matrix shapes:")
print("U_response_sz:", U_response_sz.shape)
print("S_regulator_sz:", S_regulator_sz.shape)

# --------------------------------------------
# 5. Intervention sanity check
#
# For every perturbation q, its own regulator
# coordinate should remain exactly zero.
# --------------------------------------------

reg_to_col = {
    g: j
    for j, g in enumerate(regulator_genes_vc)
}

target_nonzero = {}

for q in final_perturbations_vc:
    rows = np.flatnonzero(
        condition_final_cells == q
    )

    j = reg_to_col[q]

    nnz = S_regulator_sz[rows, j].nnz

    if nnz > 0:
        target_nonzero[q] = int(nnz)

print(
    "\nPerturbation targets with nonzero "
    "normalized spliced counts:",
    len(target_nonzero)
)

if target_nonzero:
    print(target_nonzero)

print(
    "\nAll perturbation target coordinates zero:",
    len(target_nonzero) == 0
)
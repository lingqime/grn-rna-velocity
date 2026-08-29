# ============================================
# Cell 48. Build spliced trajectories for the
# 426 response genes
# ============================================

import numpy as np
from scipy import sparse

# --------------------------------------------
# 1. Raw spliced counts for 426 responses
# --------------------------------------------

S_response_raw = adata.layers["spliced"][
    final_cell_idx, :
][:, response_idx].tocsr()

print("S_response_raw shape:", S_response_raw.shape)

# --------------------------------------------
# 2. Apply the same VeloCycle spliced
#    size normalization
# --------------------------------------------

S_response_sz = sparse.diags(
    s_factor
) @ S_response_raw

S_response_sz = S_response_sz.tocsr()

print("S_response_sz shape:", S_response_sz.shape)

# --------------------------------------------
# 3. Bin by final condition + official phase
# --------------------------------------------

S_response_traj_vc = np.full(
    (
        len(final_conditions_vc),
        10,
        len(vc_genes)
    ),
    np.nan,
    dtype=np.float64
)

for q in final_conditions_vc:

    qi = condition_to_row[q]

    q_rows = np.flatnonzero(
        condition_final_cells == q
    )

    q_bins = bin_final_cells[q_rows]

    for b in range(10):

        local_rows = np.flatnonzero(
            q_bins == b
        )

        if len(local_rows) < 5:
            continue

        rows = q_rows[local_rows]

        S_response_traj_vc[
            qi, b, :
        ] = np.asarray(
            S_response_sz[
                rows, :
            ].mean(axis=0)
        ).ravel()


# --------------------------------------------
# 4. Sanity checks
# --------------------------------------------

print(
    "\nS_response_traj_vc shape:",
    S_response_traj_vc.shape
)

print(
    "Any NaN inside usable bins:",
    bool(
        np.isnan(
            S_response_traj_vc[
                usable_mask_vc
            ]
        ).any()
    )
)

print(
    "Any inf inside usable bins:",
    bool(
        np.isinf(
            S_response_traj_vc[
                usable_mask_vc
            ]
        ).any()
    )
)

# Basic trajectory scale
usable_S = S_response_traj_vc[
    usable_mask_vc
]

print("\nBinned spliced response values:")
print(
    pd.Series(
        usable_S.ravel()
    ).describe(
        percentiles=[
            0.01, 0.05, 0.50,
            0.95, 0.99
        ]
    )
)
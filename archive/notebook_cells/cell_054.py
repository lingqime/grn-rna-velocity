# ============================================
# Cell 52c. Build RAW-count trajectories
# for the 426 response genes
# ============================================

import numpy as np

U_response_traj_raw = np.full(
    (
        len(final_conditions_vc),
        10,
        len(vc_genes)
    ),
    np.nan,
    dtype=np.float64
)

S_response_traj_raw = np.full_like(
    U_response_traj_raw,
    np.nan
)

for q in final_conditions_vc:

    qi = condition_to_row[q]

    # Row positions within the 57,172 final cells
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

        # RAW unspliced mean
        U_response_traj_raw[
            qi, b, :
        ] = np.asarray(
            U_response_raw[
                rows, :
            ].mean(axis=0)
        ).ravel()

        # RAW spliced mean
        S_response_traj_raw[
            qi, b, :
        ] = np.asarray(
            S_response_raw[
                rows, :
            ].mean(axis=0)
        ).ravel()


print(
    "U_response_traj_raw shape:",
    U_response_traj_raw.shape
)

print(
    "S_response_traj_raw shape:",
    S_response_traj_raw.shape
)

print(
    "\nAny NaN in usable U bins:",
    bool(
        np.isnan(
            U_response_traj_raw[
                usable_mask_vc
            ]
        ).any()
    )
)

print(
    "Any NaN in usable S bins:",
    bool(
        np.isnan(
            S_response_traj_raw[
                usable_mask_vc
            ]
        ).any()
    )
)

print(
    "Any inf in usable U bins:",
    bool(
        np.isinf(
            U_response_traj_raw[
                usable_mask_vc
            ]
        ).any()
    )
)

print(
    "Any inf in usable S bins:",
    bool(
        np.isinf(
            S_response_traj_raw[
                usable_mask_vc
            ]
        ).any()
    )
)

print("\nRaw U mean over usable bins:")
print(
    np.nanmean(
        U_response_traj_raw[
            usable_mask_vc
        ]
    )
)

print("Raw S mean over usable bins:")
print(
    np.nanmean(
        S_response_traj_raw[
            usable_mask_vc
        ]
    )
)
# ============================================
# Cell 90. Phase-only null for all 426 genes
#
# Question:
# How much held-out performance is explained
# WITHOUT using any regulator trajectories?
#
# Model:
# r_g(q, phi) =
#   condition-independent Fourier function
#   of phase only
#
# NT always remains in training.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

phase_only_records = []

for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

    dt_g = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    r_g = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    ) / dt_g

    groups_g = row_conditions[
        valid_mask
    ]

    rows_g = np.asarray(
        final_interval_rows,
        dtype=object
    )[valid_mask]

    # ----------------------------------------
    # Interval midpoint phase
    # bins are 0..9 with centers
    # 0.05, 0.15, ..., 0.95
    #
    # Adjacent interval midpoint:
    # 0.10, 0.20, ..., 0.90
    # ----------------------------------------

    phase_mid = np.asarray([
        0.5 * (
            bin_centers[
                row["bin_a"]
            ]
            +
            bin_centers[
                row["bin_b"]
            ]
        )
        for row in rows_g
    ], dtype=np.float64)

    theta = (
        2.0
        * np.pi
        * phase_mid
    )

    # Phase-only Fourier design.
    # Two harmonics gives enough flexibility
    # to represent shared cell-cycle shape.
    P_g = np.column_stack([
        np.ones(
            len(theta)
        ),
        np.sin(theta),
        np.cos(theta),
        np.sin(
            2.0 * theta
        ),
        np.cos(
            2.0 * theta
        ),
    ])

    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_g
        )
        if q != "non-targeting"
    ], dtype=object)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=2026
    )

    oof_pred_phase = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    for _, val_cond_idx in kf.split(
        pert_conditions
    ):

        val_conditions = pert_conditions[
            val_cond_idx
        ]

        val_fold = np.isin(
            groups_g,
            val_conditions
        )

        train_fold = ~val_fold

        # ------------------------------------
        # Ordinary least squares phase-only
        # model on training conditions
        # ------------------------------------

        coef_phase, _, _, _ = (
            np.linalg.lstsq(
                P_g[
                    train_fold
                ],
                r_g[
                    train_fold
                ],
                rcond=None
            )
        )

        oof_pred_phase[
            val_fold
        ] = (
            P_g[
                val_fold
            ]
            @ coef_phase
        )

    eval_mask = (
        groups_g
        != "non-targeting"
    )

    assert np.all(
        np.isfinite(
            oof_pred_phase[
                eval_mask
            ]
        )
    )

    y_eval = r_g[
        eval_mask
    ]

    pred_eval = oof_pred_phase[
        eval_mask
    ]

    ss_res = np.sum(
        (
            y_eval
            - pred_eval
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y_eval
            - np.mean(
                y_eval
            )
        ) ** 2
    )

    phase_r2 = (
        1.0
        - ss_res / ss_tot
    )

    grn_r2 = float(
        oof_gene_summary.loc[
            oof_gene_summary[
                "gene"
            ] == g,
            "oof_r2"
        ].iloc[0]
    )

    phase_only_records.append({
        "gene": g,
        "grn_oof_r2":
            grn_r2,
        "phase_only_oof_r2":
            phase_r2,
        "incremental_r2":
            grn_r2
            - phase_r2,
    })


phase_only_summary = pd.DataFrame(
    phase_only_records
)


print(
    "Phase-only OOF R^2 summary:"
)

print(
    phase_only_summary[
        "phase_only_oof_r2"
    ].describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nGRN minus phase-only R^2:"
)

print(
    phase_only_summary[
        "incremental_r2"
    ].describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nFraction GRN > phase-only:"
)

print(
    np.mean(
        phase_only_summary[
            "incremental_r2"
        ] > 0
    )
)


print(
    "\nTACC3:"
)

print(
    phase_only_summary[
        phase_only_summary[
            "gene"
        ] == "TACC3"
    ].to_string(
        index=False
    )
)


print(
    "\nGenes with largest GRN gain "
    "over phase-only:"
)

print(
    phase_only_summary
    .sort_values(
        "incremental_r2",
        ascending=False
    )
    .head(15)
    .to_string(
        index=False
    )
)
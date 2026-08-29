"""
RPE1 stage 5: empirical rank/conditioning and randomized perturbation paths.

The final identifiability analysis uses the exact integral design including
the dt/basal column, no centering, and fixed full-design RMS scaling.

This file was mechanically extracted from archive/Replogle_cloud.ipynb.
Notebook cell numbers are preserved below to make provenance auditable.
"""


# ============================================================================
# SOURCE NOTEBOOK INDEX 105: Cell 100. Correct identifiability design
# ============================================================================

# ============================================
# Cell 100. Correct identifiability design
#
# Full exact-integral design:
#   [dt, integral regulator features]
#
# IMPORTANT:
# - include the dt/intercept column
# - NO centering
# - only invertible column scaling
# ============================================

import numpy as np


X_ident_raw = np.asarray(
    X_final,
    dtype=np.float64
)

n_rows_ident, n_cols_ident = (
    X_ident_raw.shape
)


print(
    "Raw identifiability design shape:",
    X_ident_raw.shape
)

print(
    "Expected columns:",
    1 + len(regulator_genes_vc)
)


# --------------------------------------------
# Raw numerical rank
# --------------------------------------------

rank_raw = np.linalg.matrix_rank(
    X_ident_raw
)

print(
    "\nRaw numerical rank:",
    rank_raw,
    "/",
    n_cols_ident
)


# --------------------------------------------
# Column RMS scaling ONLY
#
# This is an invertible diagonal transform,
# so it preserves structural rank.
# --------------------------------------------

ident_col_scale = np.sqrt(
    np.mean(
        X_ident_raw ** 2,
        axis=0
    )
)

print(
    "\nInvalid/zero column scales:",
    np.sum(
        (~np.isfinite(ident_col_scale))
        |
        (ident_col_scale <= 0)
    )
)

assert np.all(
    np.isfinite(
        ident_col_scale
    )
)

assert np.all(
    ident_col_scale > 0
)


X_ident_scaled = (
    X_ident_raw
    / ident_col_scale[None, :]
)


rank_scaled = np.linalg.matrix_rank(
    X_ident_scaled
)

print(
    "Scaled numerical rank:",
    rank_scaled,
    "/",
    n_cols_ident
)

print(
    "Rank preserved:",
    rank_scaled == rank_raw
)


# --------------------------------------------
# Singular values
# --------------------------------------------

s_ident = np.linalg.svd(
    X_ident_scaled,
    compute_uv=False
)

print(
    "\nSingular values:"
)

print(
    "largest:",
    s_ident[0]
)

print(
    "median:",
    np.median(
        s_ident
    )
)

print(
    "smallest:",
    s_ident[-1]
)

print(
    "singular condition number:",
    s_ident[0]
    / s_ident[-1]
)


print(
    "\nSmallest 10 singular values:"
)

print(
    s_ident[-10:]
)


# --------------------------------------------
# Scaled information matrix
#
# I = X^T X / n
#
# Scaling changes units but not PD/rank.
# --------------------------------------------

I_ident = (
    X_ident_scaled.T
    @ X_ident_scaled
) / n_rows_ident

eig_ident = np.linalg.eigvalsh(
    I_ident
)

lambda_min_ident = eig_ident[0]
lambda_max_ident = eig_ident[-1]

print(
    "\nScaled information matrix:"
)

print(
    "lambda_min:",
    lambda_min_ident
)

print(
    "lambda_max:",
    lambda_max_ident
)

print(
    "condition number:",
    lambda_max_ident
    / lambda_min_ident
)


# --------------------------------------------
# Verify first column really is dt
# --------------------------------------------

dt_from_rows = np.asarray([
    row["delta_t"]
    for row in final_interval_rows
], dtype=np.float64)

print(
    "\nFirst column equals interval dt:",
    np.allclose(
        X_ident_raw[:, 0],
        dt_from_rows,
        rtol=0,
        atol=1e-14
    )
)

print(
    "dt min / median / max:",
    np.min(
        X_ident_raw[:, 0]
    ),
    np.median(
        X_ident_raw[:, 0]
    ),
    np.max(
        X_ident_raw[:, 0]
    )
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 106: Cell 101. Correct randomized perturbation
# ============================================================================

# ============================================
# Cell 101. Correct randomized perturbation
# identifiability path
#
# - exact integral design
# - include dt column
# - NO centering
# - fixed full-design RMS scaling
# - NT always included
# ============================================

import numpy as np
import pandas as pd


rng = np.random.default_rng(
    2026
)

n_random_orders = 100

k_grid_ident = np.asarray([
    0,
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
    120,
    len(final_perturbations_vc),
], dtype=int)


row_conditions_ident = np.asarray([
    row["condition"]
    for row in final_interval_rows
], dtype=object)


perturbations_ident = np.asarray(
    final_perturbations_vc,
    dtype=object
)


ident_path_records = []
first_full_rank = []


for rep in range(
    n_random_orders
):

    order = rng.permutation(
        perturbations_ident
    )

    first_full = None


    # ----------------------------------------
    # Search first k giving full rank
    # ----------------------------------------

    for k in range(
        len(order) + 1
    ):

        selected_conditions = np.concatenate([
            np.asarray(
                ["non-targeting"],
                dtype=object
            ),
            order[:k]
        ])

        row_mask = np.isin(
            row_conditions_ident,
            selected_conditions
        )

        X_sub = X_ident_scaled[
            row_mask
        ]

        rank_sub = np.linalg.matrix_rank(
            X_sub
        )

        if (
            rank_sub == n_cols_ident
            and first_full is None
        ):
            first_full = k
            break


    if first_full is None:
        first_full = np.nan

    first_full_rank.append(
        first_full
    )


    # ----------------------------------------
    # Fixed checkpoint path
    # ----------------------------------------

    for k in k_grid_ident:

        selected_conditions = np.concatenate([
            np.asarray(
                ["non-targeting"],
                dtype=object
            ),
            order[:k]
        ])

        row_mask = np.isin(
            row_conditions_ident,
            selected_conditions
        )

        X_sub = X_ident_scaled[
            row_mask
        ]

        n_sub_rows = X_sub.shape[0]

        rank_sub = np.linalg.matrix_rank(
            X_sub
        )


        # ------------------------------------
        # lambda_min of empirical information
        # ------------------------------------

        s_sub = np.linalg.svd(
            X_sub,
            compute_uv=False
        )

        lambda_max_sub = (
            s_sub[0] ** 2
            / n_sub_rows
        )

        # If fewer rows than columns or rank
        # deficient, lambda_min is exactly zero
        # in the full p x p information matrix.
        if rank_sub < n_cols_ident:

            lambda_min_sub = 0.0
            cond_sub = np.inf

        else:

            lambda_min_sub = (
                s_sub[-1] ** 2
                / n_sub_rows
            )

            cond_sub = (
                lambda_max_sub
                / lambda_min_sub
            )


        ident_path_records.append({
            "rep":
                rep,

            "n_perturbations":
                int(k),

            "n_conditions":
                int(k + 1),

            "n_rows":
                int(n_sub_rows),

            "rank":
                int(rank_sub),

            "full_rank":
                bool(
                    rank_sub
                    == n_cols_ident
                ),

            "lambda_min":
                float(
                    lambda_min_sub
                ),

            "condition_number":
                float(
                    cond_sub
                ),
        })


ident_path_df = pd.DataFrame(
    ident_path_records
)

first_full_rank = np.asarray(
    first_full_rank,
    dtype=float
)


# ============================================
# Summaries
# ============================================

print(
    "First-full-rank perturbation count:"
)

print(
    pd.Series(
        first_full_rank
    ).describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
)


print(
    "\nCheckpoint summary:"
)


summary_rows = []

for k in k_grid_ident:

    sub = ident_path_df[
        ident_path_df[
            "n_perturbations"
        ] == k
    ]

    positive_lambda = sub[
        "lambda_min"
    ].to_numpy()

    summary_rows.append({
        "k":
            int(k),

        "median_rank":
            float(
                np.median(
                    sub["rank"]
                )
            ),

        "rank_q10":
            float(
                np.quantile(
                    sub["rank"],
                    0.10
                )
            ),

        "rank_q90":
            float(
                np.quantile(
                    sub["rank"],
                    0.90
                )
            ),

        "full_rank_fraction":
            float(
                np.mean(
                    sub["full_rank"]
                )
            ),

        "lambda_min_median":
            float(
                np.median(
                    positive_lambda
                )
            ),

        "lambda_min_q10":
            float(
                np.quantile(
                    positive_lambda,
                    0.10
                )
            ),

        "lambda_min_q90":
            float(
                np.quantile(
                    positive_lambda,
                    0.90
                )
            ),
    })


ident_path_summary = pd.DataFrame(
    summary_rows
)

print(
    ident_path_summary.to_string(
        index=False
    )
)


print(
    "\nFull-data endpoint check:"
)

full_endpoint = ident_path_summary[
    ident_path_summary[
        "k"
    ] == len(
        final_perturbations_vc
    )
].iloc[0]

print(
    "full-rank fraction:",
    full_endpoint[
        "full_rank_fraction"
    ]
)

print(
    "lambda_min median:",
    full_endpoint[
        "lambda_min_median"
    ]
)

print(
    "Cell 100 lambda_min:",
    lambda_min_ident
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 107: Cell 102. Save corrected identifiability
# ============================================================================

# ============================================
# Cell 102. Save corrected identifiability
# results together with final GRN outputs
# ============================================

import numpy as np
import os


final_checkpoint_v2_path = (
    "/home/featurize/work/project1/"
    "replogle_final_exact_integral_grn_v2.npz"
)


np.savez_compressed(
    final_checkpoint_v2_path,

    # ----------------------------------------
    # Final exact-integral GRN
    # ----------------------------------------

    A_hat_integral=A_hat_integral,
    c_hat_integral=c_hat_integral,

    response_genes=np.asarray(
        response_genes,
        dtype=str
    ),

    regulator_genes=np.asarray(
        regulator_genes_vc,
        dtype=str
    ),

    global_alpha_integral=np.asarray(
        global_alpha_integral
    ),

    X_final=X_final,
    Y_final=Y_final,

    integral_n_nonzero=np.asarray(
        integral_fit_summary[
            "n_nonzero"
        ],
        dtype=int
    ),

    integral_skill=np.asarray(
        integral_fit_summary[
            "skill_vs_intercept"
        ],
        dtype=float
    ),

    integral_mse=np.asarray(
        integral_fit_summary[
            "integral_mse"
        ],
        dtype=float
    ),

    # ----------------------------------------
    # Strict phase-residual validation
    # ----------------------------------------

    global_alpha_resid=np.asarray(
        global_alpha_resid
    ),

    validation_genes=np.asarray(
        final_resid_oof_summary[
            "gene"
        ],
        dtype=str
    ),

    validation_incremental_r2=np.asarray(
        final_resid_oof_summary[
            "incremental_r2"
        ],
        dtype=float
    ),

    validation_relative_mse=np.asarray(
        final_resid_oof_summary[
            "relative_mse_vs_phase"
        ],
        dtype=float
    ),

    validation_mean_nonzero=np.asarray(
        final_resid_oof_summary[
            "mean_fold_nonzero"
        ],
        dtype=float
    ),

    validation_calibration_flag=np.asarray(
        final_resid_oof_summary[
            "used_for_alpha_calibration"
        ],
        dtype=bool
    ),

    # ----------------------------------------
    # Corrected identifiability analysis
    # ----------------------------------------

    ident_col_scale=ident_col_scale,

    ident_full_rank=np.asarray(
        rank_scaled
    ),

    ident_full_lambda_min=np.asarray(
        lambda_min_ident
    ),

    ident_full_lambda_max=np.asarray(
        lambda_max_ident
    ),

    ident_full_condition_number=np.asarray(
        lambda_max_ident
        / lambda_min_ident
    ),

    ident_first_full_rank=first_full_rank,

    ident_k_grid=k_grid_ident,

    ident_path_k=np.asarray(
        ident_path_df[
            "n_perturbations"
        ],
        dtype=int
    ),

    ident_path_rep=np.asarray(
        ident_path_df[
            "rep"
        ],
        dtype=int
    ),

    ident_path_rank=np.asarray(
        ident_path_df[
            "rank"
        ],
        dtype=int
    ),

    ident_path_full_rank=np.asarray(
        ident_path_df[
            "full_rank"
        ],
        dtype=bool
    ),

    ident_path_lambda_min=np.asarray(
        ident_path_df[
            "lambda_min"
        ],
        dtype=float
    ),

    ident_path_condition_number=np.asarray(
        ident_path_df[
            "condition_number"
        ],
        dtype=float
    ),
)


print(
    "Saved:",
    final_checkpoint_v2_path
)

print(
    "Exists:",
    os.path.exists(
        final_checkpoint_v2_path
    )
)

print(
    "Size MB:",
    os.path.getsize(
        final_checkpoint_v2_path
    ) / 1024**2
)


# --------------------------------------------
# Immediate reload checks
# --------------------------------------------

check_v2 = np.load(
    final_checkpoint_v2_path,
    allow_pickle=False
)

print(
    "\nA exact:",
    np.array_equal(
        check_v2[
            "A_hat_integral"
        ],
        A_hat_integral
    )
)

print(
    "Validation exact:",
    np.array_equal(
        check_v2[
            "validation_incremental_r2"
        ],
        final_resid_oof_summary[
            "incremental_r2"
        ].to_numpy()
    )
)

print(
    "First-full-rank exact:",
    np.array_equal(
        check_v2[
            "ident_first_full_rank"
        ],
        first_full_rank
    )
)

print(
    "Full lambda_min exact:",
    float(
        check_v2[
            "ident_full_lambda_min"
        ]
    )
    == lambda_min_ident
)

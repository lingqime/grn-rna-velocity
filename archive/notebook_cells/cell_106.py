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
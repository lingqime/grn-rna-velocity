# ============================================
# Cell 75. Random-order robustness of
# perturbation-assisted rank recovery
# ============================================

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)

n_repeats = 100

# Evaluate at selected perturbation counts.
# Include dense coverage around the expected
# full-rank transition.
k_grid = np.array([
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
    151,
], dtype=int)

perturbations = np.asarray(
    final_perturbations_vc,
    dtype=object
)

records = []

for rep in range(n_repeats):

    order = rng.permutation(
        perturbations
    )

    for k in k_grid:

        included = set(
            ["non-targeting"]
            + list(order[:k])
        )

        mask = np.fromiter(
            (
                q in included
                for q in row_conditions
            ),
            dtype=bool,
            count=len(row_conditions)
        )

        Xk = Xz_all[
            mask, :
        ]

        rank_k = np.linalg.matrix_rank(
            Xk
        )

        if rank_k < len(
            regulator_genes_vc
        ):
            lambda_min_k = 0.0

        else:
            s = np.linalg.svd(
                Xk,
                compute_uv=False
            )

            lambda_min_k = (
                s[-1] ** 2
                / Xk.shape[0]
            )

        records.append({
            "repeat": rep,
            "n_perturbations": k,
            "n_rows": Xk.shape[0],
            "rank": rank_k,
            "lambda_min": lambda_min_k,
        })


rank_robustness = pd.DataFrame(
    records
)

summary = (
    rank_robustness
    .groupby(
        "n_perturbations",
        observed=True
    )
    .agg(
        rank_median=(
            "rank",
            "median"
        ),
        rank_q10=(
            "rank",
            lambda x:
                np.quantile(x, 0.10)
        ),
        rank_q90=(
            "rank",
            lambda x:
                np.quantile(x, 0.90)
        ),
        full_rank_fraction=(
            "rank",
            lambda x:
                np.mean(
                    x
                    == len(
                        regulator_genes_vc
                    )
                )
        ),
        lambda_min_median=(
            "lambda_min",
            "median"
        ),
        lambda_min_q10=(
            "lambda_min",
            lambda x:
                np.quantile(x, 0.10)
        ),
        lambda_min_q90=(
            "lambda_min",
            lambda x:
                np.quantile(x, 0.90)
        ),
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False
    )
)


# --------------------------------------------
# For each random ordering, find the first
# perturbation count where full rank appears.
# --------------------------------------------

first_full_rank = []

for rep in range(n_repeats):

    order = rng.permutation(
        perturbations
    )

    first_k = None

    for k in range(
        0,
        len(perturbations) + 1
    ):

        included = set(
            ["non-targeting"]
            + list(order[:k])
        )

        mask = np.fromiter(
            (
                q in included
                for q in row_conditions
            ),
            dtype=bool,
            count=len(row_conditions)
        )

        rank_k = np.linalg.matrix_rank(
            Xz_all[
                mask, :
            ]
        )

        if rank_k == len(
            regulator_genes_vc
        ):
            first_k = k
            break

    first_full_rank.append(
        first_k
    )


first_full_rank = np.asarray(
    first_full_rank,
    dtype=float
)

print(
    "\nFirst full-rank perturbation count:"
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
# ============================================
# Cell 74. Perturbation-assisted identifiability:
# rank growth as conditions are added
# ============================================

import numpy as np
import pandas as pd

# ------------------------------------------------
# Use the SAME global centering/scaling from Cell 73
# so all subsets live in the same coordinate system.
# ------------------------------------------------

Xz_all = (
    X_final[:, 1:]
    - reg_mean[None, :]
) / reg_std[None, :]

row_conditions = np.asarray([
    r["condition"]
    for r in final_interval_rows
], dtype=object)

# Fixed condition order:
# NT first, then the frozen perturbation order
condition_order = list(
    final_conditions_vc
)

results = []

for k in range(
    1,
    len(condition_order) + 1
):

    included = set(
        condition_order[:k]
    )

    mask = np.array([
        q in included
        for q in row_conditions
    ])

    Xk = Xz_all[
        mask, :
    ]

    s = np.linalg.svd(
        Xk,
        compute_uv=False
    )

    rank_k = np.linalg.matrix_rank(
        Xk
    )

    # lambda_min of X'X/n is exactly zero
    # whenever rank < number of regulators.
    if rank_k < Xk.shape[1]:
        lambda_min_k = 0.0
        cond_info_k = np.inf
    else:
        eig = (
            s ** 2
            / Xk.shape[0]
        )

        lambda_min_k = float(
            np.min(eig)
        )

        cond_info_k = float(
            np.max(eig)
            / np.min(eig)
        )

    results.append({
        "n_conditions": k,
        "n_perturbations": k - 1,
        "n_rows": Xk.shape[0],
        "rank": rank_k,
        "lambda_min": lambda_min_k,
        "condition_number_info":
            cond_info_k,
    })


identifiability_path = pd.DataFrame(
    results
)

print(
    "NT only:"
)

print(
    identifiability_path.iloc[0]
)

print(
    "\nAll conditions:"
)

print(
    identifiability_path.iloc[-1]
)


# ------------------------------------------------
# First point at which full rank is reached
# ------------------------------------------------

full_rank_rows = (
    identifiability_path[
        identifiability_path[
            "rank"
        ] == len(
            regulator_genes_vc
        )
    ]
)

if len(full_rank_rows) > 0:

    first_full = (
        full_rank_rows.iloc[0]
    )

    print(
        "\nFirst full-rank point:"
    )

    print(
        first_full
    )

else:

    print(
        "\nFull rank was never reached."
    )


# ------------------------------------------------
# Selected checkpoints
# ------------------------------------------------

checkpoints = [
    1,
    5,
    10,
    20,
    40,
    80,
    120,
    len(condition_order),
]

checkpoints = [
    k
    for k in checkpoints
    if k <= len(
        condition_order
    )
]

print(
    "\nSelected checkpoints:"
)

print(
    identifiability_path[
        identifiability_path[
            "n_conditions"
        ].isin(
            checkpoints
        )
    ].to_string(
        index=False
    )
)
# ============================================
# Cell 89. TACC3 condition-permutation null
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

test_gene = "TACC3"

gi = np.where(
    response_genes == test_gene
)[0][0]

valid_mask = valid_row_masks[
    test_gene
]

dt_g = np.asarray(
    X_final[
        valid_mask, 0
    ],
    dtype=np.float64
)

Z_g = np.asarray(
    X_final[
        valid_mask, 1:
    ],
    dtype=np.float64
) / dt_g[:, None]

r_g = np.asarray(
    Y_final[
        valid_mask, gi
    ],
    dtype=np.float64
) / dt_g

groups_g = row_conditions[
    valid_mask
]

pert_conditions = np.asarray([
    q for q in np.unique(groups_g)
    if q != "non-targeting"
], dtype=object)


# --------------------------------------------
# Build ONE condition-level permutation
# --------------------------------------------

rng = np.random.default_rng(
    2026
)

permuted_conditions = rng.permutation(
    pert_conditions
)

condition_map = dict(
    zip(
        pert_conditions,
        permuted_conditions
    )
)

# NT maps to itself
condition_map[
    "non-targeting"
] = "non-targeting"


# --------------------------------------------
# Construct permuted regulator design:
#
# response rows for condition q receive
# regulator trajectory rows from permuted q'
#
# Match rows by bin interval index.
# --------------------------------------------

interval_keys = np.asarray([
    (
        row["condition"],
        row["bin_a"],
        row["bin_b"]
    )
    for row in np.asarray(
        final_interval_rows,
        dtype=object
    )[valid_mask]
], dtype=object)

lookup = {}

for i, (
    q,
    bin_a,
    bin_b
) in enumerate(
    interval_keys
):
    lookup[
        (
            q,
            int(bin_a),
            int(bin_b)
        )
    ] = i


Z_perm = np.full_like(
    Z_g,
    np.nan
)

usable_perm = np.zeros(
    len(Z_g),
    dtype=bool
)

for i, (
    q,
    bin_a,
    bin_b
) in enumerate(
    interval_keys
):

    q_source = condition_map[q]

    key = (
        q_source,
        int(bin_a),
        int(bin_b)
    )

    if key in lookup:

        j = lookup[key]

        Z_perm[
            i, :
        ] = Z_g[
            j, :
        ]

        usable_perm[
            i
        ] = True


print(
    "Permuted rows retained:",
    usable_perm.sum(),
    "/",
    len(usable_perm)
)

print(
    "Retention fraction:",
    usable_perm.mean()
)

print(
    "NaN/inf in retained Z_perm:",
    (
        ~np.isfinite(
            Z_perm[
                usable_perm
            ]
        )
    ).any()
)


# --------------------------------------------
# Evaluate only rows for which matching
# source condition/bin interval exists
# --------------------------------------------

Z_null = Z_perm[
    usable_perm
]

r_null = r_g[
    usable_perm
]

groups_null = groups_g[
    usable_perm
]

pert_null = np.asarray([
    q for q in np.unique(groups_null)
    if q != "non-targeting"
], dtype=object)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=2026
)

oof_pred_null = np.full(
    len(r_null),
    np.nan,
    dtype=np.float64
)

for _, val_cond_idx in kf.split(
    pert_null
):

    val_conditions = pert_null[
        val_cond_idx
    ]

    val_fold = np.isin(
        groups_null,
        val_conditions
    )

    train_fold = ~val_fold

    Z_train = Z_null[
        train_fold
    ]

    Z_val = Z_null[
        val_fold
    ]

    r_train = r_null[
        train_fold
    ]

    r_val = r_null[
        val_fold
    ]

    x_scaler = StandardScaler()

    Z_train_z = x_scaler.fit_transform(
        Z_train
    )

    Z_val_z = x_scaler.transform(
        Z_val
    )

    r_mean = np.mean(
        r_train
    )

    r_std = np.std(
        r_train,
        ddof=0
    )

    r_train_z = (
        r_train
        - r_mean
    ) / r_std

    model = Lasso(
        alpha=global_alpha,
        fit_intercept=True,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z_train_z,
        r_train_z
    )

    pred_z = model.predict(
        Z_val_z
    )

    oof_pred_null[
        val_fold
    ] = (
        r_mean
        + r_std * pred_z
    )


eval_mask = (
    groups_null
    != "non-targeting"
)

y_eval = r_null[
    eval_mask
]

p_eval = oof_pred_null[
    eval_mask
]

ss_res = np.sum(
    (
        y_eval
        - p_eval
    ) ** 2
)

ss_tot = np.sum(
    (
        y_eval
        - np.mean(y_eval)
    ) ** 2
)

null_oof_r2 = (
    1.0
    - ss_res / ss_tot
)

print(
    "\nObserved TACC3 OOF R^2:",
    float(
        oof_gene_summary.loc[
            oof_gene_summary[
                "gene"
            ] == "TACC3",
            "oof_r2"
        ].iloc[0]
    )
)

print(
    "Permuted-condition OOF R^2:",
    null_oof_r2
)

print(
    "R^2 drop:",
    float(
        oof_gene_summary.loc[
            oof_gene_summary[
                "gene"
            ] == "TACC3",
            "oof_r2"
        ].iloc[0]
    )
    - null_oof_r2
)
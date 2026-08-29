# ============================================
# Cell 92. Strict cross-fitted phase-residual
# OOF test for TACC3
#
# Question:
# After removing phase structure using
# TRAINING CONDITIONS ONLY, do regulator
# residuals predict held-out response residuals?
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

Z_g = Z_rate[
    valid_mask
]

r_g = R_rate[
    valid_mask,
    gi
]

P_g = P_all[
    valid_mask
]

groups_g = row_conditions[
    valid_mask
]


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

oof_y_res = np.full(
    len(r_g),
    np.nan
)

oof_pred_res = np.full(
    len(r_g),
    np.nan
)

oof_zero_pred = np.full(
    len(r_g),
    np.nan
)

fold_records = []


for fold, (_, val_cond_idx) in enumerate(
    kf.split(
        pert_conditions
    ),
    start=1
):

    val_conditions = pert_conditions[
        val_cond_idx
    ]

    val_fold = np.isin(
        groups_g,
        val_conditions
    )

    train_fold = ~val_fold

    # ----------------------------------------
    # 1. Learn phase components on TRAIN only
    # ----------------------------------------

    phase_coef_Z, _, _, _ = (
        np.linalg.lstsq(
            P_g[
                train_fold
            ],
            Z_g[
                train_fold
            ],
            rcond=None
        )
    )

    phase_coef_r, _, _, _ = (
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


    # ----------------------------------------
    # 2. Residualize train and validation
    # using TRAIN-fitted phase model
    # ----------------------------------------

    Z_train_res = (
        Z_g[
            train_fold
        ]
        -
        P_g[
            train_fold
        ]
        @ phase_coef_Z
    )

    Z_val_res = (
        Z_g[
            val_fold
        ]
        -
        P_g[
            val_fold
        ]
        @ phase_coef_Z
    )

    r_train_res = (
        r_g[
            train_fold
        ]
        -
        P_g[
            train_fold
        ]
        @ phase_coef_r
    )

    r_val_res = (
        r_g[
            val_fold
        ]
        -
        P_g[
            val_fold
        ]
        @ phase_coef_r
    )


    # ----------------------------------------
    # 3. Training-only scaling
    # ----------------------------------------

    x_scaler = StandardScaler()

    Z_train_z = (
        x_scaler.fit_transform(
            Z_train_res
        )
    )

    Z_val_z = (
        x_scaler.transform(
            Z_val_res
        )
    )

    r_scale = np.std(
        r_train_res,
        ddof=0
    )

    assert (
        np.isfinite(r_scale)
        and r_scale > 0
    )

    # Phase residuals already have
    # training mean approximately zero.
    r_train_z = (
        r_train_res
        / r_scale
    )


    # ----------------------------------------
    # 4. Frozen-alpha sparse regression
    #
    # No intercept:
    # phase projection already contains
    # the constant component.
    # ----------------------------------------

    model = Lasso(
        alpha=global_alpha,
        fit_intercept=False,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z_train_z,
        r_train_z
    )

    pred_res = (
        r_scale
        * model.predict(
            Z_val_z
        )
    )


    oof_y_res[
        val_fold
    ] = r_val_res

    oof_pred_res[
        val_fold
    ] = pred_res

    # Null after phase removal:
    # no regulatory contribution
    oof_zero_pred[
        val_fold
    ] = 0.0


    sse_model = np.sum(
        (
            r_val_res
            - pred_res
        ) ** 2
    )

    sse_null = np.sum(
        r_val_res ** 2
    )

    fold_records.append({
        "fold": fold,
        "n_val_conditions":
            len(val_conditions),
        "n_val_rows":
            int(
                val_fold.sum()
            ),
        "relative_mse_vs_phase":
            sse_model
            / sse_null,
        "incremental_r2":
            1.0
            - sse_model
            / sse_null,
        "n_nonzero":
            int(
                np.count_nonzero(
                    model.coef_
                )
            ),
    })


resid_fold_tacc3 = pd.DataFrame(
    fold_records
)


# --------------------------------------------
# Aggregate held-out perturbation result
# --------------------------------------------

eval_mask = (
    groups_g
    != "non-targeting"
)

y_eval = oof_y_res[
    eval_mask
]

p_eval = oof_pred_res[
    eval_mask
]

assert np.all(
    np.isfinite(
        y_eval
    )
)

assert np.all(
    np.isfinite(
        p_eval
    )
)

sse_model = np.sum(
    (
        y_eval
        - p_eval
    ) ** 2
)

sse_phase_only = np.sum(
    y_eval ** 2
)

incremental_r2 = (
    1.0
    - sse_model
    / sse_phase_only
)


print(
    "TACC3 cross-fitted "
    "phase-residual evaluation:"
)

print(
    resid_fold_tacc3.to_string(
        index=False
    )
)

print(
    "\nOverall incremental R^2 "
    "beyond phase-only:",
    incremental_r2
)

print(
    "Overall relative MSE "
    "vs phase-only:",
    sse_model
    / sse_phase_only
)

print(
    "\nMean fold nonzero:",
    resid_fold_tacc3[
        "n_nonzero"
    ].mean()
)

print(
    "All held-out perturbation "
    "residuals predicted:",
    np.all(
        np.isfinite(
            oof_pred_res[
                eval_mask
            ]
        )
    )
)
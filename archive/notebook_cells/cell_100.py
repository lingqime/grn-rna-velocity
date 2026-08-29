# ============================================
# Cell 95. Final phase-residualized OOF
# validation with frozen calibrated alpha
#
# Primary validation summary:
# 395 genes NOT used for alpha calibration.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


final_resid_oof_records = []


for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

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
        np.nan,
        dtype=np.float64
    )

    oof_pred_res = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    fold_nonzero = []


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
        # Training-only phase projections
        # ------------------------------------

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

        # ------------------------------------
        # Training-only scaling
        # ------------------------------------

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

        r_train_z = (
            r_train_res
            / r_scale
        )

        # ------------------------------------
        # Frozen calibrated residual alpha
        # ------------------------------------

        model = Lasso(
            alpha=global_alpha_resid,
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

        fold_nonzero.append(
            np.count_nonzero(
                model.coef_
            )
        )


    # ----------------------------------------
    # Held-out perturbation rows only
    # ----------------------------------------

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

    sse_phase = np.sum(
        y_eval ** 2
    )

    relative_mse = (
        sse_model
        / sse_phase
    )

    incremental_r2 = (
        1.0
        - relative_mse
    )

    final_resid_oof_records.append({
        "gene":
            g,
        "used_for_alpha_calibration":
            g in set(
                calibration_genes
            ),
        "n_eval_rows":
            int(
                eval_mask.sum()
            ),
        "n_eval_conditions":
            len(
                pert_conditions
            ),
        "incremental_r2":
            incremental_r2,
        "relative_mse_vs_phase":
            relative_mse,
        "mean_fold_nonzero":
            np.mean(
                fold_nonzero
            ),
    })


    if (
        (gi + 1) % 25 == 0
        or gi == 0
        or gi + 1
        == len(response_genes)
    ):
        print(
            f"[{gi + 1:03d}/"
            f"{len(response_genes)}] "
            f"{g} done"
        )


final_resid_oof_summary = pd.DataFrame(
    final_resid_oof_records
)


# --------------------------------------------
# Primary: non-calibration genes
# --------------------------------------------

noncal_mask = (
    ~final_resid_oof_summary[
        "used_for_alpha_calibration"
    ]
)

noncal = final_resid_oof_summary[
    noncal_mask
]

cal = final_resid_oof_summary[
    ~noncal_mask
]


print(
    "\nPRIMARY: non-calibration genes:"
)

print(
    "n =",
    len(noncal)
)

print(
    noncal[
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
    "\nFraction incremental R^2 > 0:"
)

print(
    np.mean(
        noncal[
            "incremental_r2"
        ] > 0
    )
)

print(
    "\nRelative MSE vs phase-only:"
)

print(
    noncal[
        "relative_mse_vs_phase"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)

print(
    "\nMean fold nonzero:"
)

print(
    noncal[
        "mean_fold_nonzero"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


# --------------------------------------------
# Calibration panel shown separately
# --------------------------------------------

print(
    "\nCALIBRATION PANEL:"
)

print(
    "n =",
    len(cal)
)

print(
    cal[
        "incremental_r2"
    ].describe()
)


# --------------------------------------------
# Full set, secondary summary
# --------------------------------------------

print(
    "\nALL 426 genes:"
)

print(
    final_resid_oof_summary[
        "incremental_r2"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)

print(
    "\nAll-gene fraction > 0:"
)

print(
    np.mean(
        final_resid_oof_summary[
            "incremental_r2"
        ] > 0
    )
)


print(
    "\nBest 10 non-calibration genes:"
)

print(
    noncal
    .sort_values(
        "incremental_r2",
        ascending=False
    )
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nWorst 10 non-calibration genes:"
)

print(
    noncal
    .sort_values(
        "incremental_r2"
    )
    .head(10)
    .to_string(
        index=False
    )
)
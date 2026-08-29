# ============================================
# Cell 94c. Calibrate alpha for the
# phase-residualized validation problem
#
# Fixed 31-gene panel from Cell 82.
# NT always remains in training.
# Phase projection is training-only.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


alpha_grid_resid = np.logspace(
    -4,
    0,
    40
)

resid_alpha_records = []


for pi, g in enumerate(
    calibration_genes
):

    gi = np.where(
        response_genes == g
    )[0][0]

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

    # ----------------------------------------
    # Construct folds once for this gene.
    # All alpha values see identical folds.
    # ----------------------------------------

    fold_data = []

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

        assert np.any(
            groups_g[
                train_fold
            ] == "non-targeting"
        )

        assert not np.any(
            groups_g[
                val_fold
            ] == "non-targeting"
        )

        # ------------------------------------
        # Training-only phase models
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

        # ------------------------------------
        # Apply training phase model to
        # train and validation
        # ------------------------------------

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
        # Training-only predictor scaling
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

        # ------------------------------------
        # Training-only response scale
        # ------------------------------------

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

        fold_data.append({
            "Z_train_z":
                Z_train_z,
            "Z_val_z":
                Z_val_z,
            "r_train_z":
                r_train_z,
            "r_val_res":
                r_val_res,
            "r_scale":
                r_scale,
        })


    # ----------------------------------------
    # Evaluate alpha grid
    # ----------------------------------------

    for alpha in alpha_grid_resid:

        sse_model = 0.0
        sse_phase = 0.0

        fold_nonzero = []

        for fd in fold_data:

            model = Lasso(
                alpha=float(alpha),
                fit_intercept=False,
                max_iter=20000,
                tol=1e-6,
                selection="cyclic",
            )

            model.fit(
                fd["Z_train_z"],
                fd["r_train_z"]
            )

            pred_res = (
                fd["r_scale"]
                * model.predict(
                    fd["Z_val_z"]
                )
            )

            sse_model += np.sum(
                (
                    fd["r_val_res"]
                    - pred_res
                ) ** 2
            )

            sse_phase += np.sum(
                fd["r_val_res"] ** 2
            )

            fold_nonzero.append(
                np.count_nonzero(
                    model.coef_
                )
            )

        relative_mse = (
            sse_model
            / sse_phase
        )

        resid_alpha_records.append({
            "gene":
                g,
            "alpha":
                float(alpha),
            "relative_mse":
                relative_mse,
            "incremental_r2":
                1.0
                - relative_mse,
            "mean_nonzero":
                np.mean(
                    fold_nonzero
                ),
        })


    print(
        f"[{pi + 1:02d}/"
        f"{len(calibration_genes)}] "
        f"{g} done"
    )


resid_alpha_cv = pd.DataFrame(
    resid_alpha_records
)


# --------------------------------------------
# Equal weighting across the 31 genes
# --------------------------------------------

resid_alpha_summary = (
    resid_alpha_cv
    .groupby(
        "alpha",
        as_index=False
    )
    .agg(
        mean_relative_mse=(
            "relative_mse",
            "mean"
        ),
        median_relative_mse=(
            "relative_mse",
            "median"
        ),
        mean_incremental_r2=(
            "incremental_r2",
            "mean"
        ),
        mean_nonzero=(
            "mean_nonzero",
            "mean"
        ),
    )
    .sort_values(
        "mean_relative_mse"
    )
    .reset_index(
        drop=True
    )
)


best_resid_row = (
    resid_alpha_summary.iloc[0]
)

global_alpha_resid = float(
    best_resid_row[
        "alpha"
    ]
)


print(
    "\nBest residualized alpha:",
    global_alpha_resid
)

print(
    "\nTop 10 alpha values:"
)

print(
    resid_alpha_summary
    .head(10)
    .to_string(
        index=False
    )
)

print(
    "\nBest-alpha mean incremental R^2:",
    float(
        best_resid_row[
            "mean_incremental_r2"
        ]
    )
)

print(
    "Best-alpha median relative MSE:",
    float(
        best_resid_row[
            "median_relative_mse"
        ]
    )
)

print(
    "Best-alpha mean nonzero:",
    float(
        best_resid_row[
            "mean_nonzero"
        ]
    )
)

print(
    "\nOld diagnostic alpha:",
    global_alpha
)

print(
    "New / old alpha ratio:",
    global_alpha_resid
    / global_alpha
)
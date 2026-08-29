# ============================================
# Cell 94. Recalibrate alpha for the
# phase-residualized validation problem
#
# Same fixed 31-gene calibration panel.
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
    # Precompute each fold's correctly
    # cross-fitted residualized data
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

        phase_coef_Z, _, _, _ = (
            np.linalg.lstsq(
                P_g[train_fold],
                Z_g[train_fold],
                rcond=None
            )
        )

        phase_coef_r, _, _, _ = (
            np.linalg.lstsq(
                P_g[train_fold],
                r_g[train_fold],
                rcond=None
            )
        )

        Z_train_res = (
            Z_g[train_fold]
            -
            P_g[train_fold]
            @ phase_coef_Z
        )

        Z_val_res = (
            Z_g[val_fold]
            -
            P_g[val_fold]
            @ phase_coef_Z
        )

        r_train_res = (
            r_g[train_fold]
            -
            P_g[train_fold]
            @ phase_coef_r
        )

        r_val_res = (
            r_g[val_fold]
            -
            P_g[val_fold]
            @ phase_coef_r
        )

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

        fold_data.append((
            Z_train_z,
            Z_val_z,
            r_train_z,
            r_val_res,
            r_scale
        ))

    # ----------------------------------------
    # Evaluate full alpha grid
    # ----------------------------------------

    for alpha in alpha_grid_resid:

        sse_model = 0.0
        sse_phase = 0.0
        nonzero = []

        for (
            Z_train_z,
            Z_val_z,
            r_train_z,
            r_val_res,
            r_scale
        ) in fold_data:

            model = Lasso(
                alpha=alpha,
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

            sse_model += np.sum(
                (
                    r_val_res
                    - pred_res
                ) ** 2
            )

            sse_phase += np.sum(
                r_val_res ** 2
            )

            nonzero.append(
                np.count_nonzero(
                    model.coef_
                )
            )

        resid_alpha_records.append({
            "gene": g,
            "alpha": alpha,
            "relative_mse":
                sse_model
                / sse_phase,
            "incremental_r2":
                1.0
                - sse_model
                / sse_phase,
            "mean_nonzero":
                np.mean(nonzero),
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
# Equal weight per calibration gene
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
)


best_resid_row = (
    resid_alpha_summary
    .iloc[0]
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
    best_resid_row[
        "mean_incremental_r2"
    ]
)

print(
    "Best-alpha mean nonzero:",
    best_resid_row[
        "mean_nonzero"
    ]
)
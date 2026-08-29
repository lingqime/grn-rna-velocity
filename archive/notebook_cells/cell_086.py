# ============================================
# Cell 83. Calibrate dimensionless alpha
# across the representative 31-gene panel
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

alpha_grid_global = np.logspace(
    -4,
    0,
    40
)

panel_cv_records = []

for gene_num, g in enumerate(
    panel_genes,
    start=1
):

    gi = np.where(
        response_genes == g
    )[0][0]

    valid_mask = valid_row_masks[
        g
    ]

    # Rate-form design
    dt_g = X_final[
        valid_mask,
        0
    ]

    Z_g = (
        X_final[
            valid_mask,
            1:
        ]
        / dt_g[:, None]
    )

    r_g = (
        Y_final[
            valid_mask,
            gi
        ]
        / dt_g
    )

    groups_g = row_conditions[
        valid_mask
    ]

    gkf_g = GroupKFold(
        n_splits=5
    )

    folds_g = list(
        gkf_g.split(
            Z_g,
            r_g,
            groups=groups_g
        )
    )

    for alpha in alpha_grid_global:

        fold_mse = []
        fold_nonzero = []

        for train_idx, val_idx in folds_g:

            Z_train = Z_g[
                train_idx
            ]

            Z_val = Z_g[
                val_idx
            ]

            r_train = r_g[
                train_idx
            ]

            r_val = r_g[
                val_idx
            ]

            # ------------------------------
            # X scaling: training fold only
            # ------------------------------

            x_scaler = StandardScaler()

            Z_train_z = (
                x_scaler.fit_transform(
                    Z_train
                )
            )

            Z_val_z = (
                x_scaler.transform(
                    Z_val
                )
            )

            # ------------------------------
            # y scaling: training fold only
            # ------------------------------

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

            r_val_z = (
                r_val
                - r_mean
            ) / r_std

            model = Lasso(
                alpha=alpha,
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

            fold_mse.append(
                mean_squared_error(
                    r_val_z,
                    pred_z
                )
            )

            fold_nonzero.append(
                np.count_nonzero(
                    model.coef_
                )
            )

        panel_cv_records.append({
            "gene": g,
            "alpha": alpha,
            "mean_mse_std":
                np.mean(
                    fold_mse
                ),
            "std_mse_std":
                np.std(
                    fold_mse,
                    ddof=1
                ),
            "mean_nonzero":
                np.mean(
                    fold_nonzero
                ),
        })

    print(
        f"[{gene_num:02d}/{len(panel_genes)}] "
        f"{g} done"
    )


panel_cv = pd.DataFrame(
    panel_cv_records
)


# --------------------------------------------
# Best alpha separately for each panel gene
# --------------------------------------------

best_panel_rows = []

for g in panel_genes:

    tmp = panel_cv[
        panel_cv["gene"] == g
    ]

    idx_best = tmp[
        "mean_mse_std"
    ].idxmin()

    best_panel_rows.append(
        panel_cv.loc[
            idx_best
        ]
    )

best_panel_alpha = pd.DataFrame(
    best_panel_rows
).reset_index(
    drop=True
)


print(
    "\nPer-gene optimal alpha:"
)

print(
    best_panel_alpha[
        [
            "gene",
            "alpha",
            "mean_mse_std",
            "mean_nonzero",
        ]
    ]
    .sort_values(
        "alpha"
    )
    .to_string(
        index=False
    )
)


print(
    "\nOptimal-alpha summary:"
)

print(
    best_panel_alpha[
        "alpha"
    ].describe(
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
    "\nGenes whose optimum is "
    "at grid boundary:"
)

boundary = (
    np.isclose(
        best_panel_alpha[
            "alpha"
        ],
        alpha_grid_global.min()
    )
    |
    np.isclose(
        best_panel_alpha[
            "alpha"
        ],
        alpha_grid_global.max()
    )
)

print(
    best_panel_alpha.loc[
        boundary,
        [
            "gene",
            "alpha",
            "mean_mse_std",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nBoundary count:",
    int(
        boundary.sum()
    ),
    "/",
    len(
        best_panel_alpha
    )
)
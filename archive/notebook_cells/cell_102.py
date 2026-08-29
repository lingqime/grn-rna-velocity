# ============================================
# Cell 97. Alpha calibration for the EXACT
# integral estimator (Algorithm 5.1)
#
# Model:
#   Y = c * dt + X_reg A
#
# c is unpenalized via fold-specific FWL.
# L1 penalty applies only to A.
#
# Held-out perturbation conditions;
# NT always remains in training.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso


alpha_grid_integral = np.logspace(
    -4,
    0,
    40
)

integral_alpha_records = []


for pi, g in enumerate(
    calibration_genes
):

    gi = np.where(
        response_genes == g
    )[0][0]

    valid_mask = valid_row_masks[g]

    d_g = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_g = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y_g = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )

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

    fold_data = []


    # ----------------------------------------
    # Prepare folds once per gene
    # ----------------------------------------

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


        d_train = d_g[
            train_fold
        ]

        X_train = Xr_g[
            train_fold
        ]

        y_train = y_g[
            train_fold
        ]

        d_val = d_g[
            val_fold
        ]

        X_val = Xr_g[
            val_fold
        ]

        y_val = y_g[
            val_fold
        ]


        # ------------------------------------
        # Training-only FWL projection
        # ------------------------------------

        dd = np.dot(
            d_train,
            d_train
        )

        coef_d_X = (
            d_train @ X_train
        ) / dd

        coef_d_y = (
            d_train @ y_train
        ) / dd


        Xp_train = (
            X_train
            - d_train[:, None]
            * coef_d_X[None, :]
        )

        yp_train = (
            y_train
            - d_train
            * coef_d_y
        )


        # ------------------------------------
        # Scale projected predictors
        # WITHOUT centering
        # ------------------------------------

        x_scale = np.sqrt(
            np.mean(
                Xp_train ** 2,
                axis=0
            )
        )

        assert np.all(
            np.isfinite(
                x_scale
            )
        )

        assert np.all(
            x_scale > 0
        )


        y_scale = np.sqrt(
            np.mean(
                yp_train ** 2
            )
        )

        assert (
            np.isfinite(y_scale)
            and y_scale > 0
        )


        Z_train = (
            Xp_train
            / x_scale[None, :]
        )

        yz_train = (
            yp_train
            / y_scale
        )


        fold_data.append({
            "d_train":
                d_train,
            "X_train":
                X_train,
            "y_train":
                y_train,

            "d_val":
                d_val,
            "X_val":
                X_val,
            "y_val":
                y_val,

            "dd":
                dd,
            "x_scale":
                x_scale,
            "y_scale":
                y_scale,
            "Z_train":
                Z_train,
            "yz_train":
                yz_train,
        })


    # ----------------------------------------
    # Evaluate alpha grid
    # ----------------------------------------

    for alpha in alpha_grid_integral:

        sse_model = 0.0
        sse_baseline = 0.0

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
                fd["Z_train"],
                fd["yz_train"]
            )


            # --------------------------------
            # Back-transform A
            # --------------------------------

            A_fold = (
                fd["y_scale"]
                * model.coef_
                / fd["x_scale"]
            )


            # --------------------------------
            # Exact unpenalized c on training
            # --------------------------------

            c_fold = (
                fd["d_train"]
                @ (
                    fd["y_train"]
                    - fd["X_train"]
                    @ A_fold
                )
            ) / fd["dd"]


            # --------------------------------
            # Held-out integral prediction
            # --------------------------------

            pred_val = (
                fd["d_val"]
                * c_fold
                + fd["X_val"]
                @ A_fold
            )


            # --------------------------------
            # Baseline:
            # intercept-only integral model
            # fitted on training rows
            # --------------------------------

            c0_fold = (
                fd["d_train"]
                @ fd["y_train"]
            ) / fd["dd"]

            pred0_val = (
                fd["d_val"]
                * c0_fold
            )


            sse_model += np.sum(
                (
                    fd["y_val"]
                    - pred_val
                ) ** 2
            )

            sse_baseline += np.sum(
                (
                    fd["y_val"]
                    - pred0_val
                ) ** 2
            )

            fold_nonzero.append(
                np.count_nonzero(
                    A_fold
                )
            )


        relative_mse = (
            sse_model
            / sse_baseline
        )

        integral_alpha_records.append({
            "gene":
                g,
            "alpha":
                float(alpha),
            "relative_mse":
                relative_mse,
            "skill_vs_intercept":
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


integral_alpha_cv = pd.DataFrame(
    integral_alpha_records
)


# --------------------------------------------
# Equal weighting across calibration genes
# --------------------------------------------

integral_alpha_summary = (
    integral_alpha_cv
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
        mean_skill=(
            "skill_vs_intercept",
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


best_integral_row = (
    integral_alpha_summary.iloc[0]
)

global_alpha_integral = float(
    best_integral_row[
        "alpha"
    ]
)


print(
    "\nBest exact-integral alpha:",
    global_alpha_integral
)

print(
    "\nTop 10 alpha values:"
)

print(
    integral_alpha_summary
    .head(10)
    .to_string(
        index=False
    )
)

print(
    "\nBest-alpha mean skill:",
    float(
        best_integral_row[
            "mean_skill"
        ]
    )
)

print(
    "Best-alpha median relative MSE:",
    float(
        best_integral_row[
            "median_relative_mse"
        ]
    )
)

print(
    "Best-alpha mean nonzero:",
    float(
        best_integral_row[
            "mean_nonzero"
        ]
    )
)

print(
    "\nRate-form alpha:",
    global_alpha
)

print(
    "Residual-validation alpha:",
    global_alpha_resid
)
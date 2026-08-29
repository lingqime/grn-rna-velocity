# ============================================
# Cell 81. TACC3 grouped CV with BOTH
# predictor and response scaling
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

alpha_grid_scaled = np.logspace(
    -4,
    0,
    40
)

cv_scaled_records = []

for alpha in alpha_grid_scaled:

    fold_mse_std = []
    fold_nonzero = []

    for train_idx, val_idx in fold_indices:

        Z_train = Z_test[
            train_idx
        ]

        Z_val = Z_test[
            val_idx
        ]

        r_train = r_test[
            train_idx
        ]

        r_val = r_test[
            val_idx
        ]

        # ------------------------------------
        # Predictor scaling:
        # training fold ONLY
        # ------------------------------------

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

        # ------------------------------------
        # Response scaling:
        # training fold ONLY
        # ------------------------------------

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

        fold_mse_std.append(
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

    cv_scaled_records.append({
        "alpha": alpha,
        "mean_mse_std":
            np.mean(
                fold_mse_std
            ),
        "std_mse_std":
            np.std(
                fold_mse_std,
                ddof=1
            ),
        "mean_nonzero":
            np.mean(
                fold_nonzero
            ),
    })


cv_tacc3_scaled = pd.DataFrame(
    cv_scaled_records
)

best_scaled_idx = (
    cv_tacc3_scaled[
        "mean_mse_std"
    ].idxmin()
)

best_scaled_row = (
    cv_tacc3_scaled.loc[
        best_scaled_idx
    ]
)


print(
    "Best scaled alpha:"
)

print(
    best_scaled_row
)


print(
    "\nTop 10 scaled alpha values:"
)

print(
    cv_tacc3_scaled
    .sort_values(
        "mean_mse_std"
    )
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nGrid boundary check:"
)

print(
    "best alpha =",
    best_scaled_row[
        "alpha"
    ]
)

print(
    "grid min   =",
    alpha_grid_scaled.min()
)

print(
    "grid max   =",
    alpha_grid_scaled.max()
)

print(
    "best at boundary:",
    (
        best_scaled_idx
        == cv_tacc3_scaled.index[0]
        or
        best_scaled_idx
        == cv_tacc3_scaled.index[-1]
    )
)


# --------------------------------------------
# Compare with the old unscaled-response fit
#
# Rough equivalence:
# alpha_raw / SD(response)
# --------------------------------------------

print(
    "\nTACC3 full-data response SD:",
    np.std(
        r_test,
        ddof=0
    )
)

print(
    "Old raw alpha / response SD:",
    best_alpha
    / np.std(
        r_test,
        ddof=0
    )
)
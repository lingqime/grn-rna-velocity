# ============================================
# Cell 78. Grouped-CV Lasso for TACC3
# Select lambda without condition leakage
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ------------------------------------------------
# Convert integral equation to rate form:
#
# Y / dt = c_g + A_g * (integral S / dt)
#
# This makes sklearn's unpenalized intercept
# exactly the basal term c_g.
# ------------------------------------------------

dt_test = X_test[:, 0]

Z_test = (
    X_test[:, 1:]
    / dt_test[:, None]
)

r_test = (
    y_test
    / dt_test
)

print(
    "Z_test shape:",
    Z_test.shape
)

print(
    "r_test shape:",
    r_test.shape
)

print(
    "Any NaN/inf:",
    (
        ~np.isfinite(
            Z_test
        )
    ).any()
    or
    (
        ~np.isfinite(
            r_test
        )
    ).any()
)


# ------------------------------------------------
# Lambda grid
#
# sklearn objective:
#
# (1 / (2n)) ||y - Xw||^2
# + alpha ||w||_1
# ------------------------------------------------

alpha_grid = np.logspace(
    -4,
    1,
    40
)

cv_records = []

for alpha in alpha_grid:

    fold_mse = []
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

        # Fit scaler ONLY on training fold
        scaler = StandardScaler()

        Z_train_z = (
            scaler.fit_transform(
                Z_train
            )
        )

        Z_val_z = (
            scaler.transform(
                Z_val
            )
        )

        model = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z_train_z,
            r_train
        )

        pred = model.predict(
            Z_val_z
        )

        fold_mse.append(
            mean_squared_error(
                r_val,
                pred
            )
        )

        fold_nonzero.append(
            np.count_nonzero(
                model.coef_
            )
        )

    cv_records.append({
        "alpha": alpha,
        "mean_mse":
            np.mean(
                fold_mse
            ),
        "std_mse":
            np.std(
                fold_mse,
                ddof=1
            ),
        "mean_nonzero":
            np.mean(
                fold_nonzero
            ),
    })


cv_tacc3 = pd.DataFrame(
    cv_records
)

best_idx = (
    cv_tacc3[
        "mean_mse"
    ].idxmin()
)

best_row = cv_tacc3.loc[
    best_idx
]

print(
    "\nBest CV alpha:"
)

print(
    best_row
)


print(
    "\nTop 10 alpha values by CV MSE:"
)

print(
    cv_tacc3
    .sort_values(
        "mean_mse"
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
    best_row["alpha"]
)

print(
    "grid min   =",
    alpha_grid.min()
)

print(
    "grid max   =",
    alpha_grid.max()
)

print(
    "best at boundary:",
    (
        best_idx
        == cv_tacc3.index[0]
        or
        best_idx
        == cv_tacc3.index[-1]
    )
)
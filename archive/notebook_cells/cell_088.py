# ============================================
# Cell 85. Reconstruct the full 426 x 151 GRN
# using the frozen global dimensionless alpha
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

n_response = len(response_genes)
n_regulator = len(regulator_genes_vc)

A_hat = np.zeros(
    (n_response, n_regulator),
    dtype=np.float64
)

c_hat = np.zeros(
    n_response,
    dtype=np.float64
)

fit_summary_rows = []

for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

    dt_g = np.asarray(
        X_final[
            valid_mask,
            0
        ],
        dtype=np.float64
    )

    Z_g = np.asarray(
        X_final[
            valid_mask,
            1:
        ],
        dtype=np.float64
    ) / dt_g[:, None]

    r_g = np.asarray(
        Y_final[
            valid_mask,
            gi
        ],
        dtype=np.float64
    ) / dt_g

    # ----------------------------------------
    # Predictor scaling on all valid rows
    # ----------------------------------------

    x_scaler = StandardScaler()

    Z_g_z = x_scaler.fit_transform(
        Z_g
    )

    # ----------------------------------------
    # Response scaling
    # ----------------------------------------

    r_mean = np.mean(
        r_g
    )

    r_std = np.std(
        r_g,
        ddof=0
    )

    assert (
        np.isfinite(r_std)
        and r_std > 0
    )

    r_g_z = (
        r_g
        - r_mean
    ) / r_std

    # ----------------------------------------
    # Frozen global-alpha Lasso
    # ----------------------------------------

    model = Lasso(
        alpha=global_alpha,
        fit_intercept=True,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z_g_z,
        r_g_z
    )

    # ----------------------------------------
    # Back-transform to ORIGINAL model scale
    #
    # r = c_g + Z A_g
    # ----------------------------------------

    A_g = (
        r_std
        * model.coef_
        / x_scaler.scale_
    )

    c_g = (
        r_mean
        + r_std
        * model.intercept_
        - np.sum(
            A_g
            * x_scaler.mean_
        )
    )

    A_hat[
        gi, :
    ] = A_g

    c_hat[
        gi
    ] = c_g

    # ----------------------------------------
    # Original-scale prediction sanity check
    # ----------------------------------------

    pred_original = (
        c_g
        + Z_g @ A_g
    )

    pred_scaled_back = (
        r_mean
        + r_std
        * model.predict(
            Z_g_z
        )
    )

    transform_ok = np.allclose(
        pred_original,
        pred_scaled_back,
        rtol=1e-8,
        atol=1e-8
    )

    ss_res = np.sum(
        (
            r_g
            - pred_original
        ) ** 2
    )

    ss_tot = np.sum(
        (
            r_g
            - np.mean(r_g)
        ) ** 2
    )

    train_r2 = (
        1.0
        - ss_res / ss_tot
    )

    fit_summary_rows.append({
        "gene": g,
        "n_rows":
            len(r_g),
        "response_std":
            r_std,
        "n_nonzero":
            int(
                np.count_nonzero(
                    A_g
                )
            ),
        "train_r2":
            train_r2,
        "transform_ok":
            transform_ok,
    })


fit_summary = pd.DataFrame(
    fit_summary_rows
)


print(
    "A_hat shape:",
    A_hat.shape
)

print(
    "c_hat shape:",
    c_hat.shape
)

print(
    "Global alpha:",
    global_alpha
)

print(
    "\nNaN/inf in A_hat:",
    (
        ~np.isfinite(
            A_hat
        )
    ).any()
)

print(
    "NaN/inf in c_hat:",
    (
        ~np.isfinite(
            c_hat
        )
    ).any()
)

print(
    "All back-transforms exact:",
    fit_summary[
        "transform_ok"
    ].all()
)


print(
    "\nNonzero edges per response gene:"
)

print(
    fit_summary[
        "n_nonzero"
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
    "\nTotal nonzero edges:",
    np.count_nonzero(
        A_hat
    )
)

print(
    "Total possible edges:",
    A_hat.size
)

print(
    "Network density:",
    np.count_nonzero(
        A_hat
    )
    / A_hat.size
)


print(
    "\nTraining R^2 summary "
    "(diagnostic only):"
)

print(
    fit_summary[
        "train_r2"
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
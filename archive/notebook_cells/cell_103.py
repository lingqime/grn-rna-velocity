# ============================================
# Cell 98. FINAL exact-integral GRN fit
#
# Algorithm 5.1 objective:
#
#   Y_g = c_g * dt + X_reg A_g
#
# c_g unpenalized via exact FWL projection.
# L1 penalty only on A_g.
#
# Direct q=g rows remain excluded through
# valid_row_masks.
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


n_response = len(
    response_genes
)

n_regulator = len(
    regulator_genes_vc
)


A_hat_integral = np.zeros(
    (
        n_response,
        n_regulator
    ),
    dtype=np.float64
)

c_hat_integral = np.zeros(
    n_response,
    dtype=np.float64
)


integral_fit_records = []


for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]


    # ----------------------------------------
    # Original integral system
    # ----------------------------------------

    d = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )


    # ----------------------------------------
    # Exact FWL projection of unpenalized dt
    # ----------------------------------------

    dd = np.dot(
        d,
        d
    )

    coef_d_X = (
        d @ Xr
    ) / dd

    coef_d_y = (
        d @ y
    ) / dd


    Xp = (
        Xr
        - d[:, None]
        * coef_d_X[None, :]
    )

    yp = (
        y
        - d * coef_d_y
    )


    # ----------------------------------------
    # RMS scaling only — NO centering
    # ----------------------------------------

    x_scale = np.sqrt(
        np.mean(
            Xp ** 2,
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
            yp ** 2
        )
    )

    assert (
        np.isfinite(y_scale)
        and y_scale > 0
    )


    Z = (
        Xp
        / x_scale[None, :]
    )

    yz = (
        yp
        / y_scale
    )


    # ----------------------------------------
    # Frozen exact-integral alpha
    # ----------------------------------------

    model = Lasso(
        alpha=global_alpha_integral,
        fit_intercept=False,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z,
        yz
    )


    # ----------------------------------------
    # Back-transform regulatory coefficients
    # ----------------------------------------

    A_g = (
        y_scale
        * model.coef_
        / x_scale
    )


    # ----------------------------------------
    # Exact unpenalized basal coefficient
    # conditional on A_g
    # ----------------------------------------

    c_g = (
        d
        @ (
            y
            - Xr @ A_g
        )
    ) / dd


    A_hat_integral[
        gi, :
    ] = A_g

    c_hat_integral[
        gi
    ] = c_g


    # ----------------------------------------
    # Diagnostics on original integral scale
    # ----------------------------------------

    pred = (
        d * c_g
        + Xr @ A_g
    )

    residual = (
        y
        - pred
    )


    # Intercept-only baseline
    c0 = (
        d @ y
    ) / dd

    pred0 = (
        d * c0
    )


    sse = np.sum(
        residual ** 2
    )

    sse0 = np.sum(
        (
            y
            - pred0
        ) ** 2
    )


    integral_fit_records.append({
        "gene":
            g,

        "n_rows":
            len(y),

        "n_nonzero":
            int(
                np.count_nonzero(
                    A_g
                )
            ),

        "integral_mse":
            np.mean(
                residual ** 2
            ),

        "skill_vs_intercept":
            1.0
            - sse / sse0,

        "fwl_intercept_score":
            abs(
                d @ residual
            ),
    })


    if (
        (gi + 1) % 25 == 0
        or gi == 0
        or gi + 1
        == n_response
    ):
        print(
            f"[{gi + 1:03d}/"
            f"{n_response}] "
            f"{g} done"
        )


integral_fit_summary = pd.DataFrame(
    integral_fit_records
)


# ============================================
# Global validation checks
# ============================================

print(
    "\nA_hat_integral shape:",
    A_hat_integral.shape
)

print(
    "c_hat_integral shape:",
    c_hat_integral.shape
)

print(
    "NaN/inf in A:",
    (
        ~np.isfinite(
            A_hat_integral
        )
    ).any()
)

print(
    "NaN/inf in c:",
    (
        ~np.isfinite(
            c_hat_integral
        )
    ).any()
)


print(
    "\nNonzero edges / response gene:"
)

print(
    integral_fit_summary[
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


total_nonzero = np.count_nonzero(
    A_hat_integral
)

total_possible = (
    A_hat_integral.size
)

print(
    "\nTotal nonzero edges:",
    total_nonzero,
    "/",
    total_possible
)

print(
    "Network density:",
    total_nonzero
    / total_possible
)


print(
    "\nTraining skill vs intercept-only:"
)

print(
    integral_fit_summary[
        "skill_vs_intercept"
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
    "\nFWL first-order-condition check:"
)

print(
    "max |dt^T residual|:",
    integral_fit_summary[
        "fwl_intercept_score"
    ].max()
)


# --------------------------------------------
# Compare exact-integral and old rate network
# --------------------------------------------

old_nz = np.count_nonzero(
    A_hat,
    axis=1
)

new_nz = np.count_nonzero(
    A_hat_integral,
    axis=1
)

print(
    "\nOld rate-form mean nonzero:",
    old_nz.mean()
)

print(
    "Exact-integral mean nonzero:",
    new_nz.mean()
)

print(
    "Median change in edges/gene:",
    np.median(
        new_nz
        - old_nz
    )
)
# ============================================
# Cell 96. Exact integral-objective prototype
# for TACC3 using FWL residualization
#
# This matches:
#   y = c_g * dt + X_reg A_g
#
# without dividing by dt.
#
# Diagnostic only: no final alpha calibration yet.
# ============================================

import numpy as np
from sklearn.linear_model import Lasso


test_gene = "TACC3"

gi = np.where(
    response_genes == test_gene
)[0][0]

valid_mask = valid_row_masks[
    test_gene
]


# --------------------------------------------
# Original integral system
# --------------------------------------------

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


print(
    "Rows:",
    len(y)
)

print(
    "dt min / median / max:",
    np.min(d),
    np.median(d),
    np.max(d)
)

print(
    "max/min dt ratio:",
    np.max(d) / np.min(d)
)


# --------------------------------------------
# FWL projection:
# remove the unpenalized column d = dt
#
# Xp and yp are exactly orthogonal to d.
# --------------------------------------------

dd = np.dot(
    d, d
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


print(
    "\nFWL orthogonality:"
)

print(
    "max |d^T Xp|:",
    np.max(
        np.abs(
            d @ Xp
        )
    )
)

print(
    "|d^T yp|:",
    abs(
        d @ yp
    )
)


# --------------------------------------------
# Scale projected predictors WITHOUT centering.
#
# Centering would introduce a constant direction,
# whereas the original unpenalized direction is dt.
# --------------------------------------------

x_scale = np.sqrt(
    np.mean(
        Xp ** 2,
        axis=0
    )
)

zero_scale = (
    ~np.isfinite(x_scale)
    |
    (x_scale <= 0)
)

print(
    "\nZero/invalid predictor scales:",
    zero_scale.sum()
)

assert zero_scale.sum() == 0


# Dimensionless response scale
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


# --------------------------------------------
# Use current residualized alpha ONLY as a
# numerical prototype.
#
# This alpha is NOT yet declared final for
# the exact integral estimator.
# --------------------------------------------

model = Lasso(
    alpha=global_alpha_resid,
    fit_intercept=False,
    max_iter=20000,
    tol=1e-6,
    selection="cyclic",
)

model.fit(
    Z,
    yz
)


# --------------------------------------------
# Back-transform A to original integral units
# --------------------------------------------

A_tacc3_integral = (
    y_scale
    * model.coef_
    / x_scale
)


# Exact unpenalized c conditional on A
c_tacc3_integral = (
    d
    @ (
        y
        - Xr
        @ A_tacc3_integral
    )
) / dd


# --------------------------------------------
# Verify equivalence of FWL and original
# residuals at the fitted coefficients
# --------------------------------------------

residual_original = (
    y
    - d * c_tacc3_integral
    - Xr @ A_tacc3_integral
)

residual_projected = (
    yp
    - Xp @ A_tacc3_integral
)


print(
    "\nBack-transform checks:"
)

print(
    "c_tacc3_integral:",
    c_tacc3_integral
)

print(
    "nonzero A:",
    np.count_nonzero(
        A_tacc3_integral
    )
)

print(
    "max residual difference:",
    np.max(
        np.abs(
            residual_original
            - residual_projected
        )
    )
)

print(
    "original integral MSE:",
    np.mean(
        residual_original ** 2
    )
)

print(
    "projected integral MSE:",
    np.mean(
        residual_projected ** 2
    )
)


# --------------------------------------------
# Explicit first-order condition for
# unpenalized c:
#
# d^T residual must be zero.
# --------------------------------------------

print(
    "\n|d^T final residual|:",
    abs(
        d @ residual_original
    )
)


# --------------------------------------------
# Compare with old rate-form TACC3 fit
# only as a diagnostic
# --------------------------------------------

old_tacc3_row = np.where(
    response_genes == "TACC3"
)[0][0]

print(
    "\nOld rate-form nonzero:",
    np.count_nonzero(
        A_hat[
            old_tacc3_row
        ]
    )
)

print(
    "Exact-integral prototype nonzero:",
    np.count_nonzero(
        A_tacc3_integral
    )
)
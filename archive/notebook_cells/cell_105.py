# ============================================
# Cell 100. Correct identifiability design
#
# Full exact-integral design:
#   [dt, integral regulator features]
#
# IMPORTANT:
# - include the dt/intercept column
# - NO centering
# - only invertible column scaling
# ============================================

import numpy as np


X_ident_raw = np.asarray(
    X_final,
    dtype=np.float64
)

n_rows_ident, n_cols_ident = (
    X_ident_raw.shape
)


print(
    "Raw identifiability design shape:",
    X_ident_raw.shape
)

print(
    "Expected columns:",
    1 + len(regulator_genes_vc)
)


# --------------------------------------------
# Raw numerical rank
# --------------------------------------------

rank_raw = np.linalg.matrix_rank(
    X_ident_raw
)

print(
    "\nRaw numerical rank:",
    rank_raw,
    "/",
    n_cols_ident
)


# --------------------------------------------
# Column RMS scaling ONLY
#
# This is an invertible diagonal transform,
# so it preserves structural rank.
# --------------------------------------------

ident_col_scale = np.sqrt(
    np.mean(
        X_ident_raw ** 2,
        axis=0
    )
)

print(
    "\nInvalid/zero column scales:",
    np.sum(
        (~np.isfinite(ident_col_scale))
        |
        (ident_col_scale <= 0)
    )
)

assert np.all(
    np.isfinite(
        ident_col_scale
    )
)

assert np.all(
    ident_col_scale > 0
)


X_ident_scaled = (
    X_ident_raw
    / ident_col_scale[None, :]
)


rank_scaled = np.linalg.matrix_rank(
    X_ident_scaled
)

print(
    "Scaled numerical rank:",
    rank_scaled,
    "/",
    n_cols_ident
)

print(
    "Rank preserved:",
    rank_scaled == rank_raw
)


# --------------------------------------------
# Singular values
# --------------------------------------------

s_ident = np.linalg.svd(
    X_ident_scaled,
    compute_uv=False
)

print(
    "\nSingular values:"
)

print(
    "largest:",
    s_ident[0]
)

print(
    "median:",
    np.median(
        s_ident
    )
)

print(
    "smallest:",
    s_ident[-1]
)

print(
    "singular condition number:",
    s_ident[0]
    / s_ident[-1]
)


print(
    "\nSmallest 10 singular values:"
)

print(
    s_ident[-10:]
)


# --------------------------------------------
# Scaled information matrix
#
# I = X^T X / n
#
# Scaling changes units but not PD/rank.
# --------------------------------------------

I_ident = (
    X_ident_scaled.T
    @ X_ident_scaled
) / n_rows_ident

eig_ident = np.linalg.eigvalsh(
    I_ident
)

lambda_min_ident = eig_ident[0]
lambda_max_ident = eig_ident[-1]

print(
    "\nScaled information matrix:"
)

print(
    "lambda_min:",
    lambda_min_ident
)

print(
    "lambda_max:",
    lambda_max_ident
)

print(
    "condition number:",
    lambda_max_ident
    / lambda_min_ident
)


# --------------------------------------------
# Verify first column really is dt
# --------------------------------------------

dt_from_rows = np.asarray([
    row["delta_t"]
    for row in final_interval_rows
], dtype=np.float64)

print(
    "\nFirst column equals interval dt:",
    np.allclose(
        X_ident_raw[:, 0],
        dt_from_rows,
        rtol=0,
        atol=1e-14
    )
)

print(
    "dt min / median / max:",
    np.min(
        X_ident_raw[:, 0]
    ),
    np.median(
        X_ident_raw[:, 0]
    ),
    np.max(
        X_ident_raw[:, 0]
    )
)
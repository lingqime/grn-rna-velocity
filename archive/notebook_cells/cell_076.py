# ============================================
# Cell 73. Numerical rank and conditioning
# of the FINAL integral design matrix
# ============================================

import numpy as np

print(
    "X_final shape:",
    X_final.shape
)

# --------------------------------------------
# Standardize regulator columns only.
#
# Intercept/integral-of-1 column is kept
# separate because its scale has a different
# physical meaning.
# --------------------------------------------

X_reg = np.asarray(
    X_final[:, 1:],
    dtype=np.float64
)

reg_mean = X_reg.mean(
    axis=0
)

reg_std = X_reg.std(
    axis=0,
    ddof=0
)

print(
    "\nRegulator columns with zero std:",
    np.sum(
        reg_std == 0
    )
)

X_reg_z = (
    X_reg
    - reg_mean[None, :]
) / reg_std[None, :]

# --------------------------------------------
# Singular values
# --------------------------------------------

singular_values = np.linalg.svd(
    X_reg_z,
    compute_uv=False
)

rank = np.linalg.matrix_rank(
    X_reg_z
)

condition_number = (
    singular_values[0]
    / singular_values[-1]
)

# Information matrix normalized by n
info_eigenvalues = (
    singular_values ** 2
    / X_reg_z.shape[0]
)

lambda_max = np.max(
    info_eigenvalues
)

lambda_min = np.min(
    info_eigenvalues
)

print(
    "\nNumerical rank:",
    rank,
    "/",
    X_reg_z.shape[1]
)

print(
    "\nSingular values:"
)

print(
    "largest =",
    singular_values[0]
)

print(
    "median  =",
    np.median(
        singular_values
    )
)

print(
    "smallest=",
    singular_values[-1]
)

print(
    "\nCondition number:",
    condition_number
)

print(
    "\nStandardized information matrix:"
)

print(
    "lambda_max =",
    lambda_max
)

print(
    "lambda_min =",
    lambda_min
)

print(
    "condition number =",
    lambda_max
    / lambda_min
)


# --------------------------------------------
# Bottom 10 singular values
# --------------------------------------------

print(
    "\nSmallest 10 singular values:"
)

print(
    singular_values[-10:]
)
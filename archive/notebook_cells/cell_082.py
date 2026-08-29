# ============================================
# Cell 79. Final refit for TACC3
# and recover coefficients on original scale
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

best_alpha = float(
    best_row["alpha"]
)

# --------------------------------------------
# Fit scaler on ALL valid TACC3 rows
# now that alpha has been selected by CV
# --------------------------------------------

scaler_final = StandardScaler()

Z_test_z = scaler_final.fit_transform(
    Z_test
)

model_final = Lasso(
    alpha=best_alpha,
    fit_intercept=True,
    max_iter=20000,
    tol=1e-6,
    selection="cyclic",
)

model_final.fit(
    Z_test_z,
    r_test
)

# --------------------------------------------
# Convert coefficients back to original
# regulator scale
#
# r = c + sum_j A_j Z_j
# --------------------------------------------

A_tacc3 = (
    model_final.coef_
    / scaler_final.scale_
)

c_tacc3 = (
    model_final.intercept_
    - np.sum(
        model_final.coef_
        * scaler_final.mean_
        / scaler_final.scale_
    )
)

pred_rate = (
    c_tacc3
    + Z_test @ A_tacc3
)

train_mse = mean_squared_error(
    r_test,
    pred_rate
)

ss_res = np.sum(
    (
        r_test
        - pred_rate
    ) ** 2
)

ss_tot = np.sum(
    (
        r_test
        - np.mean(r_test)
    ) ** 2
)

train_r2 = (
    1.0
    - ss_res / ss_tot
)

nonzero = np.flatnonzero(
    A_tacc3 != 0
)

print(
    "Gene:",
    test_gene
)

print(
    "Selected alpha:",
    best_alpha
)

print(
    "Basal term c_g:",
    c_tacc3
)

print(
    "\nNonzero regulators:",
    len(nonzero),
    "/",
    len(regulator_genes_vc)
)

print(
    "Training MSE:",
    train_mse
)

print(
    "Training R^2:",
    train_r2
)


# --------------------------------------------
# Ranked inferred regulators
# --------------------------------------------

coef_df = pd.DataFrame({
    "regulator":
        np.asarray(
            regulator_genes_vc
        )[nonzero],
    "A":
        A_tacc3[
            nonzero
        ],
})

coef_df[
    "abs_A"
] = np.abs(
    coef_df["A"]
)

coef_df = (
    coef_df
    .sort_values(
        "abs_A",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

print(
    "\nTop inferred TACC3 regulators:"
)

print(
    coef_df
    .head(20)
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Sign counts
# --------------------------------------------

print(
    "\nPositive edges:",
    np.sum(
        A_tacc3 > 0
    )
)

print(
    "Negative edges:",
    np.sum(
        A_tacc3 < 0
    )
)

print(
    "Zero edges:",
    np.sum(
        A_tacc3 == 0
    )
)


# --------------------------------------------
# Exact original-scale prediction check
# --------------------------------------------

pred_sklearn = model_final.predict(
    Z_test_z
)

print(
    "\nBack-transform prediction exact:",
    np.allclose(
        pred_rate,
        pred_sklearn
    )
)
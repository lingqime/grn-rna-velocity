# ============================================
# Cell 84. Select one global dimensionless
# alpha by aggregate held-out CV loss
# ============================================

import numpy as np
import pandas as pd

global_alpha_curve = (
    panel_cv
    .groupby(
        "alpha",
        as_index=False
    )
    .agg(
        mean_gene_mse=(
            "mean_mse_std",
            "mean"
        ),
        median_gene_mse=(
            "mean_mse_std",
            "median"
        ),
        std_gene_mse=(
            "mean_mse_std",
            "std"
        ),
        mean_nonzero=(
            "mean_nonzero",
            "mean"
        ),
        median_nonzero=(
            "mean_nonzero",
            "median"
        ),
    )
    .sort_values(
        "alpha"
    )
    .reset_index(
        drop=True
    )
)

best_global_idx = (
    global_alpha_curve[
        "mean_gene_mse"
    ].idxmin()
)

best_global_row = (
    global_alpha_curve.loc[
        best_global_idx
    ]
)

global_alpha = float(
    best_global_row[
        "alpha"
    ]
)


print(
    "Global alpha selected by "
    "mean standardized held-out MSE:"
)

print(
    best_global_row
)


print(
    "\nTop 10 global alpha candidates:"
)

print(
    global_alpha_curve
    .sort_values(
        "mean_gene_mse"
    )
    .head(10)
    .to_string(
        index=False
    )
)


# --------------------------------------------
# How much worse is the global alpha than
# each gene's own individually optimal alpha?
# --------------------------------------------

gene_optimum = (
    panel_cv
    .groupby("gene")[
        "mean_mse_std"
    ]
    .min()
)

global_gene_loss = (
    panel_cv[
        np.isclose(
            panel_cv["alpha"],
            global_alpha
        )
    ]
    .set_index("gene")[
        "mean_mse_std"
    ]
)

relative_excess = (
    global_gene_loss
    / gene_optimum
    - 1.0
)

print(
    "\nRelative excess CV loss from using "
    "one global alpha:"
)

print(
    relative_excess.describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nGenes most affected by global alpha:"
)

comparison = pd.DataFrame({
    "optimal_mse":
        gene_optimum,
    "global_mse":
        global_gene_loss,
    "relative_excess":
        relative_excess,
}).sort_values(
    "relative_excess",
    ascending=False
)

print(
    comparison
    .head(10)
    .to_string()
)


print(
    "\nGlobal alpha at grid boundary:",
    (
        best_global_idx == 0
        or
        best_global_idx
        == len(
            global_alpha_curve
        ) - 1
    )
)
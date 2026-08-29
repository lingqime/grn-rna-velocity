# ============================================
# Cell 109. Regulator-feature redundancy
# diagnostic
#
# Question:
# Are some regulator columns strongly
# collinear in the exact integral design?
#
# Uses X_final regulator columns.
# NO centering of the model design itself.
#
# Correlation here is diagnostic only.
# ============================================

import numpy as np
import pandas as pd


Xr_diag = np.asarray(
    X_final[:, 1:],
    dtype=np.float64
)


# --------------------------------------------
# Pearson correlation is used only as a
# redundancy diagnostic.
# --------------------------------------------

corr_reg = np.corrcoef(
    Xr_diag,
    rowvar=False
)

np.fill_diagonal(
    corr_reg,
    np.nan
)

abs_corr_reg = np.abs(
    corr_reg
)


# ============================================
# Per-regulator redundancy metrics
# ============================================

max_abs_corr = np.nanmax(
    abs_corr_reg,
    axis=1
)

mean_abs_corr = np.nanmean(
    abs_corr_reg,
    axis=1
)

n_corr_080 = np.sum(
    abs_corr_reg >= 0.80,
    axis=1
)

n_corr_090 = np.sum(
    abs_corr_reg >= 0.90,
    axis=1
)

n_corr_095 = np.sum(
    abs_corr_reg >= 0.95,
    axis=1
)


redundancy_df = pd.DataFrame({
    "regulator_gene":
        np.asarray(
            regulator_genes_vc,
            dtype=str
        ),

    "max_abs_corr":
        max_abs_corr,

    "mean_abs_corr":
        mean_abs_corr,

    "n_abs_corr_ge_080":
        n_corr_080,

    "n_abs_corr_ge_090":
        n_corr_090,

    "n_abs_corr_ge_095":
        n_corr_095,
})


regulator_hub_redundancy = (
    regulator_hub_diagnostic
    .merge(
        redundancy_df,
        on="regulator_gene",
        how="left",
        validate="one_to_one"
    )
)


# ============================================
# Global pairwise correlation distribution
# ============================================

upper = np.triu_indices(
    len(regulator_genes_vc),
    k=1
)

pair_abs_corr = abs_corr_reg[
    upper
]


print(
    "Pairwise |correlation| distribution:"
)

print(
    pd.Series(
        pair_abs_corr
    ).describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


for threshold in [
    0.80,
    0.90,
    0.95,
]:

    print(
        f"\nPairs with |corr| >= "
        f"{threshold:.2f}:",
        int(
            np.sum(
                pair_abs_corr
                >= threshold
            )
        )
    )


# ============================================
# Relationship with stable outdegree
# ============================================

print(
    "\nSpearman correlations with "
    "stable outgoing degree:"
)

cols = [
    "max_abs_corr",
    "mean_abs_corr",
    "n_abs_corr_ge_080",
    "n_abs_corr_ge_090",
    "n_abs_corr_ge_095",
]

print(
    regulator_hub_redundancy[
        [
            "n_stable80_outgoing"
        ]
        + cols
    ]
    .corr(
        method="spearman"
    )
    .loc[
        cols,
        "n_stable80_outgoing"
    ]
    .sort_values(
        ascending=False
    )
)


# ============================================
# Top stable hubs + redundancy
# ============================================

print(
    "\nTop 25 stable hubs + redundancy:"
)

print(
    regulator_hub_redundancy
    .sort_values(
        "n_stable80_outgoing",
        ascending=False
    )
    .head(25)
    [
        [
            "regulator_gene",
            "n_stable80_outgoing",
            "stable_fraction_selected",
            "max_abs_corr",
            "mean_abs_corr",
            "n_abs_corr_ge_080",
            "n_abs_corr_ge_090",
            "n_abs_corr_ge_095",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================
# Most highly correlated regulator pairs
# ============================================

pair_records = []

for i in range(
    len(regulator_genes_vc)
):

    for j in range(
        i + 1,
        len(regulator_genes_vc)
    ):

        pair_records.append({
            "regulator_1":
                str(
                    regulator_genes_vc[i]
                ),

            "regulator_2":
                str(
                    regulator_genes_vc[j]
                ),

            "correlation":
                float(
                    corr_reg[i, j]
                ),

            "abs_correlation":
                float(
                    abs_corr_reg[i, j]
                ),
        })


regulator_pair_corr = pd.DataFrame(
    pair_records
)


print(
    "\nTop 20 most correlated "
    "regulator pairs:"
)

print(
    regulator_pair_corr
    .sort_values(
        "abs_correlation",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)
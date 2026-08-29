# ============================================
# Cell 107. Regulator-level network summary
#
# Rank regulators primarily by:
# - number of stable outgoing edges
# - fraction of selected edges that are stable
#
# Do NOT rank primarily by raw coefficient size.
# ============================================

import numpy as np
import pandas as pd


regulator_summary_records = []


for regulator in regulator_genes_vc:

    sub = edge_table[
        edge_table[
            "regulator_gene"
        ] == regulator
    ]

    selected = sub[
        "full_selected"
    ].to_numpy()

    stable = sub[
        "stable_80"
    ].to_numpy()

    coef = sub[
        "coefficient"
    ].to_numpy(
        dtype=float
    )

    sel_freq = sub[
        "selection_frequency"
    ].to_numpy(
        dtype=float
    )


    n_selected = int(
        selected.sum()
    )

    n_stable = int(
        stable.sum()
    )

    n_stable_positive = int(
        np.sum(
            stable
            & (coef > 0)
        )
    )

    n_stable_negative = int(
        np.sum(
            stable
            & (coef < 0)
        )
    )


    if n_selected > 0:

        stable_fraction_selected = (
            n_stable
            / n_selected
        )

        mean_freq_selected = float(
            np.mean(
                sel_freq[
                    selected
                ]
            )
        )

    else:

        stable_fraction_selected = np.nan
        mean_freq_selected = np.nan


    if n_stable > 0:

        mean_freq_stable = float(
            np.mean(
                sel_freq[
                    stable
                ]
            )
        )

        median_freq_stable = float(
            np.median(
                sel_freq[
                    stable
                ]
            )
        )

    else:

        mean_freq_stable = np.nan
        median_freq_stable = np.nan


    regulator_summary_records.append({
        "regulator_gene":
            str(regulator),

        "n_selected_outgoing":
            n_selected,

        "n_stable80_outgoing":
            n_stable,

        "stable_fraction_selected":
            stable_fraction_selected,

        "mean_selection_freq_selected":
            mean_freq_selected,

        "mean_selection_freq_stable":
            mean_freq_stable,

        "median_selection_freq_stable":
            median_freq_stable,

        "n_stable_positive":
            n_stable_positive,

        "n_stable_negative":
            n_stable_negative,

        "stable_positive_fraction":
            (
                n_stable_positive
                / n_stable
                if n_stable > 0
                else np.nan
            ),

        "stable_negative_fraction":
            (
                n_stable_negative
                / n_stable
                if n_stable > 0
                else np.nan
            ),
    })


regulator_summary = pd.DataFrame(
    regulator_summary_records
)


# --------------------------------------------
# Global distribution
# --------------------------------------------

print(
    "Selected outgoing edges per regulator:"
)

print(
    regulator_summary[
        "n_selected_outgoing"
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
    "\nStable >=0.8 outgoing edges "
    "per regulator:"
)

print(
    regulator_summary[
        "n_stable80_outgoing"
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
    "\nStable fraction among "
    "full-selected outgoing edges:"
)

print(
    regulator_summary[
        "stable_fraction_selected"
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


# --------------------------------------------
# Regulators with most stable outgoing edges
# --------------------------------------------

print(
    "\nTop 25 regulators by "
    "stable outgoing edge count:"
)

print(
    regulator_summary
    .sort_values(
        [
            "n_stable80_outgoing",
            "stable_fraction_selected",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .head(25)
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Regulators with fewest stable outgoing edges
# --------------------------------------------

print(
    "\nBottom 15 regulators by "
    "stable outgoing edge count:"
)

print(
    regulator_summary
    .sort_values(
        [
            "n_stable80_outgoing",
            "stable_fraction_selected",
        ],
        ascending=[
            True,
            True,
        ]
    )
    .head(15)
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Sanity check: totals must match edge table
# --------------------------------------------

print(
    "\nTotal selected outgoing edges:",
    int(
        regulator_summary[
            "n_selected_outgoing"
        ].sum()
    )
)

print(
    "Expected:",
    int(
        edge_table[
            "full_selected"
        ].sum()
    )
)

print(
    "\nTotal stable outgoing edges:",
    int(
        regulator_summary[
            "n_stable80_outgoing"
        ].sum()
    )
)

print(
    "Expected:",
    int(
        edge_table[
            "stable_80"
        ].sum()
    )
)
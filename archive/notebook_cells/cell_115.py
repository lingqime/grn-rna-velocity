# ============================================
# Cell 110. Diagnose the 12 possible self-edges
#
# Goal:
# inspect whether stable self-edges are
# unusually strong / unusually stable and
# whether they are associated with regulator
# redundancy.
#
# No model refitting.
# ============================================

import numpy as np
import pandas as pd


self_edge_diag = (
    edge_table[
        edge_table[
            "self_edge"
        ]
    ]
    .merge(
        redundancy_df,
        on="regulator_gene",
        how="left",
        validate="one_to_one"
    )
    .copy()
)


# --------------------------------------------
# Rank each self-edge among all outgoing edges
# of the same regulator.
#
# Rank 1 = largest |coefficient|
# Rank 1 = highest selection frequency
# --------------------------------------------

coef_ranks = []
freq_ranks = []

for _, row in self_edge_diag.iterrows():

    regulator = row[
        "regulator_gene"
    ]

    response = row[
        "response_gene"
    ]

    outgoing = edge_table[
        edge_table[
            "regulator_gene"
        ] == regulator
    ].copy()


    coef_rank = (
        outgoing[
            "abs_coefficient"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .loc[
            outgoing[
                "response_gene"
            ] == response
        ]
        .iloc[0]
    )


    freq_rank = (
        outgoing[
            "selection_frequency"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .loc[
            outgoing[
                "response_gene"
            ] == response
        ]
        .iloc[0]
    )


    coef_ranks.append(
        int(coef_rank)
    )

    freq_ranks.append(
        int(freq_rank)
    )


self_edge_diag[
    "abs_coef_outgoing_rank"
] = coef_ranks

self_edge_diag[
    "selection_freq_outgoing_rank"
] = freq_ranks


# --------------------------------------------
# Compare self-edge stability against
# non-self selected edges
# --------------------------------------------

selected_self = edge_table[
    edge_table[
        "self_edge"
    ]
    &
    edge_table[
        "full_selected"
    ]
]

selected_nonself = edge_table[
    (~edge_table[
        "self_edge"
    ])
    &
    edge_table[
        "full_selected"
    ]
]


print(
    "Possible self edges:",
    len(
        self_edge_diag
    )
)

print(
    "Selected self edges:",
    len(
        selected_self
    )
)

print(
    "Stable >=0.8 self edges:",
    int(
        selected_self[
            "stable_80"
        ].sum()
    )
)


print(
    "\nSelf-edge selection frequencies:"
)

print(
    self_edge_diag[
        "selection_frequency"
    ].describe()
)


print(
    "\nSelected non-self edge "
    "selection frequencies:"
)

print(
    selected_nonself[
        "selection_frequency"
    ].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nDetailed self-edge diagnostics:"
)

print(
    self_edge_diag[
        [
            "response_gene",
            "regulator_gene",
            "coefficient",
            "selection_frequency",
            "positive_frequency",
            "negative_frequency",
            "sign_consistency",
            "abs_coef_outgoing_rank",
            "selection_freq_outgoing_rank",
            "max_abs_corr",
            "n_abs_corr_ge_080",
        ]
    ]
    .sort_values(
        "selection_frequency",
        ascending=False
    )
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Sign composition
# --------------------------------------------

print(
    "\nSelected self-edge signs:"
)

print(
    selected_self[
        "sign"
    ].value_counts()
    .sort_index()
)


# --------------------------------------------
# Compare stability proportions
# --------------------------------------------

self_stable_fraction = float(
    selected_self[
        "stable_80"
    ].mean()
)

nonself_stable_fraction = float(
    selected_nonself[
        "stable_80"
    ].mean()
)

print(
    "\nStable fraction among "
    "selected self edges:",
    self_stable_fraction
)

print(
    "Stable fraction among "
    "selected non-self edges:",
    nonself_stable_fraction
)

print(
    "Ratio:",
    self_stable_fraction
    / nonself_stable_fraction
)
# ============================================
# Cell 106. Build final edge-level table
#
# One row = one response gene <- regulator edge
# Includes:
# - exact-integral coefficient
# - full-data selection
# - stability frequency
# - positive/negative frequency
# - sign consistency
# - stable >= 0.8 flag
# ============================================

import numpy as np
import pandas as pd


edge_records = []


for gi, response in enumerate(
    response_genes
):

    for rj, regulator in enumerate(
        regulator_genes_vc
    ):

        coef = float(
            A_hat_integral[
                gi, rj
            ]
        )

        sel_freq = float(
            selection_freq[
                gi, rj
            ]
        )

        pos_freq = float(
            positive_freq[
                gi, rj
            ]
        )

        neg_freq = float(
            negative_freq[
                gi, rj
            ]
        )

        sign_cons = float(
            sign_consistency[
                gi, rj
            ]
        )

        full_selected = (
            coef != 0.0
        )

        stable80 = (
            sel_freq >= 0.80
        )

        if coef > 0:
            sign = 1
        elif coef < 0:
            sign = -1
        else:
            sign = 0

        edge_records.append({
            "response_gene":
                str(response),

            "regulator_gene":
                str(regulator),

            "coefficient":
                coef,

            "abs_coefficient":
                abs(coef),

            "sign":
                sign,

            "full_selected":
                full_selected,

            "selection_frequency":
                sel_freq,

            "positive_frequency":
                pos_freq,

            "negative_frequency":
                neg_freq,

            "sign_consistency":
                sign_cons,

            "stable_80":
                stable80,

            "stable_90":
                sel_freq >= 0.90,

            "stable_95":
                sel_freq >= 0.95,

            "stable_100":
                sel_freq == 1.00,

            "self_edge":
                str(response)
                == str(regulator),
        })


edge_table = pd.DataFrame(
    edge_records
)


print(
    "Edge table shape:",
    edge_table.shape
)

print(
    "Expected rows:",
    len(response_genes)
    * len(regulator_genes_vc)
)


print(
    "\nFull selected:",
    int(
        edge_table[
            "full_selected"
        ].sum()
    )
)

print(
    "Stable >=0.8:",
    int(
        edge_table[
            "stable_80"
        ].sum()
    )
)

print(
    "Stable >=0.9:",
    int(
        edge_table[
            "stable_90"
        ].sum()
    )
)

print(
    "Stable >=0.95:",
    int(
        edge_table[
            "stable_95"
        ].sum()
    )
)

print(
    "Stable ==1.0:",
    int(
        edge_table[
            "stable_100"
        ].sum()
    )
)


# --------------------------------------------
# Important invariant:
# all >=0.8 stable edges should be selected
# in the full-data network
# --------------------------------------------

print(
    "\nAll stable >=0.8 edges "
    "selected in full network:",
    bool(
        edge_table.loc[
            edge_table[
                "stable_80"
            ],
            "full_selected"
        ].all()
    )
)


# --------------------------------------------
# Stable-core sign agreement
# --------------------------------------------

stable_edges = edge_table[
    edge_table[
        "stable_80"
    ]
].copy()

stable_sign_agreement = np.where(
    stable_edges[
        "sign"
    ].to_numpy() > 0,
    stable_edges[
        "positive_frequency"
    ].to_numpy(),
    stable_edges[
        "negative_frequency"
    ].to_numpy()
)

print(
    "\nStable-core sign agreement:"
)

print(
    pd.Series(
        stable_sign_agreement
    ).describe(
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
# Self-edge bookkeeping
# Only possible where response gene is also
# one of the 151 regulator genes.
# --------------------------------------------

self_edges = edge_table[
    edge_table[
        "self_edge"
    ]
]

print(
    "\nPossible self edges:",
    len(self_edges)
)

print(
    "Selected self edges:",
    int(
        self_edges[
            "full_selected"
        ].sum()
    )
)

print(
    "Stable >=0.8 self edges:",
    int(
        self_edges[
            "stable_80"
        ].sum()
    )
)


# --------------------------------------------
# Preview strongest stable edges
# by absolute coefficient
# --------------------------------------------

print(
    "\nTop 20 stable edges "
    "by |coefficient|:"
)

print(
    stable_edges
    .sort_values(
        "abs_coefficient",
        ascending=False
    )
    .head(20)
    [
        [
            "response_gene",
            "regulator_gene",
            "coefficient",
            "selection_frequency",
            "sign_consistency",
            "self_edge",
        ]
    ]
    .to_string(
        index=False
    )
)
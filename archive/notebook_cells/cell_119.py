# ============================================
# Cell 114. Freeze corrected primary network
# and rebuild the canonical edge table
#
# From this cell onward:
#
# PRIMARY network:
#   A_primary = A_hat_integral_noself
#
# PRIMARY stability:
#   selection_freq_primary
#     = selection_freq_noself
#
# Self predictors are structurally excluded
# for the 12 response/regulator overlap genes.
# ============================================

import numpy as np
import pandas as pd


# ============================================
# Canonical primary objects
# ============================================

A_primary = (
    A_hat_integral_noself.copy()
)

c_primary = (
    c_hat_integral_noself.copy()
)

selection_freq_primary = (
    selection_freq_noself.copy()
)

positive_freq_primary = (
    positive_freq_noself.copy()
)

negative_freq_primary = (
    negative_freq_noself.copy()
)

sign_consistency_primary = (
    sign_consistency_noself.copy()
)


# ============================================
# Rebuild edge table
# ============================================

edge_records_primary = []


response_arr = np.asarray(
    response_genes,
    dtype=object
)

regulator_arr = np.asarray(
    regulator_genes_vc,
    dtype=object
)


for gi, response in enumerate(
    response_arr
):

    for rj, regulator in enumerate(
        regulator_arr
    ):

        coef = float(
            A_primary[
                gi, rj
            ]
        )

        sel_freq = float(
            selection_freq_primary[
                gi, rj
            ]
        )

        pos_freq = float(
            positive_freq_primary[
                gi, rj
            ]
        )

        neg_freq = float(
            negative_freq_primary[
                gi, rj
            ]
        )

        sign_cons = float(
            sign_consistency_primary[
                gi, rj
            ]
        )

        is_self = (
            str(response)
            == str(regulator)
        )

        edge_records_primary.append({
            "response_gene":
                str(response),

            "regulator_gene":
                str(regulator),

            "coefficient":
                coef,

            "abs_coefficient":
                abs(coef),

            "sign":
                int(
                    np.sign(coef)
                ),

            "full_selected":
                bool(
                    coef != 0
                ),

            "selection_frequency":
                sel_freq,

            "positive_frequency":
                pos_freq,

            "negative_frequency":
                neg_freq,

            "sign_consistency":
                sign_cons,

            "stable_80":
                bool(
                    sel_freq >= 0.80
                ),

            "stable_90":
                bool(
                    sel_freq >= 0.90
                ),

            "stable_95":
                bool(
                    sel_freq >= 0.95
                ),

            "stable_100":
                bool(
                    sel_freq == 1.0
                ),

            "self_edge":
                bool(
                    is_self
                ),
        })


edge_table_primary = pd.DataFrame(
    edge_records_primary
)


# ============================================
# Core checks
# ============================================

print(
    "Primary edge table shape:",
    edge_table_primary.shape
)

print(
    "Expected:",
    (
        len(response_genes)
        * len(regulator_genes_vc),
        15
    )
)


print(
    "\nFull-data selected:",
    int(
        edge_table_primary[
            "full_selected"
        ].sum()
    )
)


print(
    "Stable >=0.8:",
    int(
        edge_table_primary[
            "stable_80"
        ].sum()
    )
)

print(
    "Stable >=0.9:",
    int(
        edge_table_primary[
            "stable_90"
        ].sum()
    )
)

print(
    "Stable >=0.95:",
    int(
        edge_table_primary[
            "stable_95"
        ].sum()
    )
)

print(
    "Stable ==1:",
    int(
        edge_table_primary[
            "stable_100"
        ].sum()
    )
)


# ============================================
# Stable edges should be selected in the
# full-data primary fit
# ============================================

stable_not_selected = edge_table_primary[
    edge_table_primary[
        "stable_80"
    ]
    &
    ~edge_table_primary[
        "full_selected"
    ]
]


print(
    "\nStable >=0.8 but not selected "
    "in full-data fit:",
    len(
        stable_not_selected
    )
)


# ============================================
# Verify no self edge survives
# ============================================

self_primary = edge_table_primary[
    edge_table_primary[
        "self_edge"
    ]
]


print(
    "\nPossible self edges:",
    len(
        self_primary
    )
)

print(
    "Selected self edges:",
    int(
        self_primary[
            "full_selected"
        ].sum()
    )
)

print(
    "Stable self edges:",
    int(
        self_primary[
            "stable_80"
        ].sum()
    )
)


# ============================================
# Sign agreement for stable primary edges
# ============================================

stable_primary = edge_table_primary[
    edge_table_primary[
        "stable_80"
    ]
    &
    edge_table_primary[
        "full_selected"
    ]
].copy()


stable_primary[
    "full_sign_frequency"
] = np.where(
    stable_primary[
        "coefficient"
    ].to_numpy()
    > 0,

    stable_primary[
        "positive_frequency"
    ].to_numpy(),

    stable_primary[
        "negative_frequency"
    ].to_numpy()
)


print(
    "\nStable primary sign agreement:"
)

print(
    stable_primary[
        "full_sign_frequency"
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


# ============================================
# Final explicit consistency checks
# ============================================

print(
    "\nA_primary equals corrected matrix:",
    np.array_equal(
        A_primary,
        A_hat_integral_noself
    )
)

print(
    "Selection-frequency matrix equals "
    "corrected stability matrix:",
    np.array_equal(
        selection_freq_primary,
        selection_freq_noself
    )
)

print(
    "All primary coefficients finite:",
    bool(
        np.all(
            np.isfinite(
                A_primary
            )
        )
    )
)
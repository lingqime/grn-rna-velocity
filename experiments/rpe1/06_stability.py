"""
RPE1 stage 6: perturbation-condition subsampling stability.

The final protocol uses 100 replicates, random 80% perturbation-condition
subsets, non-targeting retained throughout, frozen exact-integral alpha, and
self-excluded treatment for the 12 overlap response genes.

This file was mechanically extracted from archive/Replogle_cloud.ipynb.
Notebook cell numbers are preserved below to make provenance auditable.
"""


# ============================================================================
# SOURCE NOTEBOOK INDEX 109: Cell 104. Full edge-stability analysis
# ============================================================================

# ============================================
# Cell 104. Full edge-stability analysis
#
# 426 response genes
# 151 candidate regulators
# 100 condition-subsampling replicates
#
# Protocol:
# - retain 80% perturbation conditions
# - NT always retained
# - exact integral estimator
# - frozen global_alpha_integral
# - direct q=g exclusion preserved
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


n_stability_reps = 100
stability_fraction = 0.80

rng_stab_full = np.random.default_rng(
    2026
)


n_response = len(
    response_genes
)

n_regulator = len(
    regulator_genes_vc
)


selection_count = np.zeros(
    (
        n_response,
        n_regulator
    ),
    dtype=np.uint16
)

positive_count = np.zeros_like(
    selection_count
)

negative_count = np.zeros_like(
    selection_count
)

edge_count_by_rep = np.zeros(
    (
        n_response,
        n_stability_reps
    ),
    dtype=np.uint16
)


for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

    d_all = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_all = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y_all = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )

    groups_all = row_conditions[
        valid_mask
    ]


    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_all
        )
        if q != "non-targeting"
    ], dtype=object)


    n_keep = int(
        np.floor(
            stability_fraction
            * len(pert_conditions)
        )
    )


    for rep in range(
        n_stability_reps
    ):

        kept_pert = rng_stab_full.choice(
            pert_conditions,
            size=n_keep,
            replace=False
        )

        kept_conditions = np.concatenate([
            np.asarray(
                ["non-targeting"],
                dtype=object
            ),
            kept_pert
        ])

        row_mask = np.isin(
            groups_all,
            kept_conditions
        )


        d = d_all[
            row_mask
        ]

        Xr = Xr_all[
            row_mask
        ]

        y = y_all[
            row_mask
        ]


        # ------------------------------------
        # Exact FWL projection
        # ------------------------------------

        dd = d @ d

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


        # ------------------------------------
        # RMS scaling only, no centering
        # ------------------------------------

        x_scale = np.sqrt(
            np.mean(
                Xp ** 2,
                axis=0
            )
        )

        y_scale = np.sqrt(
            np.mean(
                yp ** 2
            )
        )

        assert np.all(
            np.isfinite(
                x_scale
            )
        )

        assert np.all(
            x_scale > 0
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


        model = Lasso(
            alpha=global_alpha_integral,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z,
            yz
        )


        A_rep = (
            y_scale
            * model.coef_
            / x_scale
        )


        selected = (
            A_rep != 0
        )

        selection_count[
            gi
        ] += selected

        positive_count[
            gi
        ] += (
            A_rep > 0
        )

        negative_count[
            gi
        ] += (
            A_rep < 0
        )

        edge_count_by_rep[
            gi,
            rep
        ] = np.count_nonzero(
            A_rep
        )


    if (
        (gi + 1) % 25 == 0
        or gi == 0
        or gi + 1 == n_response
    ):

        print(
            f"[{gi + 1:03d}/"
            f"{n_response}] "
            f"{g} done"
        )


# ============================================
# Frequencies
# ============================================

selection_freq = (
    selection_count.astype(
        np.float64
    )
    / n_stability_reps
)

positive_freq = (
    positive_count.astype(
        np.float64
    )
    / n_stability_reps
)

negative_freq = (
    negative_count.astype(
        np.float64
    )
    / n_stability_reps
)

sign_consistency = np.maximum(
    positive_freq,
    negative_freq
)


# ============================================
# Global summaries
# ============================================

print(
    "\nEdges per subsampled fit:"
)

print(
    pd.Series(
        edge_count_by_rep.ravel()
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


print(
    "\nSelection-frequency distribution:"
)

print(
    pd.Series(
        selection_freq.ravel()
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
    0.50,
    0.80,
    0.90,
    0.95,
    1.00,
]:

    mask = (
        selection_freq
        >= threshold
    )

    print(
        f"\nSelection frequency >= "
        f"{threshold:.2f}:",
        int(
            mask.sum()
        ),
        "edges"
    )

    print(
        "fraction of all candidate edges:",
        float(
            mask.mean()
        )
    )


# ============================================
# Primary stable-edge threshold: >= 0.8
# ============================================

stable80 = (
    selection_freq >= 0.80
)

print(
    "\nStable >=0.8 edges per response gene:"
)

stable_per_gene = np.sum(
    stable80,
    axis=1
)

print(
    pd.Series(
        stable_per_gene
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


print(
    "\nSign consistency among >=0.8 edges:"
)

print(
    pd.Series(
        sign_consistency[
            stable80
        ]
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


# ============================================
# Agreement with final full-data network
# ============================================

full_selected = (
    A_hat_integral != 0
)

print(
    "\nAmong full-data selected edges:"
)

print(
    "Total:",
    int(
        full_selected.sum()
    )
)

print(
    "selection freq >=0.8:",
    int(
        np.sum(
            full_selected
            & stable80
        )
    )
)

print(
    "fraction stable >=0.8:",
    float(
        np.mean(
            selection_freq[
                full_selected
            ]
            >= 0.8
        )
    )
)


full_sign = np.sign(
    A_hat_integral
)

stable_sign = np.where(
    positive_freq >= negative_freq,
    1,
    -1
)

stable_full_mask = (
    full_selected
    & stable80
)

print(
    "\nFull-data sign agreement among "
    "stable selected edges:"
)

print(
    np.mean(
        full_sign[
            stable_full_mask
        ]
        ==
        stable_sign[
            stable_full_mask
        ]
    )
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 118: Cell 113. Update stability for the 12
# ============================================================================

# ============================================
# Cell 113. Update stability for the 12
# self-excluded overlap response genes
#
# Other 414 response rows are copied exactly
# from Cell 104.
#
# For each overlap response g:
# - exclude direct q = g condition via valid mask
# - exclude predictor g itself
# - 80% perturbation-condition subsampling
# - 100 replicates
# - frozen global_alpha_integral
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


# --------------------------------------------
# Start from original full stability results
# --------------------------------------------

selection_count_noself = (
    selection_count.copy()
)

positive_count_noself = (
    positive_count.copy()
)

negative_count_noself = (
    negative_count.copy()
)

edge_count_by_rep_noself = (
    edge_count_by_rep.copy()
)


# Fresh fixed seed for corrected 12-row analysis
rng_stab_noself = np.random.default_rng(
    2027
)


for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]

    self_rj = np.where(
        np.asarray(
            regulator_genes_vc,
            dtype=object
        ) == g
    )[0][0]


    valid_mask = valid_row_masks[g]


    d_all = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_full_all = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y_all = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )

    groups_all = row_conditions[
        valid_mask
    ]


    # ----------------------------------------
    # Explicit self-predictor exclusion
    # ----------------------------------------

    predictor_keep = np.ones(
        len(regulator_genes_vc),
        dtype=bool
    )

    predictor_keep[
        self_rj
    ] = False


    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_all
        )
        if q != "non-targeting"
    ], dtype=object)


    n_keep = int(
        np.floor(
            stability_fraction
            * len(pert_conditions)
        )
    )


    # Reset this response row
    selection_count_noself[
        gi, :
    ] = 0

    positive_count_noself[
        gi, :
    ] = 0

    negative_count_noself[
        gi, :
    ] = 0

    edge_count_by_rep_noself[
        gi, :
    ] = 0


    for rep in range(
        n_stability_reps
    ):

        kept_pert = rng_stab_noself.choice(
            pert_conditions,
            size=n_keep,
            replace=False
        )

        kept_conditions = np.concatenate([
            np.asarray(
                ["non-targeting"],
                dtype=object
            ),
            kept_pert
        ])

        row_mask = np.isin(
            groups_all,
            kept_conditions
        )


        d = d_all[
            row_mask
        ]

        Xr_full = Xr_full_all[
            row_mask
        ]

        Xr = Xr_full[
            :,
            predictor_keep
        ]

        y = y_all[
            row_mask
        ]


        # ------------------------------------
        # Exact FWL projection
        # ------------------------------------

        dd = d @ d

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


        # ------------------------------------
        # RMS scaling
        # ------------------------------------

        x_scale = np.sqrt(
            np.mean(
                Xp ** 2,
                axis=0
            )
        )

        y_scale = np.sqrt(
            np.mean(
                yp ** 2
            )
        )


        assert np.all(
            np.isfinite(
                x_scale
            )
        )

        assert np.all(
            x_scale > 0
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


        model = Lasso(
            alpha=global_alpha_integral,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z,
            yz
        )


        A_reduced = (
            y_scale
            * model.coef_
            / x_scale
        )


        A_rep = np.zeros(
            len(regulator_genes_vc),
            dtype=np.float64
        )

        A_rep[
            predictor_keep
        ] = A_reduced

        A_rep[
            self_rj
        ] = 0.0


        selected = (
            A_rep != 0
        )


        selection_count_noself[
            gi
        ] += selected.astype(
            np.uint16
        )

        positive_count_noself[
            gi
        ] += (
            A_rep > 0
        ).astype(
            np.uint16
        )

        negative_count_noself[
            gi
        ] += (
            A_rep < 0
        ).astype(
            np.uint16
        )


        edge_count_by_rep_noself[
            gi,
            rep
        ] = np.count_nonzero(
            A_rep
        )


    print(
        g,
        "done"
    )


# ============================================
# Updated frequencies
# ============================================

selection_freq_noself = (
    selection_count_noself.astype(
        np.float64
    )
    / n_stability_reps
)

positive_freq_noself = (
    positive_count_noself.astype(
        np.float64
    )
    / n_stability_reps
)

negative_freq_noself = (
    negative_count_noself.astype(
        np.float64
    )
    / n_stability_reps
)

sign_consistency_noself = np.maximum(
    positive_freq_noself,
    negative_freq_noself
)


# ============================================
# Sanity checks
# ============================================

non_overlap_mask = ~np.isin(
    np.asarray(
        response_genes,
        dtype=object
    ),
    overlap_genes
)


print(
    "\nNon-overlap stability rows unchanged:",
    np.array_equal(
        selection_freq_noself[
            non_overlap_mask
        ],
        selection_freq[
            non_overlap_mask
        ]
    )
)


self_freq_after = []

for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]

    rj = np.where(
        np.asarray(
            regulator_genes_vc,
            dtype=object
        ) == g
    )[0][0]

    self_freq_after.append(
        selection_freq_noself[
            gi, rj
        ]
    )


self_freq_after = np.asarray(
    self_freq_after
)


print(
    "All corrected self-edge frequencies zero:",
    bool(
        np.all(
            self_freq_after == 0
        )
    )
)


# ============================================
# Updated primary-network stability summary
# ============================================

full_selected_noself = (
    A_hat_integral_noself != 0
)

stable80_noself = (
    selection_freq_noself >= 0.80
)


print(
    "\nCorrected total full-data edges:",
    int(
        full_selected_noself.sum()
    )
)

print(
    "Corrected stable >=0.8 edges:",
    int(
        stable80_noself.sum()
    )
)

print(
    "Stable >=0.8 among full-selected:",
    int(
        np.sum(
            full_selected_noself
            & stable80_noself
        )
    )
)

print(
    "Fraction of full-selected edges stable >=0.8:",
    float(
        np.mean(
            selection_freq_noself[
                full_selected_noself
            ]
            >= 0.8
        )
    )
)


# ============================================
# Per-overlap-gene comparison
# ============================================

overlap_stability_records = []


for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]


    old_row_selected = (
        A_hat_integral[
            gi
        ] != 0
    )

    new_row_selected = (
        A_hat_integral_noself[
            gi
        ] != 0
    )


    overlap_stability_records.append({
        "gene":
            str(g),

        "old_n_selected":
            int(
                old_row_selected.sum()
            ),

        "new_n_selected":
            int(
                new_row_selected.sum()
            ),

        "old_n_stable80":
            int(
                np.sum(
                    old_row_selected
                    &
                    (
                        selection_freq[
                            gi
                        ] >= 0.8
                    )
                )
            ),

        "new_n_stable80":
            int(
                np.sum(
                    new_row_selected
                    &
                    (
                        selection_freq_noself[
                            gi
                        ] >= 0.8
                    )
                )
            ),

        "new_mean_freq_selected":
            float(
                np.mean(
                    selection_freq_noself[
                        gi,
                        new_row_selected
                    ]
                )
            ),
    })


overlap_stability_summary = pd.DataFrame(
    overlap_stability_records
)


print(
    "\nOverlap-gene stability after "
    "self exclusion:"
)

print(
    overlap_stability_summary
    .to_string(
        index=False
    )
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 119: Cell 114. Freeze corrected primary network
# ============================================================================

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

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
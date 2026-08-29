# ============================================
# Cell 103. Pilot edge-stability analysis
#
# Exact integral estimator only.
#
# Protocol:
# - subsample perturbation CONDITIONS
# - NT always retained
# - 80% perturbation conditions per replicate
# - frozen global_alpha_integral
# - exact FWL treatment of dt
#
# Pilot:
# - 10 response genes
# - 20 replicates
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


rng_stab = np.random.default_rng(
    2026
)

n_stability_reps_pilot = 20

stability_fraction = 0.80


# --------------------------------------------
# Deterministic pilot genes:
# spread across response_genes + TACC3
# --------------------------------------------

pilot_idx = np.unique(
    np.round(
        np.linspace(
            0,
            len(response_genes) - 1,
            9
        )
    ).astype(int)
)

pilot_genes = list(
    response_genes[
        pilot_idx
    ]
)

if "TACC3" not in pilot_genes:
    pilot_genes.append(
        "TACC3"
    )

pilot_genes = np.asarray(
    pilot_genes,
    dtype=object
)


print(
    "Pilot genes:",
    len(pilot_genes)
)

print(
    pilot_genes
)


# --------------------------------------------
# Outputs
# --------------------------------------------

pilot_selection_count = np.zeros(
    (
        len(pilot_genes),
        len(regulator_genes_vc)
    ),
    dtype=int
)

pilot_positive_count = np.zeros_like(
    pilot_selection_count
)

pilot_negative_count = np.zeros_like(
    pilot_selection_count
)

pilot_edge_counts = np.zeros(
    (
        len(pilot_genes),
        n_stability_reps_pilot
    ),
    dtype=int
)


for pgi, g in enumerate(
    pilot_genes
):

    gi = np.where(
        response_genes == g
    )[0][0]

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
        n_stability_reps_pilot
    ):

        kept_pert = rng_stab.choice(
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
        # RMS scaling only
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

        positive = (
            A_rep > 0
        )

        negative = (
            A_rep < 0
        )


        pilot_selection_count[
            pgi
        ] += selected.astype(int)

        pilot_positive_count[
            pgi
        ] += positive.astype(int)

        pilot_negative_count[
            pgi
        ] += negative.astype(int)

        pilot_edge_counts[
            pgi,
            rep
        ] = np.count_nonzero(
            A_rep
        )


    print(
        f"[{pgi + 1:02d}/"
        f"{len(pilot_genes)}] "
        f"{g} done"
    )


# ============================================
# Summaries
# ============================================

pilot_selection_freq = (
    pilot_selection_count
    / n_stability_reps_pilot
)

pilot_positive_freq = (
    pilot_positive_count
    / n_stability_reps_pilot
)

pilot_negative_freq = (
    pilot_negative_count
    / n_stability_reps_pilot
)


print(
    "\nEdges per replicate:"
)

print(
    pd.Series(
        pilot_edge_counts.ravel()
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
        pilot_selection_freq.ravel()
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


print(
    "\nEdges with selection frequency >= 0.8:",
    np.sum(
        pilot_selection_freq >= 0.8
    )
)

print(
    "Edges with selection frequency == 1:",
    np.sum(
        pilot_selection_freq == 1.0
    )
)


# --------------------------------------------
# Sign consistency among highly stable edges
# --------------------------------------------

stable_mask = (
    pilot_selection_freq >= 0.8
)

sign_consistency = np.maximum(
    pilot_positive_freq,
    pilot_negative_freq
)

if np.any(
    stable_mask
):

    print(
        "\nStable-edge sign consistency:"
    )

    print(
        pd.Series(
            sign_consistency[
                stable_mask
            ]
        ).describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
    )

else:

    print(
        "\nNo edges reached selection frequency >= 0.8"
    )
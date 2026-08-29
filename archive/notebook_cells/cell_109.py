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
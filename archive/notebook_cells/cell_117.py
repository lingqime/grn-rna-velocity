# ============================================
# Cell 112. Refit the 12 overlap response genes
# with self predictor explicitly excluded
#
# Primary exact-integral estimator:
# - same global_alpha_integral
# - same valid rows
# - same exact FWL treatment of dt
# - same RMS scaling
# - ONLY difference:
#     predictor g is removed when response = g
#
# Other 414 response genes remain unchanged.
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


A_hat_integral_noself = (
    A_hat_integral.copy()
)

c_hat_integral_noself = (
    c_hat_integral.copy()
)


noself_refit_records = []


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


    d = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_full = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )


    # ----------------------------------------
    # Explicitly remove same-gene predictor
    # ----------------------------------------

    predictor_keep = np.ones(
        len(regulator_genes_vc),
        dtype=bool
    )

    predictor_keep[
        self_rj
    ] = False


    Xr = Xr_full[
        :,
        predictor_keep
    ]


    # ----------------------------------------
    # Exact FWL projection for unpenalized
    # dt / basal term
    # ----------------------------------------

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


    # ----------------------------------------
    # RMS scaling
    # ----------------------------------------

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


    # ----------------------------------------
    # Reconstruct full 151-vector,
    # forcing self coefficient exactly zero
    # ----------------------------------------

    A_new = np.zeros(
        len(regulator_genes_vc),
        dtype=np.float64
    )

    A_new[
        predictor_keep
    ] = A_reduced

    A_new[
        self_rj
    ] = 0.0


    # ----------------------------------------
    # Recover unpenalized basal coefficient
    # ----------------------------------------

    c_new = (
        d
        @ (
            y
            - Xr_full @ A_new
        )
    ) / dd


    # ----------------------------------------
    # Diagnostics
    # ----------------------------------------

    pred_new = (
        d * c_new
        + Xr_full @ A_new
    )

    residual_new = (
        y - pred_new
    )

    baseline_c = (
        d @ y
    ) / dd

    baseline_pred = (
        d * baseline_c
    )

    mse_new = float(
        np.mean(
            residual_new ** 2
        )
    )

    mse_baseline = float(
        np.mean(
            (
                y
                - baseline_pred
            ) ** 2
        )
    )

    skill_new = (
        1.0
        - mse_new
        / mse_baseline
    )


    A_old = (
        A_hat_integral[
            gi
        ]
    )

    old_selected = (
        A_old != 0
    )

    new_selected = (
        A_new != 0
    )


    changed_selection = np.sum(
        old_selected
        != new_selected
    )


    # Save corrected row
    A_hat_integral_noself[
        gi
    ] = A_new

    c_hat_integral_noself[
        gi
    ] = c_new


    noself_refit_records.append({
        "gene":
            str(g),

        "old_self_coef":
            float(
                A_old[
                    self_rj
                ]
            ),

        "new_self_coef":
            float(
                A_new[
                    self_rj
                ]
            ),

        "old_n_nonzero":
            int(
                np.count_nonzero(
                    A_old
                )
            ),

        "new_n_nonzero":
            int(
                np.count_nonzero(
                    A_new
                )
            ),

        "changed_selection":
            int(
                changed_selection
            ),

        "old_skill":
            float(
                integral_fit_summary.loc[
                    integral_fit_summary[
                        "gene"
                    ] == g,
                    "skill_vs_intercept"
                ].iloc[0]
            ),

        "new_skill":
            float(
                skill_new
            ),

        "skill_change":
            float(
                skill_new
                -
                integral_fit_summary.loc[
                    integral_fit_summary[
                        "gene"
                    ] == g,
                    "skill_vs_intercept"
                ].iloc[0]
            ),

        "dt_residual_orthogonality":
            float(
                abs(
                    d @ residual_new
                )
            ),
    })


noself_refit_summary = pd.DataFrame(
    noself_refit_records
)


print(
    "Corrected overlap genes:",
    len(
        noself_refit_summary
    )
)


print(
    "\nRefit summary:"
)

print(
    noself_refit_summary
    .sort_values(
        "skill_change"
    )
    .to_string(
        index=False
    )
)


# ============================================
# Global network changes
# ============================================

old_selected_all = (
    A_hat_integral != 0
)

new_selected_all = (
    A_hat_integral_noself != 0
)


print(
    "\nOld total edges:",
    int(
        old_selected_all.sum()
    )
)

print(
    "New total edges:",
    int(
        new_selected_all.sum()
    )
)

print(
    "Net edge-count change:",
    int(
        new_selected_all.sum()
        - old_selected_all.sum()
    )
)


print(
    "\nTotal selection-status changes:",
    int(
        np.sum(
            old_selected_all
            != new_selected_all
        )
    )
)


# ============================================
# Verify only the 12 overlap response rows
# changed
# ============================================

overlap_mask_response = np.isin(
    np.asarray(
        response_genes,
        dtype=object
    ),
    overlap_genes
)

print(
    "\nNon-overlap rows unchanged exactly:",
    np.array_equal(
        A_hat_integral_noself[
            ~overlap_mask_response
        ],
        A_hat_integral[
            ~overlap_mask_response
        ]
    )
)


# ============================================
# Verify all possible self coefficients = 0
# ============================================

self_values_after = []

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

    self_values_after.append(
        A_hat_integral_noself[
            gi, rj
        ]
    )


self_values_after = np.asarray(
    self_values_after
)


print(
    "\nAll 12 self coefficients exactly zero:",
    bool(
        np.all(
            self_values_after == 0
        )
    )
)

print(
    "Max |self coefficient|:",
    float(
        np.max(
            np.abs(
                self_values_after
            )
        )
    )
)
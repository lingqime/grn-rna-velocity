# ============================================
# Cell 117. Exact-integral condition-held-out
# OOF prediction
#
# Compare:
#   1. exact-integral GRN estimator
#   2. phase-only 9-interval baseline
#
# IMPORTANT:
# - perturbation conditions held out
# - NT always training
# - q = response gene rows excluded
# - self predictor excluded for 12 overlap genes
# - fold-specific FWL + RMS scaling
# - frozen global_alpha_integral
#
# Outputs:
#   oof_y
#   oof_pred_grn
#   oof_pred_phase
#   oof_fold_id
# ============================================

import numpy as np
from sklearn.linear_model import Lasso


n_rows = X_final.shape[0]
n_genes = len(response_genes)
n_regs = len(regulator_genes_vc)


# ============================================
# OOF storage
# ============================================

oof_y = np.full(
    (n_rows, n_genes),
    np.nan,
    dtype=np.float64
)

oof_pred_grn = np.full(
    (n_rows, n_genes),
    np.nan,
    dtype=np.float64
)

oof_pred_phase = np.full(
    (n_rows, n_genes),
    np.nan,
    dtype=np.float64
)

oof_fold_id = np.full(
    n_rows,
    -1,
    dtype=int
)


response_arr = np.asarray(
    response_genes,
    dtype=object
)

regulator_arr = np.asarray(
    regulator_genes_vc,
    dtype=object
)


# ============================================
# Loop over held-out condition folds
# ============================================

for fold_idx, test_conditions in enumerate(
    condition_folds_cv
):

    base_test_mask = np.isin(
        row_conditions_cv,
        test_conditions
    )

    base_train_mask = ~base_test_mask


    oof_fold_id[
        base_test_mask
    ] = fold_idx


    for gi, g in enumerate(
        response_arr
    ):

        # ------------------------------------
        # Direct intervention handling:
        # remove q = g rows from BOTH train
        # and test for response gene g
        # ------------------------------------

        if g in set(
            final_perturbations_vc
        ):

            direct_mask = (
                row_conditions_cv
                == g
            )

        else:

            direct_mask = np.zeros(
                n_rows,
                dtype=bool
            )


        train_mask = (
            base_train_mask
            & ~direct_mask
        )

        test_mask = (
            base_test_mask
            & ~direct_mask
        )


        # If direct condition g belongs to this
        # test fold, those rows intentionally
        # receive no OOF prediction.
        if not np.any(
            test_mask
        ):
            continue


        # ====================================
        # Training data
        # ====================================

        d_train = np.asarray(
            X_final[
                train_mask, 0
            ],
            dtype=np.float64
        )

        Xr_train_full = np.asarray(
            X_final[
                train_mask, 1:
            ],
            dtype=np.float64
        )

        y_train = np.asarray(
            Y_final[
                train_mask, gi
            ],
            dtype=np.float64
        )


        d_test = np.asarray(
            X_final[
                test_mask, 0
            ],
            dtype=np.float64
        )

        Xr_test_full = np.asarray(
            X_final[
                test_mask, 1:
            ],
            dtype=np.float64
        )

        y_test = np.asarray(
            Y_final[
                test_mask, gi
            ],
            dtype=np.float64
        )


        # ====================================
        # Predictor set
        #
        # For overlap genes, remove the
        # same-gene predictor structurally.
        # ====================================

        predictor_keep = np.ones(
            n_regs,
            dtype=bool
        )

        if g in set(
            overlap_genes
        ):

            self_rj = np.where(
                regulator_arr == g
            )[0][0]

            predictor_keep[
                self_rj
            ] = False


        Xr_train = Xr_train_full[
            :,
            predictor_keep
        ]

        Xr_test = Xr_test_full[
            :,
            predictor_keep
        ]


        # ====================================
        # Exact FWL on TRAINING ONLY
        # ====================================

        dd_train = (
            d_train @ d_train
        )

        coef_d_X = (
            d_train @ Xr_train
        ) / dd_train

        coef_d_y = (
            d_train @ y_train
        ) / dd_train


        Xp_train = (
            Xr_train
            - d_train[:, None]
            * coef_d_X[None, :]
        )

        yp_train = (
            y_train
            - d_train
            * coef_d_y
        )


        # ====================================
        # TRAINING-ONLY RMS scaling
        # ====================================

        x_scale = np.sqrt(
            np.mean(
                Xp_train ** 2,
                axis=0
            )
        )

        y_scale = np.sqrt(
            np.mean(
                yp_train ** 2
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
            np.isfinite(
                y_scale
            )
            and y_scale > 0
        )


        Z_train = (
            Xp_train
            / x_scale[None, :]
        )

        yz_train = (
            yp_train
            / y_scale
        )


        # ====================================
        # Sparse exact-integral fit
        # ====================================

        model = Lasso(
            alpha=global_alpha_integral,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z_train,
            yz_train
        )


        A_reduced = (
            y_scale
            * model.coef_
            / x_scale
        )


        # Recover unpenalized basal coefficient
        c_fold = (
            d_train
            @ (
                y_train
                - Xr_train
                @ A_reduced
            )
        ) / dd_train


        pred_grn = (
            d_test * c_fold
            + Xr_test @ A_reduced
        )


        # ====================================
        # Phase-only baseline
        #
        # Fit 9 phase-interval rates on
        # TRAINING ONLY via least squares.
        # ====================================

        P_train = X_phase_cv[
            train_mask
        ]

        P_test = X_phase_cv[
            test_mask
        ]


        phase_coef = np.linalg.lstsq(
            P_train,
            y_train,
            rcond=None
        )[0]


        pred_phase = (
            P_test @ phase_coef
        )


        # ====================================
        # Store OOF predictions
        # ====================================

        oof_y[
            test_mask,
            gi
        ] = y_test

        oof_pred_grn[
            test_mask,
            gi
        ] = pred_grn

        oof_pred_phase[
            test_mask,
            gi
        ] = pred_phase


    print(
        f"Fold {fold_idx + 1}/5 done"
    )


# ============================================
# Basic diagnostics
# ============================================

print(
    "\nAll rows assigned a fold:",
    bool(
        np.all(
            oof_fold_id >= 0
        )
    )
)


valid_oof = np.isfinite(
    oof_y
)

print(
    "Total finite OOF outcomes:",
    int(
        valid_oof.sum()
    )
)

print(
    "Finite GRN predictions:",
    int(
        np.isfinite(
            oof_pred_grn
        ).sum()
    )
)

print(
    "Finite phase predictions:",
    int(
        np.isfinite(
            oof_pred_phase
        ).sum()
    )
)


print(
    "\nPrediction masks identical:",
    bool(
        np.array_equal(
            np.isfinite(
                oof_y
            ),
            np.isfinite(
                oof_pred_grn
            )
        )
        and
        np.array_equal(
            np.isfinite(
                oof_y
            ),
            np.isfinite(
                oof_pred_phase
            )
        )
    )
)


# ============================================
# Expected missing entries:
# only q = g direct-intervention rows for
# the 12 response genes that are themselves
# perturbation targets.
# ============================================

missing_per_gene = np.sum(
    ~np.isfinite(
        oof_y
    ),
    axis=0
)

print(
    "\nGenes with missing OOF rows:",
    int(
        np.sum(
            missing_per_gene > 0
        )
    )
)

print(
    "Total missing gene-row entries:",
    int(
        missing_per_gene.sum()
    )
)


print(
    "\nMissing rows for overlap genes:"
)

for g in overlap_genes:

    gi = np.where(
        response_arr == g
    )[0][0]

    print(
        g,
        int(
            missing_per_gene[
                gi
            ]
        )
    )
"""
RPE1 stage 7: five-fold condition-held-out validation against phase baseline.

This is conditional equation prediction, not an end-to-end holdout of the
raw VeloCycle preprocessing pipeline. Regression preprocessing is learned from
training conditions only; non-targeting remains in training.

This file was mechanically extracted from archive/Replogle_cloud.ipynb.
Notebook cell numbers are preserved below to make provenance auditable.
"""


# ============================================================================
# SOURCE NOTEBOOK INDEX 121: Cell 116. Prepare condition-held-out CV
# ============================================================================

# ============================================
# Cell 116. Prepare condition-held-out CV
# and phase-only baseline design
#
# No model fitting yet.
#
# CV rule:
# - perturbation conditions are held out
# - non-targeting is ALWAYS training
# - every perturbation appears in test once
#
# Phase-only baseline:
# - 9 adjacent phase intervals
# - one shared rate parameter per interval
# - prediction on integral scale = dt * rate
# ============================================

import numpy as np
import pandas as pd


# ============================================
# Recover row metadata explicitly
# ============================================

row_conditions_cv = np.asarray(
    [
        row["condition"]
        for row in final_interval_rows
    ],
    dtype=object
)

row_bin_a_cv = np.asarray(
    [
        row["bin_a"]
        for row in final_interval_rows
    ],
    dtype=int
)

row_bin_b_cv = np.asarray(
    [
        row["bin_b"]
        for row in final_interval_rows
    ],
    dtype=int
)

row_dt_cv = np.asarray(
    X_final[:, 0],
    dtype=np.float64
)


# ============================================
# Basic interval checks
# ============================================

print(
    "Rows:",
    len(row_conditions_cv)
)

print(
    "Unique conditions:",
    len(
        np.unique(
            row_conditions_cv
        )
    )
)

print(
    "Unique bin_a:",
    np.unique(
        row_bin_a_cv
    )
)

print(
    "Unique bin_b:",
    np.unique(
        row_bin_b_cv
    )
)

print(
    "All intervals adjacent:",
    bool(
        np.all(
            row_bin_b_cv
            == row_bin_a_cv + 1
        )
    )
)


# ============================================
# Create fixed 5-fold perturbation CV
# ============================================

perturbations_cv = np.asarray(
    final_perturbations_vc,
    dtype=object
)

rng_cv = np.random.default_rng(
    2028
)

perm_cv = rng_cv.permutation(
    perturbations_cv
)

condition_folds_cv = np.array_split(
    perm_cv,
    5
)


print(
    "\nPerturbations per fold:",
    [
        len(x)
        for x in condition_folds_cv
    ]
)


# ============================================
# Verify fold assignment
# ============================================

all_test_conditions = np.concatenate(
    condition_folds_cv
)

print(
    "Every perturbation tested exactly once:",
    bool(
        len(all_test_conditions)
        == len(perturbations_cv)
        and
        len(
            np.unique(
                all_test_conditions
            )
        )
        == len(perturbations_cv)
        and
        set(
            all_test_conditions
        )
        == set(
            perturbations_cv
        )
    )
)

print(
    "Non-targeting ever in test folds:",
    bool(
        np.any(
            all_test_conditions
            == "non-targeting"
        )
    )
)


# ============================================
# Phase-only baseline design
#
# Column k:
#   dt if interval starts at phase bin k
#   0 otherwise
#
# Thus:
#
#   y_hat = dt * phase_rate[k]
#
# This is condition-agnostic but allows an
# arbitrary shared phase profile across the
# 9 adjacent intervals.
# ============================================

n_phase_intervals = 9

X_phase_cv = np.zeros(
    (
        len(final_interval_rows),
        n_phase_intervals
    ),
    dtype=np.float64
)

for k in range(
    n_phase_intervals
):

    X_phase_cv[
        row_bin_a_cv == k,
        k
    ] = row_dt_cv[
        row_bin_a_cv == k
    ]


print(
    "\nPhase baseline design shape:",
    X_phase_cv.shape
)

print(
    "Finite:",
    bool(
        np.all(
            np.isfinite(
                X_phase_cv
            )
        )
    )
)

print(
    "Exactly one active phase column per row:",
    bool(
        np.all(
            np.sum(
                X_phase_cv != 0,
                axis=1
            )
            == 1
        )
    )
)


# ============================================
# Test-row counts by fold
# ============================================

fold_summary_records = []

for fold_idx, test_conditions in enumerate(
    condition_folds_cv
):

    test_mask = np.isin(
        row_conditions_cv,
        test_conditions
    )

    train_mask = ~test_mask

    fold_summary_records.append({
        "fold":
            fold_idx + 1,

        "n_test_conditions":
            len(
                test_conditions
            ),

        "n_train_rows":
            int(
                train_mask.sum()
            ),

        "n_test_rows":
            int(
                test_mask.sum()
            ),

        "nt_rows_in_train":
            int(
                np.sum(
                    train_mask
                    &
                    (
                        row_conditions_cv
                        == "non-targeting"
                    )
                )
            ),

        "nt_rows_in_test":
            int(
                np.sum(
                    test_mask
                    &
                    (
                        row_conditions_cv
                        == "non-targeting"
                    )
                )
            ),
    })


cv_fold_summary = pd.DataFrame(
    fold_summary_records
)

print(
    "\nFold summary:"
)

print(
    cv_fold_summary.to_string(
        index=False
    )
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 122: Cell 117. Exact-integral condition-held-out
# ============================================================================

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


# ============================================================================
# SOURCE NOTEBOOK INDEX 123: Cell 118. Verify OOF missing-entry accounting
# ============================================================================

# ============================================
# Cell 118. Verify OOF missing-entry accounting
#
# Cell 117 intentionally predicts ONLY
# held-out perturbation conditions.
#
# Therefore missing entries should be:
#   1. all NT rows, for all 426 genes
#   2. q = g direct-perturbation rows
#      for the 12 overlap response genes
#
# No model refitting.
# ============================================

import numpy as np
import pandas as pd


# ============================================
# NT rows
# ============================================

nt_row_mask = (
    row_conditions_cv
    == "non-targeting"
)

n_nt_rows = int(
    nt_row_mask.sum()
)

print(
    "NT interval rows:",
    n_nt_rows
)

print(
    "Expected NT missing entries:",
    n_nt_rows * len(response_genes)
)


# ============================================
# Direct q = g rows
# ============================================

direct_missing_records = []

expected_direct_missing = 0


for g in overlap_genes:

    gi = np.where(
        response_arr == g
    )[0][0]

    direct_rows = (
        row_conditions_cv == g
    )

    n_direct_rows = int(
        direct_rows.sum()
    )

    expected_direct_missing += (
        n_direct_rows
    )


    actual_missing_for_gene = (
        ~np.isfinite(
            oof_y[:, gi]
        )
    )


    # Missing rows beyond NT
    actual_direct_only = (
        actual_missing_for_gene
        & ~nt_row_mask
    )


    direct_missing_records.append({
        "gene":
            str(g),

        "n_nt_rows":
            n_nt_rows,

        "n_direct_rows":
            n_direct_rows,

        "expected_total_missing":
            n_nt_rows
            + n_direct_rows,

        "actual_total_missing":
            int(
                actual_missing_for_gene.sum()
            ),

        "actual_nonNT_missing":
            int(
                actual_direct_only.sum()
            ),

        "nonNT_missing_matches_qeqg":
            bool(
                np.array_equal(
                    actual_direct_only,
                    direct_rows
                )
            ),
    })


direct_missing_df = pd.DataFrame(
    direct_missing_records
)


print(
    "\nDirect-intervention missing rows:"
)

print(
    direct_missing_df.to_string(
        index=False
    )
)


# ============================================
# Global expected missing count
# ============================================

expected_total_missing = (
    n_nt_rows
    * len(response_genes)
    + expected_direct_missing
)

actual_total_missing = int(
    np.sum(
        ~np.isfinite(
            oof_y
        )
    )
)


print(
    "\nExpected direct-intervention "
    "missing entries:",
    expected_direct_missing
)

print(
    "Expected TOTAL missing entries:",
    expected_total_missing
)

print(
    "Actual TOTAL missing entries:",
    actual_total_missing
)

print(
    "Exact total match:",
    expected_total_missing
    == actual_total_missing
)


# ============================================
# Check non-overlap genes:
# they should miss ONLY NT rows
# ============================================

non_overlap_genes = [
    g
    for g in response_arr
    if g not in set(overlap_genes)
]


non_overlap_ok = True

for g in non_overlap_genes:

    gi = np.where(
        response_arr == g
    )[0][0]

    missing = (
        ~np.isfinite(
            oof_y[:, gi]
        )
    )

    if not np.array_equal(
        missing,
        nt_row_mask
    ):
        non_overlap_ok = False
        break


print(
    "\nAll 414 non-overlap genes "
    "miss ONLY NT rows:",
    non_overlap_ok
)


# ============================================
# Fold-ID interpretation
# ============================================

print(
    "\nRows with fold_id = -1:",
    int(
        np.sum(
            oof_fold_id == -1
        )
    )
)

print(
    "Those rows are exactly NT rows:",
    bool(
        np.array_equal(
            oof_fold_id == -1,
            nt_row_mask
        )
    )
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 124: Cell 119. Primary held-out validation
# ============================================================================

# ============================================
# Cell 119. Primary held-out validation
#
# Exact-integral GRN
#       vs
# strong phase-only baseline
#
# Metric:
#
# incremental R^2 =
#   1 - SSE_GRN / SSE_PHASE
#
# > 0  : GRN improves over phase-only
# = 0  : equal
# < 0  : GRN worse than phase-only
#
# IMPORTANT:
# evaluate non-calibration genes separately.
# ============================================

import numpy as np
import pandas as pd


validation_records = []


for gi, g in enumerate(
    response_arr
):

    valid = (
        np.isfinite(
            oof_y[:, gi]
        )
        &
        np.isfinite(
            oof_pred_grn[:, gi]
        )
        &
        np.isfinite(
            oof_pred_phase[:, gi]
        )
    )


    y = oof_y[
        valid, gi
    ]

    pred_grn = oof_pred_grn[
        valid, gi
    ]

    pred_phase = oof_pred_phase[
        valid, gi
    ]


    sse_grn = float(
        np.sum(
            (
                y - pred_grn
            ) ** 2
        )
    )

    sse_phase = float(
        np.sum(
            (
                y - pred_phase
            ) ** 2
        )
    )


    incremental_r2 = (
        1.0
        - sse_grn / sse_phase
        if sse_phase > 0
        else np.nan
    )


    # Conventional OOF R2 against global
    # test-outcome mean, for context only.
    sst = float(
        np.sum(
            (
                y - np.mean(y)
            ) ** 2
        )
    )


    r2_grn = (
        1.0
        - sse_grn / sst
        if sst > 0
        else np.nan
    )

    r2_phase = (
        1.0
        - sse_phase / sst
        if sst > 0
        else np.nan
    )


    validation_records.append({
        "gene":
            str(g),

        "n_oof":
            int(
                valid.sum()
            ),

        "sse_grn":
            sse_grn,

        "sse_phase":
            sse_phase,

        "incremental_r2_vs_phase":
            float(
                incremental_r2
            ),

        "r2_grn":
            float(
                r2_grn
            ),

        "r2_phase":
            float(
                r2_phase
            ),

        "grn_better_than_phase":
            bool(
                sse_grn < sse_phase
            ),
    })


exact_oof_validation = pd.DataFrame(
    validation_records
)


# ============================================
# Calibration-panel membership
# ============================================

calibration_gene_set = set(
    calibration_genes
)

exact_oof_validation[
    "is_calibration_gene"
] = exact_oof_validation[
    "gene"
].isin(
    calibration_gene_set
)


primary_validation = (
    exact_oof_validation[
        ~exact_oof_validation[
            "is_calibration_gene"
        ]
    ]
    .copy()
)

calibration_validation = (
    exact_oof_validation[
        exact_oof_validation[
            "is_calibration_gene"
        ]
    ]
    .copy()
)


# ============================================
# Summary helper
# ============================================

def print_validation_summary(
    df,
    label
):

    x = df[
        "incremental_r2_vs_phase"
    ].to_numpy(
        dtype=float
    )

    print(
        f"\n{label}"
    )

    print(
        "n genes:",
        len(df)
    )

    print(
        "incremental R2 mean:",
        float(
            np.nanmean(x)
        )
    )

    print(
        "incremental R2 median:",
        float(
            np.nanmedian(x)
        )
    )

    print(
        "fraction > 0:",
        float(
            np.nanmean(
                x > 0
            )
        )
    )

    print(
        "fraction >= 0.05:",
        float(
            np.nanmean(
                x >= 0.05
            )
        )
    )

    print(
        "fraction >= 0.10:",
        float(
            np.nanmean(
                x >= 0.10
            )
        )
    )

    print(
        "\nIncremental R2 distribution:"
    )

    print(
        pd.Series(
            x
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
        "\nMean conventional OOF R2:"
    )

    print(
        "GRN:",
        float(
            df[
                "r2_grn"
            ].mean()
        )
    )

    print(
        "Phase:",
        float(
            df[
                "r2_phase"
            ].mean()
        )
    )


# ============================================
# PRIMARY:
# genes not used for alpha calibration
# ============================================

print_validation_summary(
    primary_validation,
    "PRIMARY NON-CALIBRATION GENES"
)


# ============================================
# Calibration panel
# ============================================

print_validation_summary(
    calibration_validation,
    "CALIBRATION PANEL"
)


# ============================================
# All genes, descriptive only
# ============================================

print_validation_summary(
    exact_oof_validation,
    "ALL GENES"
)


# ============================================
# Best / worst non-calibration genes
# ============================================

print(
    "\nTop 15 non-calibration genes:"
)

print(
    primary_validation
    .sort_values(
        "incremental_r2_vs_phase",
        ascending=False
    )
    .head(15)
    [
        [
            "gene",
            "n_oof",
            "incremental_r2_vs_phase",
            "r2_grn",
            "r2_phase",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nBottom 15 non-calibration genes:"
)

print(
    primary_validation
    .sort_values(
        "incremental_r2_vs_phase",
        ascending=True
    )
    .head(15)
    [
        [
            "gene",
            "n_oof",
            "incremental_r2_vs_phase",
            "r2_grn",
            "r2_phase",
        ]
    ]
    .to_string(
        index=False
    )
)

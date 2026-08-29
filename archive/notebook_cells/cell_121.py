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
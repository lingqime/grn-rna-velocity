# ============================================
# Cell 122. Held-out regulator feature
# extrapolation diagnostic
#
# Question:
# Are test perturbation intervals outside the
# regulator-feature distribution represented
# in the training conditions?
#
# No model fitting.
# ============================================

import numpy as np
import pandas as pd


Xr_all = np.asarray(
    X_final[:, 1:],
    dtype=np.float64
)


fold_shift_records = []


for fold_idx, test_conditions in enumerate(
    condition_folds_cv
):

    test_mask = np.isin(
        row_conditions_cv,
        test_conditions
    )

    train_mask = ~test_mask


    X_train = Xr_all[
        train_mask
    ]

    X_test = Xr_all[
        test_mask
    ]


    # ========================================
    # Training-only standardization
    # ========================================

    train_mean = np.mean(
        X_train,
        axis=0
    )

    train_std = np.std(
        X_train,
        axis=0,
        ddof=0
    )

    assert np.all(
        train_std > 0
    )


    Z_train = (
        X_train
        - train_mean[None, :]
    ) / train_std[None, :]

    Z_test = (
        X_test
        - train_mean[None, :]
    ) / train_std[None, :]


    # ========================================
    # Per-row standardized distance from
    # training centroid
    #
    # RMS z-score across 151 regulators.
    # ========================================

    train_rms_distance = np.sqrt(
        np.mean(
            Z_train ** 2,
            axis=1
        )
    )

    test_rms_distance = np.sqrt(
        np.mean(
            Z_test ** 2,
            axis=1
        )
    )


    # ========================================
    # Coordinate-wise range extrapolation
    #
    # Fraction of regulator coordinates in
    # each test row lying outside the
    # training min/max.
    # ========================================

    train_min = np.min(
        X_train,
        axis=0
    )

    train_max = np.max(
        X_train,
        axis=0
    )


    outside = (
        (X_test < train_min[None, :])
        |
        (X_test > train_max[None, :])
    )

    outside_fraction_per_row = np.mean(
        outside,
        axis=1
    )


    # ========================================
    # Extreme standardized coordinates
    # ========================================

    abs_Z_test = np.abs(
        Z_test
    )

    frac_abs_z_gt_2 = np.mean(
        abs_Z_test > 2,
        axis=1
    )

    frac_abs_z_gt_3 = np.mean(
        abs_Z_test > 3,
        axis=1
    )


    fold_shift_records.append({
        "fold":
            fold_idx + 1,

        "n_train_rows":
            int(
                train_mask.sum()
            ),

        "n_test_rows":
            int(
                test_mask.sum()
            ),

        "train_rms_distance_median":
            float(
                np.median(
                    train_rms_distance
                )
            ),

        "test_rms_distance_median":
            float(
                np.median(
                    test_rms_distance
                )
            ),

        "test_rms_distance_p95":
            float(
                np.quantile(
                    test_rms_distance,
                    0.95
                )
            ),

        "test_outside_range_mean":
            float(
                np.mean(
                    outside_fraction_per_row
                )
            ),

        "test_outside_range_max":
            float(
                np.max(
                    outside_fraction_per_row
                )
            ),

        "test_frac_abs_z_gt_2_mean":
            float(
                np.mean(
                    frac_abs_z_gt_2
                )
            ),

        "test_frac_abs_z_gt_3_mean":
            float(
                np.mean(
                    frac_abs_z_gt_3
                )
            ),
    })


feature_shift_summary = pd.DataFrame(
    fold_shift_records
)


print(
    "Held-out regulator-feature shift:"
)

print(
    feature_shift_summary.to_string(
        index=False
    )
)


# ============================================
# Aggregate summaries
# ============================================

print(
    "\nMedian test/train RMS-distance ratio:"
)

print(
    float(
        np.median(
            feature_shift_summary[
                "test_rms_distance_median"
            ]
            /
            feature_shift_summary[
                "train_rms_distance_median"
            ]
        )
    )
)


print(
    "\nMean coordinate-wise outside-range "
    "fraction:"
)

print(
    float(
        feature_shift_summary[
            "test_outside_range_mean"
        ].mean()
    )
)


print(
    "\nMean fraction |z| > 2:"
)

print(
    float(
        feature_shift_summary[
            "test_frac_abs_z_gt_2_mean"
        ].mean()
    )
)


print(
    "Mean fraction |z| > 3:"
)

print(
    float(
        feature_shift_summary[
            "test_frac_abs_z_gt_3_mean"
        ].mean()
    )
)
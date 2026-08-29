"""
RPE1 diagnostics supporting interpretation of held-out validation.

These diagnostics are explanatory rather than additional primary analyses:
pooled-SSE concentration, feature extrapolation, and shared-versus-condition-
specific regulator variation.

This file was mechanically extracted from archive/Replogle_cloud.ipynb.
Notebook cell numbers are preserved below to make provenance auditable.
"""


# ============================================================================
# SOURCE NOTEBOOK INDEX 125: Cell 120. Diagnose the negative held-out result
# ============================================================================

# ============================================
# Cell 120. Diagnose the negative held-out result
#
# Questions:
# 1. Does the negative mean incremental R2 come
#    mainly from genes where phase baseline SSE
#    is already extremely small?
#
# 2. What is the pooled held-out performance?
#
# 3. How does GRN improvement depend on
#    phase-only predictive strength?
#
# No refitting. No tuning.
# ============================================

import numpy as np
import pandas as pd


df = primary_validation.copy()


# ============================================
# 1. Pooled SSE comparison
# ============================================

pooled_sse_grn = float(
    df["sse_grn"].sum()
)

pooled_sse_phase = float(
    df["sse_phase"].sum()
)

pooled_incremental_r2 = (
    1.0
    - pooled_sse_grn
    / pooled_sse_phase
)


print(
    "PRIMARY NON-CALIBRATION GENES"
)

print(
    "\nPooled SSE GRN:",
    pooled_sse_grn
)

print(
    "Pooled SSE phase:",
    pooled_sse_phase
)

print(
    "Pooled incremental R2 vs phase:",
    pooled_incremental_r2
)


# ============================================
# 2. Relationship with phase-baseline strength
# ============================================

print(
    "\nSpearman correlations:"
)

corr_cols = [
    "incremental_r2_vs_phase",
    "r2_phase",
    "r2_grn",
    "sse_phase",
]

print(
    df[
        corr_cols
    ]
    .corr(
        method="spearman"
    )
)


# ============================================
# 3. Phase-R2 quintile stratification
# ============================================

df["phase_r2_quintile"] = pd.qcut(
    df["r2_phase"],
    q=5,
    duplicates="drop"
)


phase_strata = (
    df
    .groupby(
        "phase_r2_quintile",
        observed=True
    )
    .agg(
        n_genes=(
            "gene",
            "size"
        ),

        phase_r2_mean=(
            "r2_phase",
            "mean"
        ),

        phase_r2_median=(
            "r2_phase",
            "median"
        ),

        grn_r2_mean=(
            "r2_grn",
            "mean"
        ),

        incremental_mean=(
            "incremental_r2_vs_phase",
            "mean"
        ),

        incremental_median=(
            "incremental_r2_vs_phase",
            "median"
        ),

        fraction_grn_better=(
            "grn_better_than_phase",
            "mean"
        ),

        sse_grn_sum=(
            "sse_grn",
            "sum"
        ),

        sse_phase_sum=(
            "sse_phase",
            "sum"
        ),
    )
    .reset_index()
)


phase_strata[
    "pooled_incremental_r2"
] = (
    1.0
    - phase_strata[
        "sse_grn_sum"
    ]
    / phase_strata[
        "sse_phase_sum"
    ]
)


print(
    "\nPerformance stratified by "
    "phase-only R2 quintile:"
)

print(
    phase_strata.to_string(
        index=False
    )
)


# ============================================
# 4. Distribution after excluding only the
#    most phase-predictable quintile
#
# DIAGNOSTIC ONLY.
# This is NOT a new primary result.
# ============================================

phase_cut = df[
    "r2_phase"
].quantile(
    0.80
)

lower_phase_80 = df[
    df[
        "r2_phase"
    ] <= phase_cut
]


print(
    "\nDiagnostic: genes below top 20% "
    "phase predictability"
)

print(
    "n genes:",
    len(
        lower_phase_80
    )
)

print(
    "phase R2 cutoff:",
    float(
        phase_cut
    )
)

print(
    "incremental R2 mean:",
    float(
        lower_phase_80[
            "incremental_r2_vs_phase"
        ].mean()
    )
)

print(
    "incremental R2 median:",
    float(
        lower_phase_80[
            "incremental_r2_vs_phase"
        ].median()
    )
)

print(
    "fraction > 0:",
    float(
        lower_phase_80[
            "grn_better_than_phase"
        ].mean()
    )
)


# ============================================
# 5. Genes with phase baseline already
#    explaining >= 95% of OOF variance
# ============================================

very_phase = df[
    df[
        "r2_phase"
    ] >= 0.95
]


print(
    "\nGenes with phase-only OOF R2 >= 0.95:"
)

print(
    "n genes:",
    len(
        very_phase
    )
)

if len(
    very_phase
) > 0:

    print(
        "incremental R2 mean:",
        float(
            very_phase[
                "incremental_r2_vs_phase"
            ].mean()
        )
    )

    print(
        "incremental R2 median:",
        float(
            very_phase[
                "incremental_r2_vs_phase"
            ].median()
        )
    )

    print(
        "fraction GRN better:",
        float(
            very_phase[
                "grn_better_than_phase"
            ].mean()
        )
    )


# ============================================================================
# SOURCE NOTEBOOK INDEX 126: Cell 121. Diagnose pooled SSE improvement
# ============================================================================

# ============================================
# Cell 121. Diagnose pooled SSE improvement
#
# Question:
# Is pooled +16.4% improvement broad,
# or dominated by a small number of genes
# with very large phase-baseline SSE?
#
# No refitting.
# ============================================

import numpy as np
import pandas as pd


df = primary_validation.copy()


# ============================================
# Per-gene absolute SSE improvement
#
# Positive = GRN better
# Negative = phase baseline better
# ============================================

df[
    "absolute_sse_improvement"
] = (
    df["sse_phase"]
    - df["sse_grn"]
)

df[
    "phase_sse_weight"
] = (
    df["sse_phase"]
    / df["sse_phase"].sum()
)


# ============================================
# Overall accounting
# ============================================

positive_improvement = df[
    df[
        "absolute_sse_improvement"
    ] > 0
]

negative_improvement = df[
    df[
        "absolute_sse_improvement"
    ] < 0
]


total_gain = float(
    positive_improvement[
        "absolute_sse_improvement"
    ].sum()
)

total_loss = float(
    -negative_improvement[
        "absolute_sse_improvement"
    ].sum()
)

net_gain = float(
    df[
        "absolute_sse_improvement"
    ].sum()
)


print(
    "Genes with absolute SSE improvement:",
    len(
        positive_improvement
    )
)

print(
    "Genes with absolute SSE worsening:",
    len(
        negative_improvement
    )
)

print(
    "\nTotal SSE gain from improved genes:",
    total_gain
)

print(
    "Total SSE loss from worsened genes:",
    total_loss
)

print(
    "Net SSE gain:",
    net_gain
)

print(
    "Gain / loss ratio:",
    total_gain / total_loss
)


# ============================================
# How concentrated is phase SSE?
# ============================================

df_by_weight = df.sort_values(
    "sse_phase",
    ascending=False
).copy()

df_by_weight[
    "cumulative_phase_sse_fraction"
] = (
    df_by_weight[
        "sse_phase"
    ].cumsum()
    / df_by_weight[
        "sse_phase"
    ].sum()
)


for n in [
    5,
    10,
    20,
    50,
    100,
]:

    top = df_by_weight.head(
        n
    )

    print(
        f"\nTop {n} genes by phase SSE:"
    )

    print(
        "fraction of total phase SSE:",
        float(
            top[
                "sse_phase"
            ].sum()
            / df[
                "sse_phase"
            ].sum()
        )
    )

    print(
        "fraction of net SSE gain:",
        float(
            top[
                "absolute_sse_improvement"
            ].sum()
            / net_gain
        )
    )


# ============================================
# Top contributors to pooled improvement
# ============================================

print(
    "\nTop 20 contributors to pooled "
    "GRN improvement:"
)

print(
    df
    .sort_values(
        "absolute_sse_improvement",
        ascending=False
    )
    .head(20)
    [
        [
            "gene",
            "sse_phase",
            "sse_grn",
            "absolute_sse_improvement",
            "incremental_r2_vs_phase",
            "r2_phase",
            "r2_grn",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================
# Top contributors against GRN
# ============================================

print(
    "\nTop 20 contributors favoring "
    "phase-only:"
)

print(
    df
    .sort_values(
        "absolute_sse_improvement",
        ascending=True
    )
    .head(20)
    [
        [
            "gene",
            "sse_phase",
            "sse_grn",
            "absolute_sse_improvement",
            "incremental_r2_vs_phase",
            "r2_phase",
            "r2_grn",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================
# Leave-top-K-out pooled sensitivity
#
# Remove genes with largest phase SSE and
# recompute pooled incremental R2.
#
# Diagnostic only.
# ============================================

print(
    "\nLeave-high-SSE-genes-out "
    "pooled incremental R2:"
)


for k in [
    0,
    1,
    5,
    10,
    20,
    50,
    100,
]:

    remaining = (
        df_by_weight.iloc[
            k:
        ]
    )

    pooled = (
        1.0
        - remaining[
            "sse_grn"
        ].sum()
        / remaining[
            "sse_phase"
        ].sum()
    )

    print(
        f"remove top {k:3d}: "
        f"{pooled:.6f}"
    )


# ============================================================================
# SOURCE NOTEBOOK INDEX 127: Cell 122. Held-out regulator feature
# ============================================================================

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


# ============================================================================
# SOURCE NOTEBOOK INDEX 128: Cell 123. Decompose regulator trajectories
# ============================================================================

# ============================================
# Cell 123. Decompose regulator trajectories
# into:
#
#   shared NT phase component
#   +
#   condition-specific deviation
#
# Goal:
# determine what information the current
# GRN predictor is actually using.
#
# No fitting.
# ============================================

import numpy as np
import pandas as pd


# ============================================
# Locate NT condition
# ============================================

condition_arr = np.asarray(
    final_conditions_vc,
    dtype=object
)

nt_ci = np.where(
    condition_arr
    == "non-targeting"
)[0][0]


# ============================================
# NT regulator phase trajectory
#
# shape:
#   10 bins x 151 regulators
# ============================================

S_reg_nt = np.asarray(
    S_regulator_hat[
        nt_ci
    ],
    dtype=np.float64
)


print(
    "NT regulator trajectory shape:",
    S_reg_nt.shape
)

print(
    "Finite:",
    bool(
        np.all(
            np.isfinite(
                S_reg_nt
            )
        )
    )
)


# ============================================
# Build interval-level shared-phase features
#
# Use exactly the same trapezoidal integral
# and condition-specific dt as X_final.
#
# X_shared:
#   what X would be if every condition had
#   the NT regulator phase trajectory.
#
# X_deviation:
#   actual X - shared component.
# ============================================

n_intervals = len(
    final_interval_rows
)

n_regs = len(
    regulator_genes_vc
)


X_shared = np.zeros(
    (
        n_intervals,
        n_regs
    ),
    dtype=np.float64
)


for i, row in enumerate(
    final_interval_rows
):

    a = int(
        row["bin_a"]
    )

    b = int(
        row["bin_b"]
    )

    dt = float(
        X_final[
            i, 0
        ]
    )


    X_shared[
        i
    ] = (
        0.5
        * dt
        * (
            S_reg_nt[a]
            + S_reg_nt[b]
        )
    )


X_actual = np.asarray(
    X_final[:, 1:],
    dtype=np.float64
)

X_deviation = (
    X_actual
    - X_shared
)


# ============================================
# Sanity checks
# ============================================

print(
    "\nShapes:"
)

print(
    "actual:",
    X_actual.shape
)

print(
    "shared:",
    X_shared.shape
)

print(
    "deviation:",
    X_deviation.shape
)


print(
    "\nExact decomposition:",
    bool(
        np.allclose(
            X_actual,
            X_shared
            + X_deviation,
            rtol=1e-12,
            atol=1e-12
        )
    )
)


# ============================================
# Relative magnitude
# ============================================

actual_rms = np.sqrt(
    np.mean(
        X_actual ** 2
    )
)

shared_rms = np.sqrt(
    np.mean(
        X_shared ** 2
    )
)

deviation_rms = np.sqrt(
    np.mean(
        X_deviation ** 2
    )
)


print(
    "\nGlobal RMS:"
)

print(
    "actual:",
    float(
        actual_rms
    )
)

print(
    "shared phase:",
    float(
        shared_rms
    )
)

print(
    "condition deviation:",
    float(
        deviation_rms
    )
)

print(
    "deviation / actual:",
    float(
        deviation_rms
        / actual_rms
    )
)


# ============================================
# Per-regulator decomposition
# ============================================

per_reg_actual_rms = np.sqrt(
    np.mean(
        X_actual ** 2,
        axis=0
    )
)

per_reg_deviation_rms = np.sqrt(
    np.mean(
        X_deviation ** 2,
        axis=0
    )
)

per_reg_deviation_fraction = (
    per_reg_deviation_rms
    / per_reg_actual_rms
)


regulator_decomposition = pd.DataFrame({
    "regulator_gene":
        np.asarray(
            regulator_genes_vc,
            dtype=str
        ),

    "actual_rms":
        per_reg_actual_rms,

    "deviation_rms":
        per_reg_deviation_rms,

    "deviation_fraction":
        per_reg_deviation_fraction,
})


print(
    "\nPer-regulator deviation fraction:"
)

print(
    regulator_decomposition[
        "deviation_fraction"
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


print(
    "\nLargest condition-specific "
    "deviation fractions:"
)

print(
    regulator_decomposition
    .sort_values(
        "deviation_fraction",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================
# Condition-level deviation magnitude
# ============================================

condition_deviation_records = []


for q in final_conditions_vc:

    mask = (
        row_conditions_cv == q
    )

    if not np.any(
        mask
    ):
        continue


    q_actual_rms = np.sqrt(
        np.mean(
            X_actual[
                mask
            ] ** 2
        )
    )

    q_dev_rms = np.sqrt(
        np.mean(
            X_deviation[
                mask
            ] ** 2
        )
    )


    condition_deviation_records.append({
        "condition":
            str(q),

        "n_intervals":
            int(
                mask.sum()
            ),

        "actual_rms":
            float(
                q_actual_rms
            ),

        "deviation_rms":
            float(
                q_dev_rms
            ),

        "deviation_fraction":
            float(
                q_dev_rms
                / q_actual_rms
            ),
    })


condition_decomposition = pd.DataFrame(
    condition_deviation_records
)


print(
    "\nCondition-level deviation fraction:"
)

print(
    condition_decomposition[
        "deviation_fraction"
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

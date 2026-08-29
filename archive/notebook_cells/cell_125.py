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
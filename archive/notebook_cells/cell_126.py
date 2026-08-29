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
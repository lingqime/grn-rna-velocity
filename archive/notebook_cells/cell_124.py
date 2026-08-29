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
# ============================================
# Cell 80. Inspect response-rate scale
# across all 426 genes
# ============================================

import numpy as np
import pandas as pd

dt_all = np.asarray(
    X_final[:, 0],
    dtype=np.float64
)

R_all = (
    Y_final
    / dt_all[:, None]
)

response_scale_rows = []

for gi, g in enumerate(
    response_genes
):

    mask = valid_row_masks[g]

    r = R_all[
        mask, gi
    ]

    response_scale_rows.append({
        "gene": g,
        "n_rows": len(r),
        "mean": np.mean(r),
        "std": np.std(
            r,
            ddof=0
        ),
        "median": np.median(r),
        "rms": np.sqrt(
            np.mean(
                r ** 2
            )
        ),
        "q05": np.quantile(
            r,
            0.05
        ),
        "q95": np.quantile(
            r,
            0.95
        ),
    })


response_scale_df = pd.DataFrame(
    response_scale_rows
)

print(
    "Response-rate scale summary:"
)

print(
    response_scale_df[
        ["std", "rms"]
    ].describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
)


print(
    "\nLargest response-rate SD:"
)

print(
    response_scale_df
    .sort_values(
        "std",
        ascending=False
    )
    .head(15)
    .to_string(
        index=False
    )
)


print(
    "\nSmallest response-rate SD:"
)

print(
    response_scale_df
    .sort_values(
        "std"
    )
    .head(15)
    .to_string(
        index=False
    )
)


tacc3_scale = response_scale_df[
    response_scale_df[
        "gene"
    ] == "TACC3"
]

print(
    "\nTACC3 scale:"
)

print(
    tacc3_scale.to_string(
        index=False
    )
)


print(
    "\nMax / min response SD ratio:",
    response_scale_df[
        "std"
    ].max()
    /
    response_scale_df[
        "std"
    ].min()
)
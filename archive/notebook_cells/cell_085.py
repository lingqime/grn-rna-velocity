# ============================================
# Cell 82. Build a representative response-gene
# panel for global-alpha calibration
# ============================================

import numpy as np
import pandas as pd

n_panel = 30

scale_sorted = (
    response_scale_df
    .sort_values("std")
    .reset_index(drop=True)
)

# Evenly spaced ranks across the full
# response-SD distribution
rank_idx = np.linspace(
    0,
    len(scale_sorted) - 1,
    n_panel,
    dtype=int
)

panel_genes = list(
    scale_sorted.loc[
        rank_idx,
        "gene"
    ]
)

# Force TACC3 into the calibration panel
# without introducing a duplicate
if "TACC3" not in panel_genes:
    panel_genes.append(
        "TACC3"
    )

panel_genes = list(
    dict.fromkeys(
        panel_genes
    )
)

panel_df = (
    response_scale_df
    .set_index("gene")
    .loc[panel_genes]
    .reset_index()
    .sort_values("std")
    .reset_index(drop=True)
)

print(
    "Calibration panel size:",
    len(panel_df)
)

print(
    "\nCalibration genes:"
)

print(
    panel_df[
        [
            "gene",
            "std",
            "rms",
            "n_rows",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nPanel response-SD range:"
)

print(
    "min   =",
    panel_df["std"].min()
)

print(
    "median=",
    panel_df["std"].median()
)

print(
    "max   =",
    panel_df["std"].max()
)

print(
    "\nFull response-SD range:"
)

print(
    "min   =",
    response_scale_df[
        "std"
    ].min()
)

print(
    "median=",
    response_scale_df[
        "std"
    ].median()
)

print(
    "max   =",
    response_scale_df[
        "std"
    ].max()
)
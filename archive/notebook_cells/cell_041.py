# ============================================
# Cell 41. Attach official VeloCycle phase
# and re-check 10-bin coverage
# ============================================

import numpy as np
import pandas as pd

# Official compact extract
vc_cell_ids = vc["cell_ids"].astype(str)
vc_phi = vc["phi"].astype(float)

# Sanity: official extract should cover all cells
print("Official phase cells:", len(vc_cell_ids))
print("Full adata cells:", adata.n_obs)

print(
    "Same cell set:",
    set(vc_cell_ids) == set(adata.obs_names.astype(str))
)

# Map cell ID -> phi
phi_series = pd.Series(
    vc_phi,
    index=vc_cell_ids,
    name="phi_vc"
)

# Align to full adata order
phi_aligned = phi_series.loc[
    adata.obs_names.astype(str)
].to_numpy()

# normalized circular phase x in [0, 1)
cycle_time_vc = (
    phi_aligned / (2.0 * np.pi)
) % 1.0

print(
    "\ncycle_time_vc range:",
    cycle_time_vc.min(),
    cycle_time_vc.max()
)

# Final 154-condition mask
cond = adata.obs["gene"].astype(str).to_numpy()

final_mask = np.isin(
    cond,
    final_conditions_qc
)

# 10 phase bins
bin_edges = np.linspace(0.0, 1.0, 11)

bin_id = np.digitize(
    cycle_time_vc,
    bin_edges[1:-1],
    right=False
)

# Count cells per condition x bin
coverage_rows = []

for q in final_conditions_qc:
    q_mask = final_mask & (cond == q)

    counts = np.bincount(
        bin_id[q_mask],
        minlength=10
    )

    coverage_rows.append(
        {
            "condition": q,
            "n_cells": int(q_mask.sum()),
            "occupied_bins": int((counts > 0).sum()),
            "usable_bins_ge5": int((counts >= 5).sum()),
            "min_bin_count": int(counts.min()),
            "max_bin_count": int(counts.max()),
        }
    )

coverage_vc = pd.DataFrame(
    coverage_rows
).set_index("condition")

print("\nOfficial-phase coverage summary:")
print(
    coverage_vc[
        ["occupied_bins", "usable_bins_ge5"]
    ].describe()
)

print("\nUsable-bin count distribution:")
print(
    coverage_vc["usable_bins_ge5"]
    .value_counts()
    .sort_index()
)

print("\nConditions with < 7 usable bins:")
print(
    coverage_vc.loc[
        coverage_vc["usable_bins_ge5"] < 7,
        ["n_cells", "usable_bins_ge5"]
    ]
    .sort_values("usable_bins_ge5")
)
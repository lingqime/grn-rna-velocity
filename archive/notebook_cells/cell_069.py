# ============================================
# Cell 66. Estimate perturbation-specific
# shifts relative to NT within phase bins
# ============================================

import numpy as np
import pandas as pd

# Raw spliced trajectories already constructed:
# S_response_traj_raw:
#   condition x bin x 426 genes
#
# N_traj_vc:
#   condition x bin cell counts

nt_idx = final_conditions_vc.index(
    "non-targeting"
)

S_nt = np.asarray(
    S_response_traj_raw[nt_idx],
    dtype=np.float64
)  # 10 x 426

# Small pseudocount on raw-count scale
pc = 0.1

# condition x bin x gene
logFC_bin = (
    np.log(
        S_response_traj_raw + pc
    )
    - np.log(
        S_nt[None, :, :] + pc
    )
)

# ------------------------------------------------
# Collapse phase bins robustly to one
# condition-specific gene shift.
#
# Only bins usable in that condition AND NT.
# ------------------------------------------------

condition_logFC_nt = np.full(
    (
        len(final_conditions_vc),
        len(vc_gene_names)
    ),
    np.nan,
    dtype=np.float64
)

for qi, q in enumerate(
    final_conditions_vc
):

    usable = (
        (N_traj_vc[qi] >= 5)
        & (N_traj_vc[nt_idx] >= 5)
    )

    condition_logFC_nt[
        qi, :
    ] = np.nanmedian(
        logFC_bin[
            qi,
            usable,
            :
        ],
        axis=0
    )


print(
    "condition_logFC_nt shape:",
    condition_logFC_nt.shape
)

print(
    "Any NaN:",
    np.isnan(
        condition_logFC_nt
    ).any()
)

print(
    "\nNT shift:"
)

print(
    "max abs =",
    np.nanmax(
        np.abs(
            condition_logFC_nt[
                nt_idx
            ]
        )
    )
)

print(
    "\nAll perturbation-gene shifts:"
)

pert_values = np.delete(
    condition_logFC_nt,
    nt_idx,
    axis=0
).ravel()

print(
    pd.Series(
        pert_values
    ).describe(
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


# ------------------------------------------------
# Direct-target sanity check
# ------------------------------------------------

response_lookup = {
    g: j
    for j, g in enumerate(
        vc_gene_names
    )
}

direct_rows = []

for qi, q in enumerate(
    final_conditions_vc
):

    if q == "non-targeting":
        continue

    if q not in response_lookup:
        continue

    gj = response_lookup[q]

    direct_rows.append({
        "condition": q,
        "target_logFC_vs_NT":
            condition_logFC_nt[
                qi, gj
            ],
    })

direct_logFC_df = pd.DataFrame(
    direct_rows
).sort_values(
    "target_logFC_vs_NT"
)

print(
    "\nDirect-target log fold changes vs NT:"
)

print(
    direct_logFC_df.to_string(
        index=False
    )
)

print(
    "\nDirect targets < 0:",
    (
        direct_logFC_df[
            "target_logFC_vs_NT"
        ] < 0
    ).sum(),
    "/",
    len(direct_logFC_df)
)
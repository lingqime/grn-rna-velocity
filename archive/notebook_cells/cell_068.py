# ============================================
# Cell 65. Estimate condition-specific
# expression shifts relative to VeloCycle
# latent phase + library-size baseline
# ============================================

import numpy as np
import pandas as pd

# ------------------------------------------------
# Work only on the already frozen final cells
# ------------------------------------------------

idx = np.asarray(
    final_cell_idx,
    dtype=int
)

phi_cells = np.asarray(
    cycle_time_vc[idx],
    dtype=np.float64
) * 2.0 * np.pi

cf_cells = np.asarray(
    count_factor_full[idx],
    dtype=np.float64
)

condition_cells = np.asarray(
    adata.obs["gene"].iloc[idx],
    dtype=object
)

# ------------------------------------------------
# Exact one-harmonic VeloCycle log-S template
#
# log S_ig =
#   nu0_g
# + nu1_sin_g sin(phi_i)
# + nu1_cos_g cos(phi_i)
# + count_factor_i
# ------------------------------------------------

nu0 = coef[0, :]
nu_sin = coef[1, :]
nu_cos = coef[2, :]

sin_phi = np.sin(phi_cells)
cos_phi = np.cos(phi_cells)

# ------------------------------------------------
# Observed raw spliced counts for the 426 genes
#
# Use log(S + 1), matching VeloCycle preprocessing.
# Process condition-by-condition so we never
# densify the full 57k x 426 matrix at once.
# ------------------------------------------------

condition_log_shift = np.full(
    (
        len(final_conditions_vc),
        len(vc_gene_names)
    ),
    np.nan,
    dtype=np.float64
)

condition_cell_counts = np.zeros(
    len(final_conditions_vc),
    dtype=int
)

for qi, q in enumerate(final_conditions_vc):

    local = np.flatnonzero(
        condition_cells == q
    )

    condition_cell_counts[qi] = len(local)

    # This is only condition_cells x 426,
    # not the full transcriptome.
    obs = (
        U_response_raw[local, :] * 0
    )

    # Retrieve spliced response counts
    S_obs = (
        S_response_raw[local, :]
        .toarray()
        .astype(np.float64)
    )

    logS_obs = np.log(
        S_obs + 1.0
    )

    # VeloCycle fitted log-spliced baseline
    logS_base = (
        nu0[None, :]
        + sin_phi[local, None]
          * nu_sin[None, :]
        + cos_phi[local, None]
          * nu_cos[None, :]
        + cf_cells[local, None]
    )

    residual = (
        logS_obs
        - logS_base
    )

    # Robust condition-level shift
    condition_log_shift[
        qi, :
    ] = np.median(
        residual,
        axis=0
    )


print(
    "condition_log_shift shape:",
    condition_log_shift.shape
)

print(
    "Any NaN:",
    np.isnan(
        condition_log_shift
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        condition_log_shift
    ).any()
)

print(
    "\nCells per condition:"
)

print(
    pd.Series(
        condition_cell_counts
    ).describe()
)

print(
    "\nCondition log-shift summary:"
)

print(
    pd.Series(
        condition_log_shift.ravel()
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
# Sanity check:
# direct CRISPRi target genes should tend toward
# negative shifts when the target is among the
# 426 response genes.
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
        "target_log_shift":
            condition_log_shift[
                qi, gj
            ],
    })

direct_shift_df = pd.DataFrame(
    direct_rows
)

print(
    "\nDirect-target shifts "
    "(targets also among 426 response genes):"
)

print(
    direct_shift_df
    .sort_values(
        "target_log_shift"
    )
    .to_string(
        index=False
    )
)
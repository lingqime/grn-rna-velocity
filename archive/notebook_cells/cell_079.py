# ============================================
# Cell 76. Build gene-specific valid-row masks
# for sparse reconstruction
# ============================================

import numpy as np
import pandas as pd

response_genes = np.asarray(
    vc_gene_names,
    dtype=object
)

regulator_set = set(
    regulator_genes_vc
)

row_conditions = np.asarray([
    row["condition"]
    for row in final_interval_rows
], dtype=object)

valid_row_masks = {}

excluded_summary = []

for g in response_genes:

    # Default:
    # all intervals are usable
    mask = np.ones(
        len(final_interval_rows),
        dtype=bool
    )

    # If response gene g is itself one of
    # the directly perturbed targets,
    # remove condition q = g because h_g(q)
    # is unknown.
    if g in regulator_set:

        direct_mask = (
            row_conditions == g
        )

        mask[
            direct_mask
        ] = False

        n_excluded = int(
            direct_mask.sum()
        )

    else:

        n_excluded = 0

    valid_row_masks[g] = mask

    excluded_summary.append({
        "gene": g,
        "is_perturbed_target":
            g in regulator_set,
        "n_total_rows":
            len(mask),
        "n_excluded_direct":
            n_excluded,
        "n_valid_rows":
            int(mask.sum()),
    })


valid_row_summary = pd.DataFrame(
    excluded_summary
)

print(
    "Response genes:",
    len(response_genes)
)

print(
    "Response genes also directly perturbed:",
    valid_row_summary[
        "is_perturbed_target"
    ].sum()
)

print(
    "\nValid-row count summary:"
)

print(
    valid_row_summary[
        "n_valid_rows"
    ].describe()
)

print(
    "\nDirect-intervention exclusions:"
)

print(
    valid_row_summary[
        valid_row_summary[
            "n_excluded_direct"
        ] > 0
    ][[
        "gene",
        "n_excluded_direct",
        "n_valid_rows",
    ]]
    .sort_values(
        "gene"
    )
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Sanity checks
# --------------------------------------------

assert len(
    valid_row_masks
) == 426

assert all(
    len(mask)
    == X_final.shape[0]
    for mask
    in valid_row_masks.values()
)

print(
    "\nMask bookkeeping checks passed."
)
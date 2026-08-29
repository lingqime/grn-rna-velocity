# ============================================
# Cell 67. Build perturbation-specific
# smoothed spliced response trajectories
# ============================================

import numpy as np

# condition_logFC_nt:
#   condition x 426 genes
#
# S_latent_base.T:
#   10 bins x 426 genes

condition_scale = np.exp(
    condition_logFC_nt
)

S_response_hat = (
    S_latent_base.T[None, :, :]
    * condition_scale[:, None, :]
)

print(
    "S_response_hat shape:",
    S_response_hat.shape
)

print(
    "Any NaN:",
    np.isnan(
        S_response_hat
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        S_response_hat
    ).any()
)

print(
    "\nS_response_hat range:"
)

print(
    "min   =",
    np.min(
        S_response_hat
    )
)

print(
    "median=",
    np.median(
        S_response_hat
    )
)

print(
    "max   =",
    np.max(
        S_response_hat
    )
)

# --------------------------------------------
# Sanity check:
# direct target trajectory should be strongly
# suppressed relative to NT
# --------------------------------------------

response_lookup = {
    g: j
    for j, g in enumerate(
        vc_gene_names
    )
}

condition_lookup = {
    q: i
    for i, q in enumerate(
        final_conditions_vc
    )
}

direct_ratios = []

for q in final_perturbations_vc:

    if q not in response_lookup:
        continue

    qi = condition_lookup[q]
    gi = response_lookup[q]

    ratio = (
        np.mean(
            S_response_hat[
                qi, :, gi
            ]
        )
        /
        np.mean(
            S_response_hat[
                nt_idx, :, gi
            ]
        )
    )

    direct_ratios.append(
        (q, ratio)
    )

print(
    "\nDirect-target mean trajectory ratios "
    "(perturbation / NT):"
)

for q, ratio in direct_ratios:
    print(
        f"{q:10s}  {ratio:.4f}"
    )
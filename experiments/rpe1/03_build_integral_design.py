"""
RPE1 stage 3: exact integral design/response and gene-specific row masks.

This stage builds X_final and Y_final and excludes direct q=g equations
gene-by-gene when the response gene is also a perturbed regulator.

This file was mechanically extracted from archive/Replogle_cloud.ipynb.
Notebook cell numbers are preserved below to make provenance auditable.
"""


# ============================================================================
# SOURCE NOTEBOOK INDEX 74: Cell 71. Build FINAL integral design matrix X
# ============================================================================

# ============================================
# Cell 71. Build FINAL integral design matrix X
# from smoothed regulator trajectories
# ============================================

import numpy as np
import pandas as pd

X_rows = []
final_interval_rows = []

for qi, q in enumerate(
    final_conditions_vc
):

    omega_q = float(
        speed_lookup[q]
    )

    rho_q = (
        omega_q
        / omega_nt
    )

    dt_q = (
        0.1
        / rho_q
    )

    for b in range(
        n_bins - 1
    ):

        # Require both adjacent bins
        # to contain >= 5 cells
        if (
            N_regulator_traj[
                qi, b
            ] < 5
            or
            N_regulator_traj[
                qi, b + 1
            ] < 5
        ):
            continue

        s_a = (
            S_regulator_hat[
                qi, b, :
            ].copy()
        )

        s_b = (
            S_regulator_hat[
                qi, b + 1, :
            ].copy()
        )

        # Explicit M_q:
        # zero the directly perturbed
        # regulator coordinate
        if q != "non-targeting":

            gj = reg_lookup[q]

            s_a[gj] = 0.0
            s_b[gj] = 0.0

        # Trapezoidal integral
        integral_s = (
            0.5
            * dt_q
            * (
                s_a
                + s_b
            )
        )

        # First column = intercept integral
        # integral 1 dt = Delta t
        x = np.concatenate([
            [dt_q],
            integral_s
        ])

        X_rows.append(x)

        final_interval_rows.append({
            "condition": q,
            "condition_idx": qi,
            "bin_a": b,
            "bin_b": b + 1,
            "rho": rho_q,
            "delta_t": dt_q,
            "n_a": int(
                N_regulator_traj[
                    qi, b
                ]
            ),
            "n_b": int(
                N_regulator_traj[
                    qi, b + 1
                ]
            ),
        })


X_final = np.vstack(
    X_rows
)

print(
    "X_final shape:",
    X_final.shape
)

print(
    "Number of intervals:",
    len(
        final_interval_rows
    )
)

print(
    "\nAny NaN:",
    np.isnan(
        X_final
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        X_final
    ).any()
)

print(
    "\nIntercept equals delta_t:",
    np.allclose(
        X_final[:, 0],
        np.asarray([
            r["delta_t"]
            for r in final_interval_rows
        ])
    )
)


# --------------------------------------------
# Verify M_q target coordinate = 0
# --------------------------------------------

mq_violations = 0

for i, row in enumerate(
    final_interval_rows
):

    q = row["condition"]

    if q == "non-targeting":
        continue

    gj = reg_lookup[q]

    if X_final[
        i, 1 + gj
    ] != 0.0:
        mq_violations += 1

print(
    "M_q design violations:",
    mq_violations
)


# --------------------------------------------
# Interval counts by condition
# --------------------------------------------

interval_counts = (
    pd.Series([
        r["condition"]
        for r in final_interval_rows
    ])
    .value_counts()
)

print(
    "\nIntervals per condition:"
)

print(
    interval_counts.describe()
)


# --------------------------------------------
# Numerical scale of regulator integrals
# --------------------------------------------

print(
    "\nRegulator integral summary:"
)

print(
    pd.Series(
        X_final[:, 1:].ravel()
    ).describe(
        percentiles=[
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
        ]
    )
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 75: Cell 72. Build FINAL integral response Y
# ============================================================================

# ============================================
# Cell 72. Build FINAL integral response Y
# from perturbation-specific latent trajectories
# ============================================

import numpy as np
import pandas as pd

# Convert VeloCycle beta to our normalized
# time coordinate:
#
# t = omega_NT * tau / (2*pi)
#
# therefore:
# beta_tilde = (2*pi / omega_NT) * beta

time_scale = (
    2.0
    * np.pi
    / omega_nt
)

beta_tilde = (
    time_scale
    * beta
)

Y_rows = []
delta_U_rows = []
beta_integral_rows = []

for row in final_interval_rows:

    qi = row[
        "condition_idx"
    ]

    a = row[
        "bin_a"
    ]

    b = row[
        "bin_b"
    ]

    dt = row[
        "delta_t"
    ]

    u_a = U_response_hat[
        qi, a, :
    ]

    u_b = U_response_hat[
        qi, b, :
    ]

    delta_u = (
        u_b
        - u_a
    )

    integral_u = (
        0.5
        * dt
        * (
            u_a
            + u_b
        )
    )

    beta_integral = (
        beta_tilde
        * integral_u
    )

    # h_g(q) = 0 here.
    # Direct-target rows will be excluded
    # gene-by-gene during reconstruction.
    y = (
        delta_u
        + beta_integral
    )

    Y_rows.append(y)
    delta_U_rows.append(
        delta_u
    )
    beta_integral_rows.append(
        beta_integral
    )


Y_final = np.vstack(
    Y_rows
)

delta_U_final = np.vstack(
    delta_U_rows
)

beta_integral_final = np.vstack(
    beta_integral_rows
)


print(
    "Y_final shape:",
    Y_final.shape
)

print(
    "Expected shape:",
    (
        X_final.shape[0],
        len(vc_gene_names)
    )
)

print(
    "\nAny NaN:",
    np.isnan(
        Y_final
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        Y_final
    ).any()
)


print(
    "\nY_final summary:"
)

print(
    pd.Series(
        Y_final.ravel()
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


print(
    "\ndelta_U summary:"
)

print(
    pd.Series(
        delta_U_final.ravel()
    ).describe(
        percentiles=[
            0.01,
            0.50,
            0.99,
        ]
    )
)


print(
    "\nbeta-integral summary:"
)

print(
    pd.Series(
        beta_integral_final.ravel()
    ).describe(
        percentiles=[
            0.01,
            0.50,
            0.99,
        ]
    )
)


# --------------------------------------------
# Exact row-level formula check
# --------------------------------------------

check = (
    delta_U_final
    + beta_integral_final
)

print(
    "\nFormula exact:",
    np.allclose(
        Y_final,
        check
    )
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 79: Cell 76. Build gene-specific valid-row masks
# ============================================================================

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

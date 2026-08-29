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
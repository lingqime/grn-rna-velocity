# ============================================
# Cell 68. Build perturbation-specific
# smoothed unspliced response trajectories
# ============================================

import numpy as np

U_response_hat = np.empty_like(
    S_response_hat
)

for qi, q in enumerate(
    final_conditions_vc
):

    omega = float(
        speed_lookup[q]
    )

    inside = (
        omega
        * dlogS_dphi
        + gamma[:, None]
    )

    # gene x bin
    kinetic_ratio = (
        np.maximum(
            inside,
            0.0
        )
        + 1e-5
    ) / beta[:, None]

    # S_response_hat is bin x gene,
    # so transpose kinetic_ratio
    U_response_hat[
        qi, :, :
    ] = (
        S_response_hat[
            qi, :, :
        ]
        * kinetic_ratio.T
    )


print(
    "U_response_hat shape:",
    U_response_hat.shape
)

print(
    "Any NaN:",
    np.isnan(
        U_response_hat
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        U_response_hat
    ).any()
)

print(
    "\nU_response_hat range:"
)

print(
    "min   =",
    np.min(
        U_response_hat
    )
)

print(
    "median=",
    np.median(
        U_response_hat
    )
)

print(
    "max   =",
    np.max(
        U_response_hat
    )
)


# --------------------------------------------
# Verify that the perturbation-specific
# trajectories still satisfy the VeloCycle
# kinetic identity analytically
# --------------------------------------------

rel_errors = []

for qi, q in enumerate(
    final_conditions_vc
):

    omega = float(
        speed_lookup[q]
    )

    # ds/dphi = s * d(log s)/dphi
    lhs = (
        omega
        * S_response_hat[qi].T
        * dlogS_dphi
    )

    rhs = (
        beta[:, None]
        * U_response_hat[qi].T
        - gamma[:, None]
        * S_response_hat[qi].T
    )

    inside = (
        omega
        * dlogS_dphi
        + gamma[:, None]
    )

    active = inside > 0

    denom = np.maximum(
        np.abs(lhs),
        1e-12
    )

    rel_errors.append(
        np.median(
            np.abs(
                rhs[active]
                - lhs[active]
            )
            / denom[active]
        )
    )


rel_errors = np.asarray(
    rel_errors
)

print(
    "\nLatent kinetic identity "
    "median relative error:"
)

print(
    "median =",
    np.median(rel_errors)
)

print(
    "max    =",
    np.max(rel_errors)
)
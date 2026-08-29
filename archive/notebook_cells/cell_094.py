# ============================================
# Cell 91. Residualize shared phase structure
# from response rates and regulator features
# ============================================

import numpy as np

# --------------------------------------------
# Rate-form system
# --------------------------------------------

dt_all = np.asarray(
    X_final[:, 0],
    dtype=np.float64
)

Z_rate = np.asarray(
    X_final[:, 1:],
    dtype=np.float64
) / dt_all[:, None]

R_rate = np.asarray(
    Y_final,
    dtype=np.float64
) / dt_all[:, None]


# --------------------------------------------
# Interval midpoint phase
# --------------------------------------------

phase_mid_all = np.asarray([
    0.5 * (
        bin_centers[
            row["bin_a"]
        ]
        +
        bin_centers[
            row["bin_b"]
        ]
    )
    for row in final_interval_rows
], dtype=np.float64)

theta_all = (
    2.0
    * np.pi
    * phase_mid_all
)


# --------------------------------------------
# Shared phase basis:
# intercept + first two Fourier harmonics
# --------------------------------------------

P_all = np.column_stack([
    np.ones(
        len(theta_all)
    ),
    np.sin(
        theta_all
    ),
    np.cos(
        theta_all
    ),
    np.sin(
        2.0 * theta_all
    ),
    np.cos(
        2.0 * theta_all
    ),
])


# --------------------------------------------
# Residualize regulator features
#
# Z_res = Z - projection onto phase basis
# --------------------------------------------

coef_phase_Z, _, _, _ = np.linalg.lstsq(
    P_all,
    Z_rate,
    rcond=None
)

Z_res = (
    Z_rate
    - P_all @ coef_phase_Z
)


# --------------------------------------------
# Residualize response rates
# independently for every response gene
# --------------------------------------------

coef_phase_R, _, _, _ = np.linalg.lstsq(
    P_all,
    R_rate,
    rcond=None
)

R_res = (
    R_rate
    - P_all @ coef_phase_R
)


print(
    "Z_res shape:",
    Z_res.shape
)

print(
    "R_res shape:",
    R_res.shape
)

print(
    "NaN/inf in Z_res:",
    (
        ~np.isfinite(
            Z_res
        )
    ).any()
)

print(
    "NaN/inf in R_res:",
    (
        ~np.isfinite(
            R_res
        )
    ).any()
)


# --------------------------------------------
# Orthogonality checks
# --------------------------------------------

phase_corr_Z = np.max(
    np.abs(
        P_all.T @ Z_res
    )
)

phase_corr_R = np.max(
    np.abs(
        P_all.T @ R_res
    )
)

print(
    "\nMax |P^T Z_res|:",
    phase_corr_Z
)

print(
    "Max |P^T R_res|:",
    phase_corr_R
)


# Relative reconstruction energy remaining
Z_energy_fraction = (
    np.sum(
        Z_res ** 2
    )
    /
    np.sum(
        Z_rate ** 2
    )
)

R_energy_fraction = (
    np.sum(
        R_res ** 2
    )
    /
    np.sum(
        R_rate ** 2
    )
)

print(
    "\nRegulator residual energy fraction:",
    Z_energy_fraction
)

print(
    "Response residual energy fraction:",
    R_energy_fraction
)


# --------------------------------------------
# TACC3 specifically
# --------------------------------------------

tacc3_idx = np.where(
    response_genes == "TACC3"
)[0][0]

tacc3_total_var = np.var(
    R_rate[
        :, tacc3_idx
    ]
)

tacc3_res_var = np.var(
    R_res[
        :, tacc3_idx
    ]
)

print(
    "\nTACC3 residual variance fraction:",
    tacc3_res_var
    / tacc3_total_var
)
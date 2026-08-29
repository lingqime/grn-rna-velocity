# ============================================
# Cell 55. Audit kinetic variables directly
# from the official compact checkpoint
# ============================================

import numpy as np
import pandas as pd

# Directly from official extracted VeloCycle output
log_beta_vc = np.asarray(
    vc["log_betas"],
    dtype=np.float64
).squeeze()

log_gamma_vc = np.asarray(
    vc["log_gammas"],
    dtype=np.float64
).squeeze()

print("log_beta_vc shape:", log_beta_vc.shape)
print("log_gamma_vc shape:", log_gamma_vc.shape)

print("\nOfficial log_beta summary:")
print(
    pd.Series(log_beta_vc).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nOfficial log_gamma summary:")
print(
    pd.Series(log_gamma_vc).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

# Exponentiate exactly once
beta_vc_direct = np.exp(log_beta_vc)
gamma_vc_direct = np.exp(log_gamma_vc)

print("\nexp(log_beta) summary:")
print(
    pd.Series(beta_vc_direct).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nexp(log_gamma) summary:")
print(
    pd.Series(gamma_vc_direct).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

# Compare against the vectors actually used in Cell 49/53
expected_beta_cycle = (
    (2.0 * np.pi / omega_nt)
    * beta_vc_direct
)

expected_gamma_cycle = (
    (2.0 * np.pi / omega_nt)
    * gamma_vc_direct
)

print("\nCompare to beta_vec used in kinetic test:")
print(
    "max abs difference:",
    np.max(
        np.abs(
            expected_beta_cycle
            - beta_vec
        )
    )
)

print("\nCompare to gamma_vec used in kinetic test:")
print(
    "max abs difference:",
    np.max(
        np.abs(
            expected_gamma_cycle
            - gamma_vec
        )
    )
)

# Correct official global gamma scale
gamma_geom_direct = np.exp(
    np.mean(log_gamma_vc)
)

print("\nCorrect exp(mean(log_gamma)):")
print(gamma_geom_direct)

print("\nOfficial-style omega_NT / gamma_geom:")
print(
    omega_nt / gamma_geom_direct
)
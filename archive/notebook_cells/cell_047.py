# ============================================
# Cell 47. Build integral-response matrix Y
# for all 426 response genes
# ============================================

import numpy as np
import pandas as pd

n_intervals = len(interval_meta)
n_response = len(vc_genes)

Y_integral = np.full(
    (n_intervals, n_response),
    np.nan,
    dtype=np.float64
)

# Keep the two components separately for QC
delta_U = np.full_like(Y_integral, np.nan)
beta_int_U = np.full_like(Y_integral, np.nan)

beta_vec = beta_cycle.loc[vc_genes].to_numpy(dtype=float)

for i, row in interval_meta.iterrows():

    qi = int(row["condition_idx"])
    a = int(row["bin_a"])
    b = int(row["bin_b"])
    dt = float(row["delta_t"])

    u_a = U_traj_vc[qi, a, :]
    u_b = U_traj_vc[qi, b, :]

    # Endpoint change
    du = u_b - u_a

    # Trapezoidal integral of u
    int_u = 0.5 * dt * (u_a + u_b)

    # beta_g * integral u_g dt
    beta_term = beta_vec * int_u

    delta_U[i, :] = du
    beta_int_U[i, :] = beta_term

    # Manuscript integral-equation response
    # h_g(q) = 0 in the current perturbation encoding
    Y_integral[i, :] = du + beta_term


# --------------------------------------------
# Basic diagnostics
# --------------------------------------------

print("Y_integral shape:", Y_integral.shape)

print(
    "Expected shape:",
    (len(interval_meta), len(vc_genes))
)

print("\nAny NaN in Y:",
      bool(np.isnan(Y_integral).any()))

print("Any inf in Y:",
      bool(np.isinf(Y_integral).any()))

print("\nY summary:")
print(
    pd.Series(Y_integral.ravel()).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nDelta-U component summary:")
print(
    pd.Series(delta_U.ravel()).describe(
        percentiles=[0.01, 0.50, 0.99]
    )
)

print("\nBeta-integral component summary:")
print(
    pd.Series(beta_int_U.ravel()).describe(
        percentiles=[0.01, 0.50, 0.99]
    )
)

# --------------------------------------------
# Per-gene scale diagnostics
# --------------------------------------------

y_rms = np.sqrt(
    np.mean(Y_integral ** 2, axis=0)
)

du_rms = np.sqrt(
    np.mean(delta_U ** 2, axis=0)
)

beta_rms = np.sqrt(
    np.mean(beta_int_U ** 2, axis=0)
)

response_scale_qc = pd.DataFrame(
    {
        "beta_cycle": beta_vec,
        "Y_rms": y_rms,
        "dU_rms": du_rms,
        "betaU_rms": beta_rms,
    },
    index=vc_genes
)

response_scale_qc["beta_to_dU_rms"] = (
    response_scale_qc["betaU_rms"]
    / np.maximum(
        response_scale_qc["dU_rms"],
        1e-12
    )
)

print("\nGenes with largest Y RMS:")
print(
    response_scale_qc
    .sort_values("Y_rms", ascending=False)
    .head(15)
)

print("\nGenes with largest beta_cycle:")
print(
    response_scale_qc
    .sort_values("beta_cycle", ascending=False)
    .head(15)
)

# --------------------------------------------
# Exact first-row sanity check
# --------------------------------------------

i = 0
qi = int(interval_meta.loc[i, "condition_idx"])
a = int(interval_meta.loc[i, "bin_a"])
b = int(interval_meta.loc[i, "bin_b"])
dt = float(interval_meta.loc[i, "delta_t"])

manual_first = (
    U_traj_vc[qi, b, :]
    - U_traj_vc[qi, a, :]
    + beta_vec
    * 0.5
    * dt
    * (
        U_traj_vc[qi, a, :]
        + U_traj_vc[qi, b, :]
    )
)

print(
    "\nFirst interval formula matches:",
    np.allclose(
        Y_integral[0],
        manual_first
    )
)
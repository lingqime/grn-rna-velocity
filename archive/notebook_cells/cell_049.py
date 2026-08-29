# ============================================
# Cell 49. Kinetic consistency check using
# ds/dt = beta*u - gamma*s
# ============================================

import numpy as np
import pandas as pd

n_intervals = len(interval_meta)
n_response = len(vc_genes)

gamma_vec = gamma_cycle.loc[
    vc_genes
].to_numpy(dtype=float)

delta_S = np.full(
    (n_intervals, n_response),
    np.nan,
    dtype=np.float64
)

kinetic_rhs = np.full_like(
    delta_S,
    np.nan
)

for i, row in interval_meta.iterrows():

    qi = int(row["condition_idx"])
    a = int(row["bin_a"])
    b = int(row["bin_b"])
    dt = float(row["delta_t"])

    # Spliced trajectory
    s_a = S_response_traj_vc[qi, a, :]
    s_b = S_response_traj_vc[qi, b, :]

    # Unspliced trajectory
    u_a = U_traj_vc[qi, a, :]
    u_b = U_traj_vc[qi, b, :]

    # Observed change in spliced abundance
    ds = s_b - s_a

    # Trapezoidal integrals
    int_u = 0.5 * dt * (u_a + u_b)
    int_s = 0.5 * dt * (s_a + s_b)

    rhs = (
        beta_vec * int_u
        - gamma_vec * int_s
    )

    delta_S[i, :] = ds
    kinetic_rhs[i, :] = rhs


# --------------------------------------------
# Global diagnostics
# --------------------------------------------

print("delta_S shape:", delta_S.shape)
print("kinetic_rhs shape:", kinetic_rhs.shape)

print(
    "\nAny NaN:",
    bool(
        np.isnan(delta_S).any()
        or np.isnan(kinetic_rhs).any()
    )
)

print(
    "Any inf:",
    bool(
        np.isinf(delta_S).any()
        or np.isinf(kinetic_rhs).any()
    )
)

# Global correlation
global_corr = np.corrcoef(
    delta_S.ravel(),
    kinetic_rhs.ravel()
)[0, 1]

print(
    "\nGlobal correlation "
    "DeltaS vs kinetic RHS:",
    global_corr
)

# --------------------------------------------
# Per-gene diagnostics
# --------------------------------------------

rows = []

for j, g in enumerate(vc_genes):

    obs = delta_S[:, j]
    pred = kinetic_rhs[:, j]

    corr = np.corrcoef(obs, pred)[0, 1]

    rmse = np.sqrt(
        np.mean((obs - pred) ** 2)
    )

    obs_rms = np.sqrt(
        np.mean(obs ** 2)
    )

    pred_rms = np.sqrt(
        np.mean(pred ** 2)
    )

    rel_rmse = (
        rmse / max(obs_rms, 1e-12)
    )

    rows.append(
        {
            "gene": g,
            "beta_cycle": beta_vec[j],
            "gamma_cycle": gamma_vec[j],
            "corr": corr,
            "obs_dS_rms": obs_rms,
            "rhs_rms": pred_rms,
            "relative_rmse": rel_rmse,
        }
    )

kinetic_qc = pd.DataFrame(
    rows
).set_index("gene")


print("\nPer-gene correlation summary:")
print(
    kinetic_qc["corr"].describe(
        percentiles=[
            0.01, 0.05, 0.10,
            0.25, 0.50, 0.75,
            0.90, 0.95, 0.99
        ]
    )
)

print("\nPer-gene relative RMSE summary:")
print(
    kinetic_qc["relative_rmse"].describe(
        percentiles=[
            0.01, 0.05, 0.10,
            0.25, 0.50, 0.75,
            0.90, 0.95, 0.99
        ]
    )
)

print("\nGenes with largest beta:")
print(
    kinetic_qc
    .sort_values(
        "beta_cycle",
        ascending=False
    )
    .head(15)
)

print("\nGenes with best kinetic agreement:")
print(
    kinetic_qc
    .sort_values(
        "corr",
        ascending=False
    )
    .head(15)
)

print("\nGenes with worst kinetic agreement:")
print(
    kinetic_qc
    .sort_values(
        "corr",
        ascending=True
    )
    .head(15)
)
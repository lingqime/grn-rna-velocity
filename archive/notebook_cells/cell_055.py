# ============================================
# Cell 53. Kinetic consistency check
# on RAW count trajectories
# ============================================

import numpy as np
import pandas as pd

n_intervals = len(interval_rows)
n_response = len(vc_genes)

delta_S_raw = np.full(
    (n_intervals, n_response),
    np.nan,
    dtype=np.float64
)

kinetic_rhs_raw = np.full_like(
    delta_S_raw,
    np.nan
)

for i, row in enumerate(interval_rows):

    qi = int(row["condition_idx"])
    a = int(row["bin_a"])
    b = int(row["bin_b"])
    dt = float(row["delta_t"])

    s_a = S_response_traj_raw[qi, a, :]
    s_b = S_response_traj_raw[qi, b, :]

    u_a = U_response_traj_raw[qi, a, :]
    u_b = U_response_traj_raw[qi, b, :]

    # Observed change in raw spliced counts
    ds = s_b - s_a

    # Trapezoidal integrals
    int_u = 0.5 * dt * (u_a + u_b)
    int_s = 0.5 * dt * (s_a + s_b)

    rhs = (
        beta_vec * int_u
        - gamma_vec * int_s
    )

    delta_S_raw[i, :] = ds
    kinetic_rhs_raw[i, :] = rhs


# --------------------------------------------
# Basic checks
# --------------------------------------------

print("delta_S_raw shape:", delta_S_raw.shape)
print("kinetic_rhs_raw shape:", kinetic_rhs_raw.shape)

print(
    "\nAny NaN:",
    bool(
        np.isnan(delta_S_raw).any()
        or np.isnan(kinetic_rhs_raw).any()
    )
)

print(
    "Any inf:",
    bool(
        np.isinf(delta_S_raw).any()
        or np.isinf(kinetic_rhs_raw).any()
    )
)


# --------------------------------------------
# Global correlation
# --------------------------------------------

global_corr_raw = np.corrcoef(
    delta_S_raw.ravel(),
    kinetic_rhs_raw.ravel()
)[0, 1]

print(
    "\nGlobal correlation "
    "DeltaS vs kinetic RHS:",
    global_corr_raw
)


# --------------------------------------------
# Per-gene diagnostics
# --------------------------------------------

rows = []

for j, g in enumerate(vc_genes):

    obs = delta_S_raw[:, j]
    pred = kinetic_rhs_raw[:, j]

    if (
        np.std(obs) > 0
        and np.std(pred) > 0
    ):
        corr = np.corrcoef(
            obs,
            pred
        )[0, 1]
    else:
        corr = np.nan

    obs_rms = np.sqrt(
        np.mean(obs ** 2)
    )

    rhs_rms = np.sqrt(
        np.mean(pred ** 2)
    )

    rmse = np.sqrt(
        np.mean(
            (obs - pred) ** 2
        )
    )

    rel_rmse = (
        rmse
        / max(obs_rms, 1e-12)
    )

    rows.append(
        {
            "gene": g,
            "beta_cycle": beta_vec[j],
            "gamma_cycle": gamma_vec[j],
            "corr": corr,
            "obs_dS_rms": obs_rms,
            "rhs_rms": rhs_rms,
            "relative_rmse": rel_rmse,
        }
    )

kinetic_qc_raw = pd.DataFrame(
    rows
).set_index("gene")


print("\nPer-gene correlation summary:")
print(
    kinetic_qc_raw["corr"].describe(
        percentiles=[
            0.01, 0.05, 0.10,
            0.25, 0.50, 0.75,
            0.90, 0.95, 0.99
        ]
    )
)

print("\nPer-gene relative RMSE summary:")
print(
    kinetic_qc_raw[
        "relative_rmse"
    ].describe(
        percentiles=[
            0.01, 0.05, 0.10,
            0.25, 0.50, 0.75,
            0.90, 0.95, 0.99
        ]
    )
)

print("\nGenes with largest beta:")
print(
    kinetic_qc_raw
    .sort_values(
        "beta_cycle",
        ascending=False
    )
    .head(15)
)

print("\nGenes with best kinetic agreement:")
print(
    kinetic_qc_raw
    .sort_values(
        "corr",
        ascending=False
    )
    .head(15)
)
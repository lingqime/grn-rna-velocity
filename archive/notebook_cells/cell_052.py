# ============================================
# Cell 52. Re-check VeloCycle kinetics
# on the correct RAW count scale
# ============================================

import numpy as np
import pandas as pd

# We should already have raw binned trajectories:
# S_response_traj_raw : condition x bin x gene
# U_response_traj_raw : condition x bin x gene
#
# interval_rows contains the same 1214 usable adjacent-bin intervals
# used previously.

delta_S_raw = []
kinetic_rhs_raw = []

for row in interval_rows.itertuples(index=False):

    q = int(row.condition_idx)
    a = int(row.bin_a)
    b = int(row.bin_b)

    dt = float(row.dt)

    S_a = S_response_traj_raw[q, a, :]
    S_b = S_response_traj_raw[q, b, :]

    U_a = U_response_traj_raw[q, a, :]
    U_b = U_response_traj_raw[q, b, :]

    # observed change in spliced counts
    dS = S_b - S_a

    # trapezoidal integral of beta*U - gamma*S
    rhs = dt * 0.5 * (
        beta_cycle * (U_a + U_b)
        - gamma_cycle * (S_a + S_b)
    )

    delta_S_raw.append(dS)
    kinetic_rhs_raw.append(rhs)

delta_S_raw = np.asarray(delta_S_raw)
kinetic_rhs_raw = np.asarray(kinetic_rhs_raw)

print("delta_S_raw shape:", delta_S_raw.shape)
print("kinetic_rhs_raw shape:", kinetic_rhs_raw.shape)

print("\nAny NaN:",
      np.isnan(delta_S_raw).any()
      or np.isnan(kinetic_rhs_raw).any())

print("Any inf:",
      np.isinf(delta_S_raw).any()
      or np.isinf(kinetic_rhs_raw).any())


# --------------------------------------------
# Global correlation
# --------------------------------------------

global_corr_raw = np.corrcoef(
    delta_S_raw.ravel(),
    kinetic_rhs_raw.ravel()
)[0, 1]

print(
    "\nGlobal correlation DeltaS vs kinetic RHS:",
    global_corr_raw
)


# --------------------------------------------
# Per-gene diagnostics
# --------------------------------------------

rows = []

for j, g in enumerate(response_genes_vc):

    obs = delta_S_raw[:, j]
    pred = kinetic_rhs_raw[:, j]

    if np.std(obs) > 0 and np.std(pred) > 0:
        corr = np.corrcoef(obs, pred)[0, 1]
    else:
        corr = np.nan

    obs_rms = np.sqrt(np.mean(obs**2))
    rhs_rms = np.sqrt(np.mean(pred**2))

    rmse = np.sqrt(np.mean((obs - pred)**2))

    relative_rmse = (
        rmse / obs_rms
        if obs_rms > 0
        else np.nan
    )

    rows.append({
        "gene": g,
        "beta_cycle": beta_cycle[j],
        "gamma_cycle": gamma_cycle[j],
        "corr": corr,
        "obs_dS_rms": obs_rms,
        "rhs_rms": rhs_rms,
        "relative_rmse": relative_rmse,
    })

kinetic_check_raw = (
    pd.DataFrame(rows)
    .set_index("gene")
)

print("\nPer-gene correlation summary:")
print(
    kinetic_check_raw["corr"].describe(
        percentiles=[
            0.01, 0.05, 0.10, 0.25,
            0.50, 0.75, 0.90, 0.95, 0.99
        ]
    )
)

print("\nPer-gene relative RMSE summary:")
print(
    kinetic_check_raw["relative_rmse"].describe(
        percentiles=[
            0.01, 0.05, 0.10, 0.25,
            0.50, 0.75, 0.90, 0.95, 0.99
        ]
    )
)

print("\nGenes with best kinetic agreement:")
print(
    kinetic_check_raw
    .sort_values("corr", ascending=False)
    .head(15)
)

print("\nGenes with largest beta:")
print(
    kinetic_check_raw
    .sort_values("beta_cycle", ascending=False)
    .head(15)
)
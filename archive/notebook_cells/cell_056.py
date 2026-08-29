# ============================================
# Cell 54. Diagnose VeloCycle's global
# kinetic time-scale normalization
# ============================================

import numpy as np
import pandas as pd

# Raw VeloCycle kinetic parameters
beta_raw = np.exp(
    beta_series.loc[vc_genes].to_numpy(dtype=float)
)

gamma_raw = np.exp(
    gamma_series.loc[vc_genes].to_numpy(dtype=float)
)

# Official notebook's global gamma scale
gamma_geom = np.exp(
    np.mean(
        gamma_series.loc[vc_genes].to_numpy(dtype=float)
    )
)

# Official notebook displays omega / gamma_geom
omega_nt_over_gamma = (
    omega_nt / gamma_geom
)

# Our previous conversion
old_time_factor = (
    2.0 * np.pi / omega_nt
)

# Alternative dimensionless kinetics using
# gamma_geom as the global time unit
beta_over_gamma_geom = (
    beta_raw / gamma_geom
)

gamma_over_gamma_geom = (
    gamma_raw / gamma_geom
)

print("Global geometric-mean gamma:")
print(gamma_geom)

print("\nNT omega:")
print(omega_nt)

print("\nOfficial-style NT omega / gamma_geom:")
print(omega_nt_over_gamma)

print("\nPrevious 2pi / omega_NT factor:")
print(old_time_factor)


print("\nRaw beta summary:")
print(
    pd.Series(beta_raw).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nRaw gamma summary:")
print(
    pd.Series(gamma_raw).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nBeta / gamma_geom summary:")
print(
    pd.Series(
        beta_over_gamma_geom
    ).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nGamma / gamma_geom summary:")
print(
    pd.Series(
        gamma_over_gamma_geom
    ).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)


# Inspect the problematic genes directly
inspect_genes = [
    "YBX1",
    "RPL23",
    "RPS27L",
    "RHOA",
    "ACTB",
    "MKI67",
    "CENPE",
    "ASPM",
]

rows = []

for g in inspect_genes:

    j = vc_genes.index(g)

    rows.append({
        "gene": g,
        "beta_raw": beta_raw[j],
        "gamma_raw": gamma_raw[j],
        "beta/gamma_geom":
            beta_over_gamma_geom[j],
        "gamma/gamma_geom":
            gamma_over_gamma_geom[j],
        "old_beta_cycle":
            beta_vec[j],
        "old_gamma_cycle":
            gamma_vec[j],
        "kinetic_corr_raw":
            kinetic_qc_raw.loc[g, "corr"],
        "relative_rmse_raw":
            kinetic_qc_raw.loc[
                g,
                "relative_rmse"
            ],
    })

scale_diagnostic = (
    pd.DataFrame(rows)
    .set_index("gene")
)

print("\nSelected genes:")
print(scale_diagnostic)
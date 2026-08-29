# ============================================
# Cell 43. Put VeloCycle speed and kinetics
# into NT-cycle normalized time units
# ============================================

import numpy as np
import pandas as pd

# NT angular speed
omega_nt = float(speed_series.loc["non-targeting"])

# --------------------------------------------
# 1. Condition-specific relative cycle speed
# --------------------------------------------

rho_final = (
    speed_series.loc[final_conditions_vc]
    / omega_nt
)

# --------------------------------------------
# 2. Kinetic rates in normalized time:
#    1 time unit = one NT cell cycle
# --------------------------------------------

time_scale = 2.0 * np.pi / omega_nt

beta_cycle = (
    beta_series.loc[vc_genes]
    * time_scale
)

gamma_cycle = (
    gamma_series.loc[vc_genes]
    * time_scale
)

# --------------------------------------------
# 3. Duration of one 10-bin phase interval
# --------------------------------------------

delta_x = 1.0 / 10.0

delta_t = delta_x / rho_final

# --------------------------------------------
# Diagnostics
# --------------------------------------------

print("omega_NT:", omega_nt)
print("time-scale factor 2pi / omega_NT:", time_scale)

print("\nRelative speed rho:")
print(rho_final.describe())

print("\nDuration of one 0.1-cycle interval:")
print(delta_t.describe())

print("\nBeta in NT-cycle units:")
print(beta_cycle.describe())

print("\nGamma in NT-cycle units:")
print(gamma_cycle.describe())

print("\nSlowest condition:")
slowest = rho_final.idxmin()
print(
    slowest,
    "rho =", rho_final.loc[slowest],
    "delta_t =", delta_t.loc[slowest]
)

print("\nFastest condition:")
fastest = rho_final.idxmax()
print(
    fastest,
    "rho =", rho_final.loc[fastest],
    "delta_t =", delta_t.loc[fastest]
)

print("\nNT interval duration:")
print(delta_t.loc["non-targeting"])
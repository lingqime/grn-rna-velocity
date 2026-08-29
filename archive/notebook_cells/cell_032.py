# ============================================
# Cell 33. Inspect VeloCycle speed estimates
# for NT and our final 153 perturbations
# ============================================

import numpy as np
import pandas as pd

vc = np.load(extract_path, allow_pickle=True)

condition_names_vc = vc["condition_names"].astype(str)
speed_raw_vc = vc["speed_raw"].astype(float)

speed_series = pd.Series(
    speed_raw_vc,
    index=condition_names_vc,
    name="nu0"
)

# NT
nt_speed = speed_series.loc["non-targeting"]

print("NT nu0:", nt_speed)

print("\nAll 986 conditions:")
print(speed_series.describe(
    percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\nSign counts, all conditions:")
print("positive:", int((speed_series > 0).sum()))
print("zero:", int((speed_series == 0).sum()))
print("negative:", int((speed_series < 0).sum()))

# Our final perturbation set
missing = [
    g for g in final_perturbations_qc
    if g not in speed_series.index
]

speed_qc = speed_series.loc[final_perturbations_qc]

print("\nFinal perturbations:", len(final_perturbations_qc))
print("Missing from VeloCycle speed:", len(missing))

print("\nSelected 153 speed summary:")
print(speed_qc.describe(
    percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\nSign counts, selected perturbations:")
print("positive:", int((speed_qc > 0).sum()))
print("zero:", int((speed_qc == 0).sum()))
print("negative:", int((speed_qc < 0).sum()))

print("\nNegative selected conditions:")
print(speed_qc[speed_qc < 0].sort_values())
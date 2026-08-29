# ============================================
# Cell 29b. Validate full VeloCycle phase
# against official phase metadata
# ============================================

import subprocess
import os

validate_code = r"""
import sys
import gzip
import pickle
import io
import torch
import numpy as np
import pandas as pd

repo = r"__VELO_REPO__"
sys.path.insert(0, repo)

import velocycle

pickle_path = r"__PICKLE_PATH__"
phase_csv = r"__PHASE_CSV__"

torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b),
    map_location="cpu"
)

with gzip.open(pickle_path, "rb") as f:
    phase_fit, velocity_fit, data_to_fit = pickle.load(f)

# Official full-cell phase
phase_obj = phase_fit.phase_pyro

phi_full = np.asarray(
    phase_obj.directions,
    dtype=float
)

cell_ids = np.asarray(
    phase_obj.phi_xy.columns.astype(str)
)

print("Full phase cells:", len(phi_full))
print("phi range:", phi_full.min(), phi_full.max())

print(
    "Phase columns match data_to_fit.obs_names:",
    np.array_equal(
        cell_ids,
        data_to_fit.obs_names.astype(str).values
    )
)

print(
    "phis_pyro shape:",
    np.asarray(phase_fit.phis_pyro).shape
)

print(
    "phis_pyro == phase_pyro.phi_xy:",
    np.allclose(
        np.asarray(phase_fit.phis_pyro),
        phase_obj.phi_xy.values
    )
)

# Validate against official CSV
meta = pd.read_csv(
    phase_csv,
    index_col=0
)

phi_series = pd.Series(
    phi_full,
    index=cell_ids,
    name="phi_pickle"
)

common = meta.index.intersection(phi_series.index)

phi_a = phi_series.loc[common].values
phi_b = meta.loc[common, "cell_cycle_phi"].values

delta = np.angle(
    np.exp(1j * (phi_a - phi_b))
)

abs_error = np.abs(delta)

print()
print("Matched cells with official CSV:", len(common))
print("Median circular error (rad):", np.median(abs_error))
print("Mean circular error (rad):", np.mean(abs_error))
print("Max circular error (rad):", np.max(abs_error))

print()
print("First 5 comparisons:")
print(
    pd.DataFrame({
        "pickle_phi": phi_a[:5],
        "csv_phi": phi_b[:5],
        "abs_circular_error": abs_error[:5],
    })
)
"""

validate_code = (
    validate_code
    .replace("__VELO_REPO__", str(velocycle_legacy))
    .replace("__PICKLE_PATH__", str(pickle_986))
    .replace("__PHASE_CSV__", str(phase_metadata))
)

env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
}

result = subprocess.run(
    [str(python_official), "-c", validate_code],
    capture_output=True,
    text=True,
    env=env,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)
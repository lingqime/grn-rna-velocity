# ============================================
# Cell 30. Check phase consistency between
# phase_fit and velocity_fit
# ============================================

import subprocess
import os

check_code = r"""
import sys
import gzip
import pickle
import io
import torch
import numpy as np

repo = r"__VELO_REPO__"
sys.path.insert(0, repo)

import velocycle

pickle_path = r"__PICKLE_PATH__"

torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b),
    map_location="cpu"
)

with gzip.open(pickle_path, "rb") as f:
    phase_fit, velocity_fit, data_to_fit = pickle.load(f)

p_phase = phase_fit.phase_pyro
p_vel   = velocity_fit.phase_pyro

phi_phase = np.asarray(p_phase.directions, dtype=float)
phi_vel   = np.asarray(p_vel.directions, dtype=float)

ids_phase = np.asarray(p_phase.phi_xy.columns.astype(str))
ids_vel   = np.asarray(p_vel.phi_xy.columns.astype(str))

print("phase_fit cells:", len(phi_phase))
print("velocity_fit cells:", len(phi_vel))

print(
    "Cell IDs identical:",
    np.array_equal(ids_phase, ids_vel)
)

print(
    "phi_xy identical:",
    np.allclose(
        p_phase.phi_xy.values,
        p_vel.phi_xy.values
    )
)

delta = np.angle(
    np.exp(1j * (phi_phase - phi_vel))
)
abs_error = np.abs(delta)

print()
print(
    "Median circular difference (rad):",
    np.median(abs_error)
)
print(
    "Mean circular difference (rad):",
    np.mean(abs_error)
)
print(
    "Max circular difference (rad):",
    np.max(abs_error)
)

print()
print("velocity_fit phase range:",
      phi_vel.min(), phi_vel.max())
"""

check_code = (
    check_code
    .replace("__VELO_REPO__", str(velocycle_legacy))
    .replace("__PICKLE_PATH__", str(pickle_986))
)

env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
}

result = subprocess.run(
    [str(python_official), "-c", check_code],
    capture_output=True,
    text=True,
    env=env,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)
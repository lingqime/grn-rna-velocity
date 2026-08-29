# ============================================
# Cell 31. Export compact official VeloCycle
# checkpoint for downstream GRN analysis
# ============================================

import subprocess
import os

extract_path = base_dir / "velocycle_986_official_extract.npz"

extract_code = r"""
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
out_path = r"__OUT_PATH__"

torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b),
    map_location="cpu"
)

print("Loading official VeloCycle pickle...")

with gzip.open(pickle_path, "rb") as f:
    phase_fit, velocity_fit, data_to_fit = pickle.load(f)

# Use velocity-fit phase for internal consistency
phase_obj = velocity_fit.phase_pyro

cell_ids = np.asarray(
    phase_obj.phi_xy.columns.astype(str)
)

phi = np.asarray(
    phase_obj.directions,
    dtype=np.float64
)

# Condition-specific angular speed posterior means
speed_df = velocity_fit.speed_pyro.means

condition_names = np.asarray(
    speed_df.columns.astype(str)
)

speed_raw = np.asarray(
    speed_df.loc["nu0"].values,
    dtype=np.float64
)

# Kinetic parameters align with data_to_fit's genes
gene_names = np.asarray(
    data_to_fit.var_names.astype(str)
)

log_betas = np.asarray(
    velocity_fit.log_betas,
    dtype=np.float64
)

log_gammas = np.asarray(
    velocity_fit.log_gammas,
    dtype=np.float64
)

assert len(cell_ids) == len(phi)
assert len(condition_names) == len(speed_raw)
assert len(gene_names) == len(log_betas) == len(log_gammas)

np.savez_compressed(
    out_path,
    cell_ids=cell_ids,
    phi=phi,
    condition_names=condition_names,
    speed_raw=speed_raw,
    gene_names=gene_names,
    log_betas=log_betas,
    log_gammas=log_gammas,
)

print()
print("Saved:", out_path)
print("cells:", len(cell_ids))
print("conditions:", len(condition_names))
print("genes:", len(gene_names))
print("phi range:", phi.min(), phi.max())
print("speed range:", speed_raw.min(), speed_raw.max())
print("log_beta range:", log_betas.min(), log_betas.max())
print("log_gamma range:", log_gammas.min(), log_gammas.max())
"""

extract_code = (
    extract_code
    .replace("__VELO_REPO__", str(velocycle_legacy))
    .replace("__PICKLE_PATH__", str(pickle_986))
    .replace("__OUT_PATH__", str(extract_path))
)

env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
}

result = subprocess.run(
    [str(python_official), "-c", extract_code],
    capture_output=True,
    text=True,
    env=env,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)
print("Extract exists:", extract_path.exists())

if extract_path.exists():
    print(
        "Extract size (MB):",
        round(extract_path.stat().st_size / 1024**2, 2)
    )
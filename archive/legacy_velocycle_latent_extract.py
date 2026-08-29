"""Historical helper extracted from the original notebook.

This code required the legacy VeloCycle environment and machine-specific
runtime paths. It is archived for provenance and is not part of the canonical
public GRN workflow. The public workflow expects the resulting compact NPZ
under data/processed; see data/README.md.
"""

# ============================================================================
# SOURCE NOTEBOOK INDEX 60: Cell 58. Extract compact latent VeloCycle
# ============================================================================

# ============================================
# Cell 58. Extract compact latent VeloCycle
# quantities from the official 986-condition
# LargeGene pickle
# ============================================

import subprocess
from pathlib import Path

python_official = (
    "/environment/miniconda3/envs/"
    "velocycle-official/bin/python"
)

pkl_path = (
    "/home/featurize/work/project1/"
    "pickle_result_outputs/"
    "Replogle_Saunders_PerturbRPE1_75cells_"
    "phase_velocity_data_fit_LargeGene_986conditions.pkl.gz"
)

out_path = (
    "/home/featurize/work/project1/"
    "velocycle_986_latent_extract.npz"
)

worktree = (
    "/home/featurize/work/project1/"
    "velocycle_91123be"
)

script = r'''
import sys
import io
import gzip
import pickle
import numpy as np
import torch

pkl_path = sys.argv[1]
out_path = sys.argv[2]
worktree = sys.argv[3]

sys.path.insert(0, worktree)

# Force CUDA-saved tensors onto CPU
_original_load_from_bytes = torch.storage._load_from_bytes

torch.storage._load_from_bytes = (
    lambda b:
    torch.load(
        io.BytesIO(b),
        map_location="cpu"
    )
)

print("Opening pickle...")

with gzip.open(pkl_path, "rb") as f:
    phase_fit, velocity_fit, data_to_fit = pickle.load(f)

print("Loaded.")

payload = {}

# ------------------------------------------------
# Velocity-fit Fourier coefficients
# ------------------------------------------------

if hasattr(velocity_fit, "fourier_coef"):
    x = np.asarray(
        velocity_fit.fourier_coef
    )
    payload["velocity_fourier_coef"] = x

    print(
        "velocity_fourier_coef:",
        x.shape,
        x.dtype
    )

if hasattr(velocity_fit, "cycle_pyro"):
    cp = velocity_fit.cycle_pyro

    if hasattr(cp, "means"):
        payload[
            "velocity_cycle_means"
        ] = cp.means.to_numpy()

        payload[
            "velocity_cycle_index"
        ] = np.asarray(
            cp.means.index,
            dtype=object
        )

        payload[
            "velocity_cycle_genes"
        ] = np.asarray(
            cp.means.columns,
            dtype=object
        )

        print(
            "velocity_cycle_means:",
            cp.means.shape
        )


# ------------------------------------------------
# Phase-fit Fourier coefficients
# ------------------------------------------------

if hasattr(phase_fit, "fourier_coef"):
    x = np.asarray(
        phase_fit.fourier_coef
    )
    payload["phase_fourier_coef"] = x

    print(
        "phase_fourier_coef:",
        x.shape,
        x.dtype
    )

if hasattr(phase_fit, "cycle_pyro"):
    cp = phase_fit.cycle_pyro

    if hasattr(cp, "means"):
        payload[
            "phase_cycle_means"
        ] = cp.means.to_numpy()

        payload[
            "phase_cycle_index"
        ] = np.asarray(
            cp.means.index,
            dtype=object
        )

        payload[
            "phase_cycle_genes"
        ] = np.asarray(
            cp.means.columns,
            dtype=object
        )

        print(
            "phase_cycle_means:",
            cp.means.shape
        )


# ------------------------------------------------
# Count factor used by the phase model
# ------------------------------------------------

if (
    hasattr(phase_fit, "metaparams")
    and phase_fit.metaparams is not None
    and hasattr(
        phase_fit.metaparams,
        "count_factor"
    )
):
    cf = (
        phase_fit.metaparams
        .count_factor
        .detach()
        .cpu()
        .numpy()
    )

    payload["count_factor"] = cf

    print(
        "count_factor:",
        cf.shape,
        cf.dtype,
        "min =", np.nanmin(cf),
        "max =", np.nanmax(cf)
    )

else:
    print(
        "phase_fit.metaparams.count_factor "
        "NOT AVAILABLE"
    )


# ------------------------------------------------
# Cell / gene identifiers for alignment
# ------------------------------------------------

payload["cell_ids"] = np.asarray(
    data_to_fit.obs_names,
    dtype=object
)

payload["gene_names"] = np.asarray(
    data_to_fit.var_names,
    dtype=object
)

print(
    "data_to_fit:",
    data_to_fit.shape
)

np.savez_compressed(
    out_path,
    **payload
)

print("\nSaved:")
print(out_path)

print("\nSaved keys:")
print(list(payload.keys()))
'''

result = subprocess.run(
    [
        python_official,
        "-c",
        script,
        pkl_path,
        out_path,
        worktree,
    ],
    text=True,
    capture_output=True,
)

print(result.stdout)

if result.stderr:
    print("STDERR:")
    print(result.stderr)

print(
    "\nReturn code:",
    result.returncode
)

print(
    "Output exists:",
    Path(out_path).exists()
)

if Path(out_path).exists():
    print(
        "Output size MB:",
        Path(out_path).stat().st_size
        / 1024**2
    )



# ============================================
# Cell 27. Inspect phase, speed, and kinetics
# from official VeloCycle result
# ============================================

import subprocess
import os

inspect_code = rf"""
import sys
import gzip
import pickle
import io
import torch
import numpy as np
import pandas as pd

repo = r"{velocycle_legacy}"
sys.path.insert(0, repo)

import velocycle

pickle_path = r"{pickle_986}"

# Force CUDA tensors onto CPU
torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b),
    map_location="cpu"
)

with gzip.open(pickle_path, "rb") as f:
    phase_fit, velocity_fit, data_to_fit = pickle.load(f)


def summarize(name, x):
    print()
    print("=" * 70)
    print(name)
    print("type:", type(x))

    if hasattr(x, "shape"):
        try:
            print("shape:", x.shape)
        except Exception:
            pass

    if isinstance(x, torch.Tensor):
        print("dtype:", x.dtype)
        print("device:", x.device)

    if isinstance(x, pd.DataFrame):
        print("index head:", list(x.index[:5]))
        print("columns head:", list(x.columns[:10]))

    elif isinstance(x, pd.Series):
        print("index head:", list(x.index[:5]))
        print("head values:", x.head().tolist())

    elif isinstance(x, dict):
        print("keys head:", list(x.keys())[:20])

    elif hasattr(x, "means"):
        print("has .means:", True)
        print(".means type:", type(x.means))

        if hasattr(x.means, "shape"):
            print(".means shape:", x.means.shape)

        if isinstance(x.means, pd.DataFrame):
            print(".means index head:", list(x.means.index[:5]))
            print(".means columns head:", list(x.means.columns[:10]))


print("data_to_fit shape:", data_to_fit.shape)
print("obs columns:", list(data_to_fit.obs.columns))
print("var columns:", list(data_to_fit.var.columns))
print("first 10 genes:", data_to_fit.var_names[:10].tolist())

summarize("phase_fit.condition", phase_fit.condition)
summarize("phase_fit.phis_pyro", phase_fit.phis_pyro)
summarize("phase_fit.phase_pyro", phase_fit.phase_pyro)

summarize("velocity_fit.condition", velocity_fit.condition)
summarize("velocity_fit.speed_pyro", velocity_fit.speed_pyro)
summarize("velocity_fit.log_betas", velocity_fit.log_betas)
summarize("velocity_fit.log_gammas", velocity_fit.log_gammas)
"""

env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
}

result = subprocess.run(
    [str(python_official), "-c", inspect_code],
    capture_output=True,
    text=True,
    env=env,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)
# ============================================
# Cell 17b. Test VeloCycle directly from source
# without installing the package
# ============================================

import subprocess

check_code = rf"""
import sys

repo = r"{velocycle_repo}"
sys.path.insert(0, repo)

print("Python:", sys.version)
print("Using source repo:", repo)

import numpy as np
import scipy
import pandas as pd
import anndata
import torch
import pyro

print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pd.__version__)
print("anndata:", anndata.__version__)
print("torch:", torch.__version__)
print("pyro:", pyro.__version__)

import velocycle

print("velocycle imported successfully")
print("velocycle path:", velocycle.__file__)

print("CUDA available:", torch.cuda.is_available())
"""

result = subprocess.run(
    [str(python_official), "-c", check_code],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)
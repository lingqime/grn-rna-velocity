# ============================================
# Cell 19. Re-test VeloCycle import
# ============================================

import subprocess

check_code = rf"""
import sys

repo = r"{velocycle_repo}"
sys.path.insert(0, repo)

import numpy as np
import scipy
import pandas as pd
import anndata
import torch
import pyro
import matplotlib
import scanpy

print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pd.__version__)
print("anndata:", anndata.__version__)
print("torch:", torch.__version__)
print("pyro:", pyro.__version__)
print("matplotlib:", matplotlib.__version__)
print("scanpy:", scanpy.__version__)

import velocycle

print("\nvelocycle imported successfully")
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
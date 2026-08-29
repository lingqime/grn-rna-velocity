# ============================================
# Cell 25. Test import using legacy VeloCycle source
# ============================================

import subprocess

check_code = rf"""
import sys

repo = r"{velocycle_legacy}"
sys.path.insert(0, repo)

import torch
import pyro

print("torch:", torch.__version__)
print("pyro:", pyro.__version__)

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
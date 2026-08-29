# ============================================
# Cell 26. Load official 986-condition pickle
# safely onto CPU and inspect top-level objects
# ============================================

import subprocess

pickle_986 = (
    base_dir
    / "pickle_result_outputs"
    / "Replogle_Saunders_PerturbRPE1_75cells_phase_velocity_data_fit_LargeGene_986conditions.pkl.gz"
)

inspect_code = rf"""
import sys
import gzip
import pickle
import io
import torch

# Use the syntax-valid historical VeloCycle source
repo = r"{velocycle_legacy}"
sys.path.insert(0, repo)

import velocycle

pickle_path = r"{pickle_986}"

print("VeloCycle:", velocycle.__file__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Pickle:", pickle_path)
print()

# ------------------------------------------------
# Force tensors stored on cuda:0 to load onto CPU
# ------------------------------------------------
_original_load_from_bytes = torch.storage._load_from_bytes

torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b),
    map_location="cpu"
)

print("Loading gzip pickle onto CPU...")
print("This may take several minutes.")
print()

with gzip.open(pickle_path, "rb") as f:
    obj = pickle.load(f)

print("LOAD SUCCESSFUL")
print()

print("Top-level type:", type(obj))

try:
    print("Top-level length:", len(obj))
except Exception:
    pass

if isinstance(obj, (list, tuple)):
    for i, x in enumerate(obj):
        print()
        print("=" * 60)
        print("Object", i)
        print("type:", type(x))

        if hasattr(x, "__dict__"):
            keys = sorted(x.__dict__.keys())
            print("attributes:")
            for k in keys:
                print("  ", k)
"""

env = {
    **__import__("os").environ,
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
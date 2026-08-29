# ============================================
# Cell 18b. Install missing VeloCycle
# runtime scientific dependencies
# ============================================

import subprocess

runtime_packages = [
    "matplotlib==3.7.4",
    "scanpy==1.9.6",
    "seaborn==0.12.2",
    "numba==0.58.1",
    "umap-learn==0.5.5",
    "pynndescent==0.5.11",
]

result = subprocess.run(
    [
        str(pip_official),
        "install",
        *runtime_packages,
    ],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)
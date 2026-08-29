# ============================================
# Cell 16. Install core official VeloCycle stack
# ============================================

import subprocess

cmd = [
    str(pip_official),
    "install",

    # Core numerical stack used by official environment
    "numpy==1.24.4",
    "scipy==1.10.1",
    "pandas==2.0.3",
    "anndata==0.9.2",

    # PyTorch / Pyro versions from official requirements
    "torch==2.1.1",
    "pyro-api==0.1.2",
    "pyro-ppl==1.8.6",

    # Other dependencies directly used by VeloCycle
    "scikit-learn==1.3.2",
    "statsmodels==0.14.0",
    "pycircstat==0.0.2",
    "tqdm==4.66.1",
]

print("Installing official-compatible core dependencies...")
print()

subprocess.run(
    cmd,
    check=True,
)

print("\nCore dependency installation completed.")
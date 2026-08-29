# ============================================
# Cell 14. Create isolated official VeloCycle env
# ============================================

import subprocess
from pathlib import Path

env_name = "velocycle-official"

# Check existing conda environments
result = subprocess.run(
    ["conda", "env", "list"],
    capture_output=True,
    text=True,
    check=True,
)

print(result.stdout)

if env_name not in result.stdout:
    print(f"Creating conda environment: {env_name}")

    subprocess.run(
        [
            "conda", "create",
            "-n", env_name,
            "python=3.10",
            "-y",
        ],
        check=True,
    )
else:
    print(f"Environment already exists: {env_name}")

# Locate its Python executable
conda_prefix = Path.home() / ".conda" / "envs" / env_name

# Also check common alternative location
if not conda_prefix.exists():
    conda_prefix = Path.home() / "anaconda3" / "envs" / env_name

if not conda_prefix.exists():
    conda_prefix = Path.home() / "miniconda3" / "envs" / env_name

print("\nCandidate environment path:", conda_prefix)
print("Exists:", conda_prefix.exists())

if conda_prefix.exists():
    python_exe = conda_prefix / "bin" / "python"

    out = subprocess.run(
        [str(python_exe), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )

    print("Python executable:", python_exe)
    print("Python version:", out.stdout.strip() or out.stderr.strip())
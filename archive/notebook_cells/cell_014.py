# ============================================
# Cell 15. Resolve official VeloCycle env path
# ============================================

from pathlib import Path
import subprocess

velocycle_env = Path(
    "/environment/miniconda3/envs/velocycle-official"
)

python_official = velocycle_env / "bin" / "python"
pip_official = velocycle_env / "bin" / "pip"

print("Environment:", velocycle_env)
print("Exists:", velocycle_env.exists())

print("\nPython:")
print("  path:", python_official)
print("  exists:", python_official.exists())

print("\nPip:")
print("  path:", pip_official)
print("  exists:", pip_official.exists())

if python_official.exists():
    result = subprocess.run(
        [str(python_official), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    print("\nVersion:", result.stdout.strip() or result.stderr.strip())
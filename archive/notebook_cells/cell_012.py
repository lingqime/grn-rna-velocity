# ============================================
# Cell 13. Clone official VeloCycle repository
# and inspect dependency specifications
# ============================================

import subprocess
from pathlib import Path

velocycle_repo = base_dir / "velocycle"

if not velocycle_repo.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/lamanno-epfl/velocycle.git",
            str(velocycle_repo),
        ],
        check=True,
    )
else:
    print("Repository already exists:", velocycle_repo)

print("\n--- requirements.txt ---")
requirements_file = velocycle_repo / "requirements.txt"
print(requirements_file.read_text())

print("\n--- setup.py dependency-related lines ---")
setup_text = (velocycle_repo / "setup.py").read_text()

for line in setup_text.splitlines():
    if (
        "install_requires" in line
        or "torch" in line.lower()
        or "pyro" in line.lower()
        or "numpy" in line.lower()
        or "pandas" in line.lower()
        or "anndata" in line.lower()
    ):
        print(line)
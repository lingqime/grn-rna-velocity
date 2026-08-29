# ============================================
# Cell 20. Install missing IPython dependency
# ============================================

import subprocess

result = subprocess.run(
    [
        str(pip_official),
        "install",
        "ipython==8.12.3",
    ],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)
# ============================================
# Cell 24. Create a clean VeloCycle worktree
# at commit 91123be
# ============================================

import subprocess
from pathlib import Path

velocycle_legacy = base_dir / "velocycle_91123be"

if not velocycle_legacy.exists():
    subprocess.run(
        [
            "git",
            "-C", str(velocycle_repo),
            "worktree",
            "add",
            str(velocycle_legacy),
            "91123be",
        ],
        check=True,
    )
else:
    print("Worktree already exists:", velocycle_legacy)

print("\nLegacy VeloCycle path:")
print(velocycle_legacy)

# Verify commit
result = subprocess.run(
    [
        "git",
        "-C", str(velocycle_legacy),
        "log",
        "-1",
        "--format=%H%n%ad%n%s",
        "--date=iso",
    ],
    capture_output=True,
    text=True,
    check=True,
)

print("\nCommit:")
print(result.stdout)

print("phase_inference_model.py exists:",
      (velocycle_legacy / "velocycle" / "phase_inference_model.py").exists())
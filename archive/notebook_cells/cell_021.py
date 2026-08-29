# ============================================
# Cell 22. Inspect VeloCycle git history
# ============================================

import subprocess

def run_git(args):
    result = subprocess.run(
        ["git", "-C", str(velocycle_repo)] + args,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout

print("--- Current commit ---")
print(
    run_git([
        "log", "-1",
        "--format=%H%n%ad%n%s",
        "--date=iso"
    ])
)

print("--- Recent commits touching phase_inference_model.py ---")
print(
    run_git([
        "log",
        "--follow",
        "--format=%h  %ad  %s",
        "--date=short",
        "-20",
        "--",
        "velocycle/phase_inference_model.py",
    ])
)

print("--- Tags ---")
tags = run_git(["tag", "--list"])
print(tags if tags.strip() else "(no tags)")
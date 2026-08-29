# ============================================
# Cell 59. Inspect exact count_factor
# construction in phase preprocessing
# ============================================

from pathlib import Path

pyfile = Path(
    "/home/featurize/work/project1/"
    "velocycle_91123be/velocycle/"
    "preprocessing.py"
)

lines = pyfile.read_text(
    errors="ignore"
).splitlines()

# Print the phase-preprocessing block with
# enough surrounding context
for start, end in [
    (95, 205),
]:
    print(
        f"\n{'=' * 80}\n"
        f"{pyfile.name}: lines {start}-{end}\n"
        f"{'=' * 80}"
    )

    for i in range(start, end + 1):
        if i <= len(lines):
            print(
                f"{i:4d}: {lines[i - 1]}"
            )
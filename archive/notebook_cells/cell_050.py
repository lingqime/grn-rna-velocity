# ============================================
# Cell 50. Inspect exact VeloCycle kinetic
# parameterization in the velocity model
# ============================================

from pathlib import Path

# Historical VeloCycle source that matches
# the successfully loaded official pickle
src_root = Path(
    "/home/featurize/work/project1/"
    "velocycle_91123be/velocycle"
)

keywords = [
    "log_betas",
    "log_gammas",
    "beta",
    "gamma",
    "ElogU",
    "ElogS",
    "count_factor",
    "velocity_coef",
    "νω",
    "omega",
]

for pyfile in src_root.rglob("*.py"):

    try:
        lines = pyfile.read_text().splitlines()
    except Exception:
        continue

    hit_idx = []

    for i, line in enumerate(lines):
        if any(k in line for k in keywords):
            hit_idx.append(i)

    if not hit_idx:
        continue

    print("\n" + "=" * 80)
    print(pyfile.relative_to(src_root))
    print("=" * 80)

    # Merge nearby hits into context blocks
    blocks = []

    for i in hit_idx:
        start = max(0, i - 5)
        end = min(len(lines), i + 8)

        if blocks and start <= blocks[-1][1]:
            blocks[-1] = (
                blocks[-1][0],
                max(blocks[-1][1], end)
            )
        else:
            blocks.append((start, end))

    for start, end in blocks:
        print(
            f"\n--- lines {start + 1}-{end} ---"
        )

        for j in range(start, end):
            print(
                f"{j + 1:4d}: {lines[j]}"
            )
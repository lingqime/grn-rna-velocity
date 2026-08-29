# ============================================
# Cell 60. Inspect the exact Fourier basis
# zeta(phi) and its derivative
# ============================================

from pathlib import Path

src_dir = Path(
    "/home/featurize/work/project1/"
    "velocycle_91123be/velocycle"
)

search_terms = [
    "ζ",
    "zeta",
    "fourier",
    "cos(",
    "sin(",
    "ζ_dϕ",
]

for pyfile in sorted(src_dir.glob("*.py")):

    lines = pyfile.read_text(
        errors="ignore"
    ).splitlines()

    hits = []

    for i, line in enumerate(lines, start=1):
        if any(term in line for term in search_terms):
            hits.append(i)

    if not hits:
        continue

    # Print compact neighborhoods around relevant hits
    printed = set()

    for hit in hits:

        start = max(1, hit - 4)
        end = min(len(lines), hit + 8)

        key = (start, end)

        if key in printed:
            continue

        printed.add(key)

        print("\n" + "=" * 80)
        print(
            f"{pyfile.name}: "
            f"lines {start}-{end}"
        )
        print("=" * 80)

        for j in range(start, end + 1):
            print(
                f"{j:5d}: {lines[j - 1]}"
            )
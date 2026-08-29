# ============================================
# Cell 57. Locate where VeloCycle stores
# Fourier/cycle coefficients in fitted objects
# ============================================

from pathlib import Path

src_dir = Path(
    "/home/featurize/work/project1/"
    "velocycle_91123be/velocycle"
)

patterns = [
    "self.cycle_pyro",
    "cycle_pyro =",
    "cycle_pyro=",
    "self.cycle",
    "cycle_obj",
    ".means",
    "ν",
]

for pyfile in sorted(src_dir.glob("*.py")):

    text = pyfile.read_text(
        errors="ignore"
    )

    hits = []

    for lineno, line in enumerate(
        text.splitlines(),
        start=1
    ):
        if any(p in line for p in patterns):
            hits.append(
                (lineno, line)
            )

    if hits:
        print(
            "\n" + "=" * 80
        )
        print(pyfile.name)
        print("=" * 80)

        for lineno, line in hits:
            print(
                f"{lineno:5d}: {line}"
            )
# ============================================
# Cell 51. Inspect the exact preprocessing /
# velocity-fit call in the official Fig. 7
# 986-condition notebook
# ============================================

import json
from pathlib import Path

nb_path = Path(
    "/home/featurize/work/project1/ipynb_notebooks/"
    "Fig7_Replogle_Saunders_PerturbRPE1_"
    "MedGene_986conditions.ipynb"
)

with open(nb_path, "r") as f:
    nb = json.load(f)

keywords = [
    "prep_velocity",
    "velocity_fit",
    "velocity_inference",
    "normalize=",
    "count_factor",
    "S_sz",
    "U_sz",
    "data_to_fit",
]

for ci, cell in enumerate(nb["cells"]):

    if cell.get("cell_type") != "code":
        continue

    src = "".join(cell.get("source", []))

    if any(k in src for k in keywords):
        print("\n" + "=" * 80)
        print(f"CODE CELL {ci}")
        print("=" * 80)
        print(src)
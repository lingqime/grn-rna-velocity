# ============================================
# Cell 38. Inspect VeloCycle normalization
# before rebuilding trajectories
# ============================================

print("Full H5AD layers:")
print(list(adata.layers.keys()))

print("\nMedGene H5AD layers:")
print(list(adata_med.layers.keys()))

print("\nMedGene obs normalization-related columns:")
print([
    c for c in adata_med.obs.columns
    if (
        "count" in c.lower()
        or "size" in c.lower()
        or "scale" in c.lower()
    )
])

# Search VeloCycle source for S_sz / U_sz construction
print("\n========== VeloCycle source occurrences ==========")

for pyfile in velocycle_legacy.rglob("*.py"):
    try:
        lines = pyfile.read_text().splitlines()
    except Exception:
        continue

    hits = []

    for i, line in enumerate(lines, start=1):
        if (
            "S_sz" in line
            or "U_sz" in line
            or "n_scounts" in line
            or "n_ucounts" in line
        ):
            hits.append((i, line.strip()))

    if hits:
        print(f"\n--- {pyfile.relative_to(velocycle_legacy)} ---")
        for i, line in hits:
            print(f"{i:4d}: {line}")
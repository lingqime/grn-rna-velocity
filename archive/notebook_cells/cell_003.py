# ============================================
# Cell 4. First perturbation QC: cell count
# ============================================

min_cells_per_perturbation = 150

# Exclude control when defining perturbations
perturbation_counts = condition_counts.drop(
    labels=["non-targeting"],
    errors="ignore"
)

print("Number of perturbation targets:", len(perturbation_counts))

for threshold in [50, 75, 100, 150, 200, 300]:
    n = (perturbation_counts >= threshold).sum()
    print(f"Perturbations with >= {threshold:3d} cells: {n}")

# First QC selection
selected_perturbations = perturbation_counts[
    perturbation_counts >= min_cells_per_perturbation
].index.tolist()

print(
    "\nSelected perturbations (>=150 cells):",
    len(selected_perturbations)
)

print("\nFirst 20:")
print(selected_perturbations[:20])
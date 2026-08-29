# ============================================
# Cell 9. Match official phase metadata to full AnnData
# ============================================

# Use cell ID as index
phase_df = phase_df.set_index("Unnamed: 0", drop=True)

# Match phase-metadata cells to full AnnData
matched_cells = phase_df.index.intersection(adata.obs_names)

print("Phase metadata cells:", len(phase_df))
print("Matched to full adata:", len(matched_cells))
print("Unmatched:", len(phase_df) - len(matched_cells))

# Matched metadata only
phase_matched = phase_df.loc[matched_cells].copy()

print("\nConditions represented in matched phase metadata:")
print(phase_matched["gene"].nunique())

print("\nTop 15 conditions:")
print(phase_matched["gene"].value_counts().head(15))

# Specifically inspect non-targeting control
n_nt_phase = (phase_matched["gene"] == "non-targeting").sum()
n_nt_full = (adata.obs["gene"].astype(str) == "non-targeting").sum()

print("\nNon-targeting:")
print("  phase metadata:", n_nt_phase)
print("  full adata:", n_nt_full)
print("  coverage:", n_nt_phase / n_nt_full)

# How many of our 165 retained perturbations have official phase labels?
phase_conditions = set(phase_matched["gene"].astype(str))

overlap_perturbations = [
    g for g in final_perturbations
    if g in phase_conditions
]

print("\nOur retained perturbations:", len(final_perturbations))
print("With official phase labels:", len(overlap_perturbations))
print("Names:", overlap_perturbations)
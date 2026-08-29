# ============================================
# Cell 5. Second perturbation QC:
# control expression of perturbation targets
# ============================================

# Control cells
control_mask = (adata.obs["gene"].astype(str).values == "non-targeting")

# Map gene names to var indices
gene_to_idx = pd.Series(
    np.arange(adata.n_vars),
    index=adata.var_names
)

# Keep only selected perturbation targets that are measured genes
selected_measured = [
    g for g in selected_perturbations
    if g in gene_to_idx.index
]

print("Selected perturbations:", len(selected_perturbations))
print("Measured in dataset:", len(selected_measured))

selected_idx = gene_to_idx.loc[selected_measured].values

# Control mean expression
S_control_mean = np.asarray(
    S[control_mask][:, selected_idx].mean(axis=0)
).ravel()

U_control_mean = np.asarray(
    U[control_mask][:, selected_idx].mean(axis=0)
).ravel()

control_expression_qc = pd.DataFrame({
    "gene": selected_measured,
    "mean_spliced_control": S_control_mean,
    "mean_unspliced_control": U_control_mean,
})

# Expression thresholds
min_mean_spliced = 0.1
min_mean_unspliced = 0.02

keep_mask = (
    (control_expression_qc["mean_spliced_control"] >= min_mean_spliced)
    &
    (control_expression_qc["mean_unspliced_control"] >= min_mean_unspliced)
)

filtered_perturbations = (
    control_expression_qc.loc[keep_mask, "gene"]
    .tolist()
)

print("\nPassed control-expression QC:", len(filtered_perturbations))

print("\nSummary:")
print(
    control_expression_qc[
        ["mean_spliced_control", "mean_unspliced_control"]
    ].describe()
)

print("\nFirst 20 retained:")
print(filtered_perturbations[:20])
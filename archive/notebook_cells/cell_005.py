# ============================================
# Cell 6. Verify target suppression
# under its own perturbation
# ============================================

suppression_results = []

all_conditions = adata.obs["gene"].astype(str).values

for target in filtered_perturbations:

    # Cells under perturbation of this target
    perturb_mask = (all_conditions == target)

    # Target gene column
    j = gene_to_idx[target]

    # Mean target expression under its own perturbation
    s_perturb = float(S[perturb_mask, j].mean())
    u_perturb = float(U[perturb_mask, j].mean())

    # Control means already computed above
    row = control_expression_qc[
        control_expression_qc["gene"] == target
    ].iloc[0]

    s_control = row["mean_spliced_control"]
    u_control = row["mean_unspliced_control"]

    suppression_results.append({
        "gene": target,
        "n_cells": int(perturb_mask.sum()),
        "S_control": s_control,
        "S_perturb": s_perturb,
        "U_control": u_control,
        "U_perturb": u_perturb,
        "S_ratio": s_perturb / s_control,
        "U_ratio": u_perturb / u_control,
    })

suppression_df = pd.DataFrame(suppression_results)

print("Perturbations checked:", len(suppression_df))

print(
    "Targets with S_perturb == 0:",
    (suppression_df["S_perturb"] == 0).sum(),
    "/",
    len(suppression_df)
)

print(
    "Targets with U_perturb == 0:",
    (suppression_df["U_perturb"] == 0).sum(),
    "/",
    len(suppression_df)
)

print("\nMedian ratios:")
print("S perturb/control:", suppression_df["S_ratio"].median())
print("U perturb/control:", suppression_df["U_ratio"].median())

print("\nLargest S_perturb values:")
print(
    suppression_df.sort_values(
        "S_perturb",
        ascending=False
    ).head(10)
)
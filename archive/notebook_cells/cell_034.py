# ============================================
# Cell 34. Restore final perturbations
# and finish speed QC
# ============================================

final_conditions_qc = ckpt["final_conditions_qc"].astype(str).tolist()
regulator_genes_qc = ckpt["regulator_genes_qc"].astype(str).tolist()

final_perturbations_qc = [
    x for x in final_conditions_qc
    if x != "non-targeting"
]

print("Final conditions:", len(final_conditions_qc))
print("Final perturbations:", len(final_perturbations_qc))
print("Regulator genes:", len(regulator_genes_qc))

print(
    "Perturbations == regulator genes:",
    final_perturbations_qc == regulator_genes_qc
)

# Continue speed QC
missing = [
    g for g in final_perturbations_qc
    if g not in speed_series.index
]

speed_qc = speed_series.loc[final_perturbations_qc]

print("\nMissing from VeloCycle speed:", len(missing))
if missing:
    print(missing)

print("\nSelected 153 speed summary:")
print(
    speed_qc.describe(
        percentiles=[
            0.01, 0.05, 0.10, 0.25,
            0.50, 0.75, 0.90, 0.95, 0.99
        ]
    )
)

print("\nSign counts, selected perturbations:")
print("positive:", int((speed_qc > 0).sum()))
print("zero:", int((speed_qc == 0).sum()))
print("negative:", int((speed_qc < 0).sum()))

print("\nNegative selected conditions:")
print(speed_qc[speed_qc < 0].sort_values())
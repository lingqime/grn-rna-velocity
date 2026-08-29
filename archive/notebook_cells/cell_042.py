# ============================================
# Cell 42. Freeze final condition set under
# official VeloCycle phase
# ============================================

min_usable_bins = 7

final_conditions_vc = [
    q for q in final_conditions_qc
    if coverage_vc.loc[q, "usable_bins_ge5"] >= min_usable_bins
]

final_perturbations_vc = [
    q for q in final_conditions_vc
    if q != "non-targeting"
]

removed_by_official_phase = [
    q for q in final_conditions_qc
    if q not in final_conditions_vc
]

print("Final conditions:", len(final_conditions_vc))
print("Final perturbations:", len(final_perturbations_vc))

print("\nRemoved after official-phase coverage QC:")
print(removed_by_official_phase)

# Regulator set follows the retained perturbation targets
regulator_genes_vc = final_perturbations_vc.copy()

print("\nRegulator genes:", len(regulator_genes_vc))
print(
    "Perturbations == regulator genes:",
    final_perturbations_vc == regulator_genes_vc
)

# Speed QC for the frozen set
speed_final = speed_series.loc[final_conditions_vc]

print("\nFinal-condition speed summary:")
print(speed_final.describe())

print("\nMinimum-speed conditions:")
print(speed_final.sort_values().head(10))

print(
    "\nAll final speeds positive:",
    bool((speed_final > 0).all())
)

# Relative speed with NT = 1
omega_nt = float(speed_series.loc["non-targeting"])
rho_final = speed_final / omega_nt

print("\nNT speed:", omega_nt)

print("\nRelative-speed summary (NT = 1):")
print(rho_final.describe())
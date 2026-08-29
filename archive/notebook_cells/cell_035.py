# ============================================
# Cell 35. Check kinetics coverage for the
# final 153 regulator / response genes
# ============================================

gene_names_vc = vc["gene_names"].astype(str)
log_betas_vc = vc["log_betas"].astype(float)
log_gammas_vc = vc["log_gammas"].astype(float)

vc_gene_set = set(gene_names_vc)

covered_genes = [
    g for g in regulator_genes_qc
    if g in vc_gene_set
]

missing_genes = [
    g for g in regulator_genes_qc
    if g not in vc_gene_set
]

print("Final regulator genes:", len(regulator_genes_qc))
print("Covered by VeloCycle kinetics:", len(covered_genes))
print("Missing kinetics:", len(missing_genes))

if missing_genes:
    print("\nMissing genes:")
    print(missing_genes)

# Map gene -> kinetic parameters
beta_series = pd.Series(
    np.exp(log_betas_vc),
    index=gene_names_vc,
    name="beta"
)

gamma_series = pd.Series(
    np.exp(log_gammas_vc),
    index=gene_names_vc,
    name="gamma"
)

if covered_genes:
    print("\nBeta summary for covered genes:")
    print(beta_series.loc[covered_genes].describe())

    print("\nGamma summary for covered genes:")
    print(gamma_series.loc[covered_genes].describe())
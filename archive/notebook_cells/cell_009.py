# ============================================
# Cell 10. Inspect VeloCycle MedGeneSet
# ============================================

adata_med = ad.read_h5ad(med_h5ad)

print("MedGeneSet shape:", adata_med.shape)

print("\nLayers:")
print(list(adata_med.layers.keys()))

print("\nNumber of genes:")
print(adata_med.n_vars)

print("\nFirst 20 genes:")
print(adata_med.var_names[:20].tolist())

# Check whether all 120 genes exist in the full dataset
med_genes = adata_med.var_names.tolist()

missing_from_full = [
    g for g in med_genes
    if g not in adata.var_names
]

print("\nMedGene genes:", len(med_genes))
print("Missing from full adata:", len(missing_from_full))

if missing_from_full:
    print("Missing genes:", missing_from_full)
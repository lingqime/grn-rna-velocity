# ============================================
# Cell 36. Check the 426 VeloCycle kinetic genes
# against the full RPE1 H5AD
# ============================================

vc_genes = gene_names_vc.tolist()

adata_gene_set = set(adata.var_names.astype(str))

vc_in_adata = [
    g for g in vc_genes
    if g in adata_gene_set
]

vc_missing_adata = [
    g for g in vc_genes
    if g not in adata_gene_set
]

print("VeloCycle kinetic genes:", len(vc_genes))
print("Present in full H5AD:", len(vc_in_adata))
print("Missing from full H5AD:", len(vc_missing_adata))

if vc_missing_adata:
    print("\nMissing genes:")
    print(vc_missing_adata)

# Also inspect overlap structure
reg_set = set(regulator_genes_qc)
vc_set = set(vc_genes)

overlap = [
    g for g in regulator_genes_qc
    if g in vc_set
]

print("\nRegulator set R:", len(regulator_genes_qc))
print("Candidate response set G:", len(vc_genes))
print("R ∩ G:", len(overlap))

print("\nR ∩ G genes:")
print(overlap)
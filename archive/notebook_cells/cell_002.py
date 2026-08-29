# ============================================
# Cell 3. Check spliced/unspliced and conditions
# ============================================

S = adata.layers["spliced"]
U = adata.layers["unspliced"]

print("Spliced:")
print("  type:", type(S))
print("  shape:", S.shape)
print("  sparse:", sp.issparse(S))
print("  dtype:", S.dtype)

print("\nUnspliced:")
print("  type:", type(U))
print("  shape:", U.shape)
print("  sparse:", sp.issparse(U))
print("  dtype:", U.dtype)

# Perturbation labels
condition_counts = (
    adata.obs["gene"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nNumber of perturbation labels:", len(condition_counts))

print("\nTop 10 conditions:")
print(condition_counts.head(10))

print("\nnon-targeting cells:")
print(condition_counts.get("non-targeting", 0))
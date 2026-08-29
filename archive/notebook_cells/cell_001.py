# ============================================
# Cell 2. Load full RPE1 AnnData
# ============================================

import anndata as ad

adata = ad.read_h5ad(full_h5ad)

print("adata shape:", adata.shape)

print("\nobs columns:")
print(list(adata.obs.columns))

print("\nvar columns:")
print(list(adata.var.columns))

print("\nlayers:")
print(list(adata.layers.keys()))

print("\nMemory after loading:")
print(
    "Available RAM (GB):",
    round(psutil.virtual_memory().available / 1024**3, 2)
)
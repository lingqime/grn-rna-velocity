# ============================================
# Cell 62b. Restore fixed 10-bin phase grid
# ============================================

import numpy as np

n_bins = 10

bin_edges = np.linspace(
    0.0,
    1.0,
    n_bins + 1
)

bin_centers = (
    bin_edges[:-1]
    + bin_edges[1:]
) / 2.0

print("bin_edges:")
print(bin_edges)

print("\nbin_centers:")
print(bin_centers)

print(
    "\nNumber of bins:",
    len(bin_centers)
)
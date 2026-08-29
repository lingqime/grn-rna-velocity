# Expected frozen result files

The figure script expects the following files if they are recovered from the
original analysis runtime:

- `replogle_final_exact_integral_grn_v2.npz`
- `replogle_primary_exact_integral_grn.npz`
- `replogle_manuscript_validation.csv`

The final manuscript-level master checkpoint produced by the notebook was:

- `replogle_manuscript_final_checkpoint.npz`

These files are not bundled here because they were not included with the
uploaded notebook. If recovered, place them in this directory locally. Large
NPZ files are ignored by `.gitignore`; small final CSV summaries may be
committed intentionally.

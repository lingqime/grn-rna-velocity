# RPE1 analysis workflow

The numbered scripts are a cleaned, stage-oriented transcription of the final
analysis path in `archive/Replogle_cloud.ipynb`.

## Canonical order

1. `01_prepare_rpe1.py` — load RPE1 data, apply the recorded QC steps, align
   VeloCycle metadata, and restore the historical intermediate condition panel.
2. `02_build_trajectories.py` — build the final 10-bin response and regulator
   trajectories.
3. `03_build_integral_design.py` — construct the exact integral design `X_final`,
   response `Y_final`, and gene-specific valid-row masks.
4. `04_fit_grn.py` — calibrate the exact-integral L1 penalty and fit the corrected
   primary GRN, including self-predictor exclusion for the 12 response/regulator
   overlap genes.
5. `05_identifiability.py` — compute full-design rank/conditioning and randomized
   perturbation-order paths.
6. `06_stability.py` — 100 perturbation-condition subsampling refits.
7. `07_heldout_validation.py` — five-fold condition-held-out validation against
   the phase-only baseline.
8. `08_make_figures.py` — regenerate the three manuscript figures from frozen
   NPZ/CSV outputs only.
9. `diagnostics.py` — secondary diagnostics used to interpret held-out results.

## Important execution note

The original analysis was developed as one sequential notebook. The numbered
files intentionally preserve that sequence and therefore share the namespace
created by earlier stages. They are primarily the readable/auditable source
representation of the paper analysis, not eight independent command-line
programs.

The original notebook remains the exact execution provenance record. A future
packaging step could serialize stage boundaries into standalone checkpoints,
but this repository does not pretend that such a refactor was part of the
original analysis.

## External inputs

Machine-specific paths have been removed from the canonical scripts. Required
large files are expected under `data/raw/` and `data/processed/`; see
`data/README.md`.

The legacy VeloCycle-pickle extraction helper is archived rather than included
in the canonical workflow because it depends on the historical VeloCycle
environment.

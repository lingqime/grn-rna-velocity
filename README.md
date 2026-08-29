# Decoding Gene Regulatory Networks from Single-Cell RNA Velocity

Code and technical provenance for the perturbation-resolved RPE1 analysis used
in the manuscript *Decoding Gene Regulatory Networks from Single-Cell RNA
Velocity*.


## Reproducibility status

This repository preserves the complete original analysis notebook and a cleaned,
stage-oriented representation of the final RPE1 workflow. The final numerical
analysis and manuscript-level settings are recoverable from the notebook.
However, a fresh raw-data-to-final-network rerun still requires one historical
intermediate checkpoint because the exact code for the 165 -> 153 perturbation
filter is absent from the recovered notebook. This limitation is documented
explicitly in `PROVENANCE.md`; no missing filtering rule has been guessed.

The numbered RPE1 scripts are therefore intended as readable and auditable
research code matching the final analysis, while
`archive/Replogle_cloud.ipynb` remains the exact execution-provenance record.

## What this repository contains

The uploaded analysis consisted of one 134-cell Jupyter notebook,
`Replogle_cloud.ipynb`. This repository preserves that notebook unchanged under
`archive/` and reorganizes the final analysis path into stage-oriented Python
scripts.

The final RPE1 analysis has three empirical targets:

1. perturbation-assisted empirical identifiability and conditioning;
2. perturbation-condition subsampling stability of the sparse integral GRN;
3. five-fold condition-held-out conditional equation prediction relative to a
   cell-cycle phase baseline.

## Repository structure

```text
grn-rna-velocity/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── rpe1.yaml
├── src/
│   ├── integral_lasso.py
│   ├── metrics.py
│   └── phase.py
├── experiments/
│   └── rpe1/
│       ├── 01_prepare_rpe1.py
│       ├── 02_build_trajectories.py
│       ├── 03_build_integral_design.py
│       ├── 04_fit_grn.py
│       ├── 05_identifiability.py
│       ├── 06_stability.py
│       ├── 07_heldout_validation.py
│       ├── 08_make_figures.py
│       └── diagnostics.py
├── data/
│   └── README.md
├── results/
│   └── rpe1/
├── figures/
│   └── rpe1/
├── supplement/
│   └── rpe1_real_data_siam_supplement.tex
└── archive/
    ├── Replogle_cloud.ipynb
    ├── Replogle_cloud_export.py
    ├── CELL_MAP.md
    └── notebook_cells/
```

## Status of the extraction

The stage scripts are faithful, cleaned extracts of the code that produced the
final analysis, with notebook cell numbers preserved in comments. Machine-
specific paths have been removed from the canonical scripts. Because the
original computation was a sequential notebook, the numbered stage files share
the namespace created by earlier stages; see `experiments/rpe1/README.md`.

The original notebook is still the canonical provenance record. A completely
fresh raw-data rerun has one known provenance gap: the notebook restores a
154-condition intermediate panel from the historical
`replogle_grn_checkpoint.npz`. The notebook contains the exact first two QC
steps (985 -> 286 -> 165 perturbations) but not the complete code/bytes for the
subsequent 165 -> 153 intermediate filtering step. See `data/README.md`.

No missing preprocessing rule has been guessed or silently reconstructed.

## Frozen final analysis specification

- 151 perturbations + non-targeting control;
- 151 candidate regulators;
- 426 VeloCycle response genes;
- 10 circular phase bins;
- minimum 5 cells per condition-phase bin;
- 1,214 usable adjacent-phase integral equations;
- exact-integral sparse estimator with unpenalized basal/dt coefficient;
- FWL residualization, no ordinary mean-centering, RMS scaling;
- global Lasso alpha `0.03665241237079626`;
- 100 randomized perturbation-order identifiability repetitions;
- 100 stability repetitions with 80% perturbation-condition subsampling;
- five-fold perturbation-condition held-out validation;
- non-targeting control always retained in training/subsamples.

The parameter file `configs/rpe1.yaml` records these settings and the fixed
31-gene alpha-calibration panel.

## Upstream data and software

The real-data analysis uses RPE1 Perturb-seq/VeloCycle materials released with:

- Lederer et al., *Statistical inference with a manifold-constrained RNA
  velocity model uncovers cell cycle speed modulations*, Nature Methods 21,
  2271--2286 (2024), DOI: https://doi.org/10.1038/s41592-024-02471-8
- VeloCycle source: https://github.com/lamanno-epfl/velocycle
- VeloCycle publication materials: https://doi.org/10.5281/zenodo.12517650

See `data/README.md` for the local file layout and data-provenance notes.

## Installation

A minimal reconstruction/figure environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The historical VeloCycle environment used to deserialize the official upstream
result is not recreated here. The notebook's environment-recovery cells are
retained under `archive/`.

## Data

Large H5AD and VeloCycle pickle files are not stored in Git. Canonical scripts
use repository-local `data/raw/` and `data/processed/` paths. See
`data/README.md` for expected inputs and the known intermediate-checkpoint
provenance limitation. Additional recovered details are recorded in
`PROVENANCE.md`.

## Reproducing figures from frozen outputs

If the frozen NPZ/CSV outputs are recovered, place them in `results/rpe1/` and
run:

```bash
python experiments/rpe1/08_make_figures.py
```

This writes the three RPE1 manuscript figures to `figures/rpe1/` without
refitting the model.

## Supplement

The SIAM supplementary-material source is included in `supplement/` and records
the real-data construction, estimator, identifiability, stability, validation,
and reproducibility details.

## Before changing this repository from private to public

Two metadata decisions are intentionally left to the repository owner rather
than guessed during code packaging:

- choose and add the software license;
- add the final manuscript citation/author metadata (for example in
  `CITATION.cff`) once the author list and manuscript identifier are confirmed.

The unresolved 165 -> 153 condition-selection provenance gap should remain
disclosed unless the historical checkpoint-generation code is recovered.

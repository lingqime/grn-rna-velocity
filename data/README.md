# RPE1 external data

Large biological data and historical binary checkpoints are intentionally not
committed to GitHub.

## Upstream sources

The RPE1 Perturb-seq/VeloCycle analysis used here is based on the VeloCycle
publication and its released analysis materials:

- VeloCycle article: https://doi.org/10.1038/s41592-024-02471-8
- VeloCycle source code: https://github.com/lamanno-epfl/velocycle
- VeloCycle publication materials: https://doi.org/10.5281/zenodo.12517650

The VeloCycle paper describes the genome-wide RPE1 Perturb-seq application,
including filtering perturbed cells to retain complete target knockdown and
condition-specific velocity learning.

## Local file layout

After obtaining the upstream data/materials, the cleaned analysis scripts expect
the following local layout:

```text
data/
├── raw/
│   ├── Replogle_Saunders_PerturbRPE1_All_75cutoff_QC.h5ad
│   ├── Replogle_Saunders_PerturbRPE1_All_75cutoff_QC_MedGeneSet_Filtered.h5ad
│   └── PerturbRPE1_NT_CCKO_MedGeneSet_cycle_cell_phase_1harm_metadata.csv.gz
└── processed/
    ├── velocycle_986_official_extract.npz
    ├── velocycle_986_latent_extract.npz
    └── replogle_grn_checkpoint.npz
```

The exact filenames above are the filenames used by the recovered analysis
notebook. Upstream archives may organize or name files differently; do not
rename or overwrite upstream source files without preserving their provenance.

## What the recovered notebook establishes exactly

The notebook records two explicit perturbation-QC filters:

1. at least 150 cells per perturbation, reducing 985 measured perturbation
   targets to 286;
2. non-targeting mean spliced expression >= 0.1 and mean unspliced expression
   >= 0.02 for the target gene, reducing 286 to 165 perturbations.

The notebook then restores a 153-perturbation intermediate panel from
`replogle_grn_checkpoint.npz`. The notebook does not contain the bytes or the
complete earlier code that produced the 165 -> 153 reduction, so this remains
the one unresolved condition-selection provenance gap.

The final VeloCycle phase-coverage filter removes `NPLOC4` and `TRAPPC11`,
leaving 151 perturbations plus the non-targeting control.

## VeloCycle compact extracts

`velocycle_986_official_extract.npz` stores the official cell phase,
condition-specific speed, and log-beta/log-gamma arrays used downstream.

`velocycle_986_latent_extract.npz` stores the Fourier coefficients and related
latent quantities needed to reconstruct the 10-bin VeloCycle phase template.

The historical deserialization helper for the original VeloCycle pickle is kept
under `archive/legacy_velocycle_latent_extract.py`. It is not a canonical public
entry point because it relied on a legacy environment.

## Files intentionally excluded from Git

Do not commit H5AD files, the original large VeloCycle pickle, or large NPZ
checkpoints directly to GitHub. If frozen numerical outputs are released, a
versioned archival service such as Zenodo is preferable for the large files.

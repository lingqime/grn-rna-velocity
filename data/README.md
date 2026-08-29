# RPE1 data inputs

Large biological data files are intentionally not committed to GitHub.

The original notebook used the following upstream inputs:

- `Replogle_Saunders_PerturbRPE1_All_75cutoff_QC.h5ad`
- `Replogle_Saunders_PerturbRPE1_All_75cutoff_QC_MedGeneSet_Filtered.h5ad`
- `PerturbRPE1_NT_CCKO_MedGeneSet_cycle_cell_phase_1harm_metadata.csv.gz`
- the official 986-condition VeloCycle LargeGene result pickle
- compact VeloCycle extracts produced from that pickle
- `replogle_grn_checkpoint.npz`, an intermediate historical checkpoint that
  stores the 154-condition pre-final-phase-QC panel.

## Important provenance limitation

The uploaded notebook contains the exact first two perturbation QC thresholds:

1. at least 150 cells per perturbation;
2. non-targeting mean spliced expression >= 0.1 and mean unspliced expression >= 0.02.

These reduce 985 perturbation targets to 286 and then 165 targets. The notebook
then restores a 153-perturbation intermediate panel from
`replogle_grn_checkpoint.npz`. The bytes of that historical checkpoint are not
contained in the notebook, so the exact 165 -> 153 intermediate filtering step
cannot be reconstructed from this repository alone unless that checkpoint or
the earlier preprocessing code is recovered.

Using the official VeloCycle phase, the final phase-coverage filter removes
`NPLOC4` and `TRAPPC11`, leaving 151 perturbations plus non-targeting control.

Do not commit the large H5AD or VeloCycle pickle files directly to GitHub.

# Analysis provenance and known limitations

This repository was reconstructed from the complete analysis notebook
`archive/Replogle_cloud.ipynb`; no missing numerical-analysis branch was
invented during packaging.

Recovered exactly from the notebook:

- perturbation QC thresholds: 150 cells, NT mean spliced >= 0.1, NT mean
  unspliced >= 0.02;
- final 151 perturbations + non-targeting control after VeloCycle phase coverage;
- 426 response genes and 151 candidate regulators;
- 10 phase bins and >=5 cells/bin usability rule;
- NT-relative response amplitude
  `median_b[log(S_q,b + 0.1) - log(S_NT,b + 0.1)]` over bins usable in both
  condition and NT;
- exact-integral FWL/RMS Lasso estimator;
- fixed 31-gene alpha calibration panel;
- alpha grid `logspace(-4, 0, 40)`, five perturbation-condition folds,
  `shuffle=True`, `random_state=2026`, NT always in training;
- alpha selected by minimum mean relative held-out MSE with equal weight across
  the 31 calibration genes;
- final alpha `0.03665241237079626`;
- corrected self-predictor exclusion for the 12 response/regulator overlap
  genes;
- 100 randomized identifiability paths, 100 stability subsamples, and five-fold
  held-out validation.

One condition-selection gap remains: the notebook contains the 985 -> 286 ->
165 filters and then loads a historical checkpoint containing 153
perturbations. The exact 165 -> 153 filter is not represented in the notebook.
No replacement rule has been guessed.

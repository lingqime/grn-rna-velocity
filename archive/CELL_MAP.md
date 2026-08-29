# Notebook cell map

`archive/Replogle_cloud.ipynb` is the immutable provenance record. The main
experiment scripts are organized extractions of the final analysis path.

| Notebook indices | Role | Repository location |
|---|---|---|
| 0--9 | Initial data inspection and first two perturbation QC steps | `01_prepare_rpe1.py` |
| 10--31 | Historical VeloCycle environment/source recovery and compact extract creation | archive only |
| 32--44, 48, 54 | VeloCycle alignment, final condition/response preparation | `01_prepare_rpe1.py` |
| 45--59 | Intermediate trajectory/kinetic diagnostics and superseded constructions | archive only |
| 60--73 | Final VeloCycle latent/response/regulator trajectory construction | `02_build_trajectories.py` |
| 74--75, 79 | Final exact integral X/Y and direct-intervention row masks | `03_build_integral_design.py` |
| 76--101 | Earlier estimators, TACC3 pilots, phase-residual diagnostics, and superseded validation branches | archive only |
| 98, 102--103, 115--117, 119 | Exact-integral alpha calibration, primary fit, self-excluded correction | `04_fit_grn.py` |
| 105--107 | Correct identifiability/conditioning analysis | `05_identifiability.py` |
| 108 | Pilot stability | archive only |
| 109, 118--119 | Final stability and corrected primary network | `06_stability.py` |
| 110--116 | Edge/regulator/self-edge diagnostics | archive only |
| 121--124 | Primary five-fold conditional held-out validation | `07_heldout_validation.py` |
| 125--128 | Interpretation diagnostics | `diagnostics.py` |
| 129--130 | Final manuscript summary/checkpoint writers | archive only |

## Why some cells are archive-only

The notebook records the actual exploratory history. Several earlier estimators
were intentionally superseded by the final exact-integral formulation, and the
VeloCycle source/environment recovery cells were operational recovery work
rather than part of the GRN method. They are retained for auditability but are
not presented as the canonical paper workflow.

# Public release checklist

The code/content cleanup is complete enough for a private manuscript repository.
Before changing the GitHub visibility to **Public**, confirm the following.

- [x] Canonical scripts contain no machine-specific analysis-runtime paths.
- [x] Large H5AD, pickle, NPY, and NPZ inputs/outputs are excluded by `.gitignore`.
- [x] The complete original notebook is retained under `archive/`.
- [x] The known 165 -> 153 perturbation-selection provenance gap is disclosed.
- [x] Upstream VeloCycle article, source-code, and Zenodo links are documented.
- [x] Final exact-integral alpha and calibration settings are recorded.
- [x] Public Python files pass syntax compilation.
- [x] MIT software license added as `LICENSE`.
- [x] Manuscript author list confirmed and `CITATION.cff` added.
- [ ] If large frozen result checkpoints will be released, archive them
      separately (for example on Zenodo) and add their persistent link.
- [ ] Perform one final visual check of the GitHub README after the last push.

The core repository metadata required for public release are now present. Any remaining unchecked items are optional or depend on later archival decisions.

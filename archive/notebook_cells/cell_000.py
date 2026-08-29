# ============================================
# Cell 1. Imports and project paths
# ============================================

from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import anndata as ad
import h5py
import matplotlib.pyplot as plt
import psutil

# Project directory
base_dir = Path("/home/featurize/work/project1")

# Main RPE1 dataset
full_h5ad = (
    base_dir /
    "Replogle_Saunders_PerturbRPE1_All_75cutoff_QC.h5ad"
)

# 120-gene VeloCycle dataset
med_h5ad = (
    base_dir /
    "Replogle_Saunders_PerturbRPE1_All_75cutoff_QC_MedGeneSet_Filtered.h5ad"
)

# VeloCycle phase metadata
phase_metadata = (
    base_dir /
    "PerturbRPE1_NT_CCKO_MedGeneSet_cycle_cell_phase_1harm_metadata.csv.gz"
)

# Our previous compact checkpoint
checkpoint_path = base_dir / "replogle_grn_checkpoint.npz"

print("Project directory:", base_dir)
print()
print("Full H5AD:", full_h5ad.exists())
print("MedGene H5AD:", med_h5ad.exists())
print("Phase metadata:", phase_metadata.exists())
print("Checkpoint:", checkpoint_path.exists())

print()
print(
    "Available RAM (GB):",
    round(psutil.virtual_memory().available / 1024**3, 2)
)
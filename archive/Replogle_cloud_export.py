"""Linear export of all code cells from Replogle_cloud.ipynb.
Generated for archival provenance; this is not the recommended entry point.
"""


# ============================================================================
# NOTEBOOK INDEX 0: Cell 1. Imports and project paths
# ============================================================================

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


# ============================================================================
# NOTEBOOK INDEX 1: Cell 2. Load full RPE1 AnnData
# ============================================================================

# ============================================
# Cell 2. Load full RPE1 AnnData
# ============================================

import anndata as ad

adata = ad.read_h5ad(full_h5ad)

print("adata shape:", adata.shape)

print("\nobs columns:")
print(list(adata.obs.columns))

print("\nvar columns:")
print(list(adata.var.columns))

print("\nlayers:")
print(list(adata.layers.keys()))

print("\nMemory after loading:")
print(
    "Available RAM (GB):",
    round(psutil.virtual_memory().available / 1024**3, 2)
)


# ============================================================================
# NOTEBOOK INDEX 2: Cell 3. Check spliced/unspliced and conditions
# ============================================================================

# ============================================
# Cell 3. Check spliced/unspliced and conditions
# ============================================

S = adata.layers["spliced"]
U = adata.layers["unspliced"]

print("Spliced:")
print("  type:", type(S))
print("  shape:", S.shape)
print("  sparse:", sp.issparse(S))
print("  dtype:", S.dtype)

print("\nUnspliced:")
print("  type:", type(U))
print("  shape:", U.shape)
print("  sparse:", sp.issparse(U))
print("  dtype:", U.dtype)

# Perturbation labels
condition_counts = (
    adata.obs["gene"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nNumber of perturbation labels:", len(condition_counts))

print("\nTop 10 conditions:")
print(condition_counts.head(10))

print("\nnon-targeting cells:")
print(condition_counts.get("non-targeting", 0))


# ============================================================================
# NOTEBOOK INDEX 3: Cell 4. First perturbation QC: cell count
# ============================================================================

# ============================================
# Cell 4. First perturbation QC: cell count
# ============================================

min_cells_per_perturbation = 150

# Exclude control when defining perturbations
perturbation_counts = condition_counts.drop(
    labels=["non-targeting"],
    errors="ignore"
)

print("Number of perturbation targets:", len(perturbation_counts))

for threshold in [50, 75, 100, 150, 200, 300]:
    n = (perturbation_counts >= threshold).sum()
    print(f"Perturbations with >= {threshold:3d} cells: {n}")

# First QC selection
selected_perturbations = perturbation_counts[
    perturbation_counts >= min_cells_per_perturbation
].index.tolist()

print(
    "\nSelected perturbations (>=150 cells):",
    len(selected_perturbations)
)

print("\nFirst 20:")
print(selected_perturbations[:20])


# ============================================================================
# NOTEBOOK INDEX 4: Cell 5. Second perturbation QC:
# ============================================================================

# ============================================
# Cell 5. Second perturbation QC:
# control expression of perturbation targets
# ============================================

# Control cells
control_mask = (adata.obs["gene"].astype(str).values == "non-targeting")

# Map gene names to var indices
gene_to_idx = pd.Series(
    np.arange(adata.n_vars),
    index=adata.var_names
)

# Keep only selected perturbation targets that are measured genes
selected_measured = [
    g for g in selected_perturbations
    if g in gene_to_idx.index
]

print("Selected perturbations:", len(selected_perturbations))
print("Measured in dataset:", len(selected_measured))

selected_idx = gene_to_idx.loc[selected_measured].values

# Control mean expression
S_control_mean = np.asarray(
    S[control_mask][:, selected_idx].mean(axis=0)
).ravel()

U_control_mean = np.asarray(
    U[control_mask][:, selected_idx].mean(axis=0)
).ravel()

control_expression_qc = pd.DataFrame({
    "gene": selected_measured,
    "mean_spliced_control": S_control_mean,
    "mean_unspliced_control": U_control_mean,
})

# Expression thresholds
min_mean_spliced = 0.1
min_mean_unspliced = 0.02

keep_mask = (
    (control_expression_qc["mean_spliced_control"] >= min_mean_spliced)
    &
    (control_expression_qc["mean_unspliced_control"] >= min_mean_unspliced)
)

filtered_perturbations = (
    control_expression_qc.loc[keep_mask, "gene"]
    .tolist()
)

print("\nPassed control-expression QC:", len(filtered_perturbations))

print("\nSummary:")
print(
    control_expression_qc[
        ["mean_spliced_control", "mean_unspliced_control"]
    ].describe()
)

print("\nFirst 20 retained:")
print(filtered_perturbations[:20])


# ============================================================================
# NOTEBOOK INDEX 5: Cell 6. Verify target suppression
# ============================================================================

# ============================================
# Cell 6. Verify target suppression
# under its own perturbation
# ============================================

suppression_results = []

all_conditions = adata.obs["gene"].astype(str).values

for target in filtered_perturbations:

    # Cells under perturbation of this target
    perturb_mask = (all_conditions == target)

    # Target gene column
    j = gene_to_idx[target]

    # Mean target expression under its own perturbation
    s_perturb = float(S[perturb_mask, j].mean())
    u_perturb = float(U[perturb_mask, j].mean())

    # Control means already computed above
    row = control_expression_qc[
        control_expression_qc["gene"] == target
    ].iloc[0]

    s_control = row["mean_spliced_control"]
    u_control = row["mean_unspliced_control"]

    suppression_results.append({
        "gene": target,
        "n_cells": int(perturb_mask.sum()),
        "S_control": s_control,
        "S_perturb": s_perturb,
        "U_control": u_control,
        "U_perturb": u_perturb,
        "S_ratio": s_perturb / s_control,
        "U_ratio": u_perturb / u_control,
    })

suppression_df = pd.DataFrame(suppression_results)

print("Perturbations checked:", len(suppression_df))

print(
    "Targets with S_perturb == 0:",
    (suppression_df["S_perturb"] == 0).sum(),
    "/",
    len(suppression_df)
)

print(
    "Targets with U_perturb == 0:",
    (suppression_df["U_perturb"] == 0).sum(),
    "/",
    len(suppression_df)
)

print("\nMedian ratios:")
print("S perturb/control:", suppression_df["S_ratio"].median())
print("U perturb/control:", suppression_df["U_ratio"].median())

print("\nLargest S_perturb values:")
print(
    suppression_df.sort_values(
        "S_perturb",
        ascending=False
    ).head(10)
)


# ============================================================================
# NOTEBOOK INDEX 6: Cell 7. Define conditions and perturbation operators M_q
# ============================================================================

# ============================================
# Cell 7. Define conditions and perturbation operators M_q
# ============================================

# Current perturbation set after the first two QC steps
final_perturbations = filtered_perturbations.copy()

# q = 0 is the non-targeting control
final_conditions = ["non-targeting"] + final_perturbations

# For now, use the perturbation targets themselves
# as the candidate regulator set R
regulator_genes = final_perturbations.copy()

K = len(regulator_genes)
Q = len(final_conditions)

regulator_to_index = {
    gene: j for j, gene in enumerate(regulator_genes)
}

condition_to_index = {
    condition: q for q, condition in enumerate(final_conditions)
}


def make_Mq(condition):
    """
    Perturbation operator M_q for the manuscript model.

    Control:
        M_0 = I

    Target perturbation:
        zero the corresponding regulator coordinate.
    """
    M = np.eye(K)

    if condition != "non-targeting":
        j = regulator_to_index[condition]
        M[j, j] = 0.0

    return M


print("Candidate regulators K:", K)
print("Conditions including control:", Q)

# Check control
M0 = make_Mq("non-targeting")
print("\nControl M0 is identity:", np.allclose(M0, np.eye(K)))

# Check one perturbation
test_gene = "TACC3"
Mq_test = make_Mq(test_gene)
j = regulator_to_index[test_gene]

print("\nTest perturbation:", test_gene)
print("Target diagonal entry:", Mq_test[j, j])
print("Number of zero diagonal entries:", np.sum(np.diag(Mq_test) == 0))
print("All other diagonal entries = 1:",
      np.all(np.delete(np.diag(Mq_test), j) == 1))


# ============================================================================
# NOTEBOOK INDEX 7: Cell 8. Load official VeloCycle phase metadata
# ============================================================================

# ============================================
# Cell 8. Load official VeloCycle phase metadata
# ============================================

phase_df = pd.read_csv(phase_metadata)

print("phase_df shape:", phase_df.shape)
print("\nColumns:")
print(list(phase_df.columns))

print("\nFirst 5 rows:")
print(phase_df.head())

print("\nUnique cell IDs:")
print(phase_df["Unnamed: 0"].nunique())

print("\ncell_cycle_phi summary:")
print(phase_df["cell_cycle_phi"].describe())


# ============================================================================
# NOTEBOOK INDEX 8: Cell 9. Match official phase metadata to full AnnData
# ============================================================================

# ============================================
# Cell 9. Match official phase metadata to full AnnData
# ============================================

# Use cell ID as index
phase_df = phase_df.set_index("Unnamed: 0", drop=True)

# Match phase-metadata cells to full AnnData
matched_cells = phase_df.index.intersection(adata.obs_names)

print("Phase metadata cells:", len(phase_df))
print("Matched to full adata:", len(matched_cells))
print("Unmatched:", len(phase_df) - len(matched_cells))

# Matched metadata only
phase_matched = phase_df.loc[matched_cells].copy()

print("\nConditions represented in matched phase metadata:")
print(phase_matched["gene"].nunique())

print("\nTop 15 conditions:")
print(phase_matched["gene"].value_counts().head(15))

# Specifically inspect non-targeting control
n_nt_phase = (phase_matched["gene"] == "non-targeting").sum()
n_nt_full = (adata.obs["gene"].astype(str) == "non-targeting").sum()

print("\nNon-targeting:")
print("  phase metadata:", n_nt_phase)
print("  full adata:", n_nt_full)
print("  coverage:", n_nt_phase / n_nt_full)

# How many of our 165 retained perturbations have official phase labels?
phase_conditions = set(phase_matched["gene"].astype(str))

overlap_perturbations = [
    g for g in final_perturbations
    if g in phase_conditions
]

print("\nOur retained perturbations:", len(final_perturbations))
print("With official phase labels:", len(overlap_perturbations))
print("Names:", overlap_perturbations)


# ============================================================================
# NOTEBOOK INDEX 9: Cell 10. Inspect VeloCycle MedGeneSet
# ============================================================================

# ============================================
# Cell 10. Inspect VeloCycle MedGeneSet
# ============================================

adata_med = ad.read_h5ad(med_h5ad)

print("MedGeneSet shape:", adata_med.shape)

print("\nLayers:")
print(list(adata_med.layers.keys()))

print("\nNumber of genes:")
print(adata_med.n_vars)

print("\nFirst 20 genes:")
print(adata_med.var_names[:20].tolist())

# Check whether all 120 genes exist in the full dataset
med_genes = adata_med.var_names.tolist()

missing_from_full = [
    g for g in med_genes
    if g not in adata.var_names
]

print("\nMedGene genes:", len(med_genes))
print("Missing from full adata:", len(missing_from_full))

if missing_from_full:
    print("Missing genes:", missing_from_full)


# ============================================================================
# NOTEBOOK INDEX 10: Cell 11. Locate official 986-condition
# ============================================================================

# ============================================
# Cell 11. Locate official 986-condition
# VeloCycle result pickle
# ============================================

pickle_dir = base_dir / "pickle_result_outputs"

pickle_files = sorted(
    pickle_dir.glob("*986conditions*.pkl.gz")
)

print("986-condition pickle files found:", len(pickle_files))

for p in pickle_files:
    size_gb = p.stat().st_size / 1024**3
    print(f"\n{p.name}")
    print(f"  compressed size: {size_gb:.2f} GB")

print(
    "\nAvailable RAM (GB):",
    round(psutil.virtual_memory().available / 1024**3, 2)
)

print(
    "Free disk (GB):",
    round(psutil.disk_usage(base_dir).free / 1024**3, 2)
)


# ============================================================================
# NOTEBOOK INDEX 11: Cell 12. Check VeloCycle dependencies
# ============================================================================

# ============================================
# Cell 12. Check VeloCycle dependencies
# ============================================

import importlib.util
import sys

packages = [
    "torch",
    "pyro",
    "velocycle",
]

print("Python:", sys.version)
print()

for pkg in packages:
    spec = importlib.util.find_spec(pkg)

    if spec is None:
        print(f"{pkg:10s}: NOT INSTALLED")
    else:
        module = __import__(pkg)

        version = getattr(module, "__version__", "version unavailable")

        print(f"{pkg:10s}: installed")
        print(f"             version = {version}")
        print(f"             path    = {spec.origin}")


# ============================================================================
# NOTEBOOK INDEX 12: Cell 13. Clone official VeloCycle repository
# ============================================================================

# ============================================
# Cell 13. Clone official VeloCycle repository
# and inspect dependency specifications
# ============================================

import subprocess
from pathlib import Path

velocycle_repo = base_dir / "velocycle"

if not velocycle_repo.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/lamanno-epfl/velocycle.git",
            str(velocycle_repo),
        ],
        check=True,
    )
else:
    print("Repository already exists:", velocycle_repo)

print("\n--- requirements.txt ---")
requirements_file = velocycle_repo / "requirements.txt"
print(requirements_file.read_text())

print("\n--- setup.py dependency-related lines ---")
setup_text = (velocycle_repo / "setup.py").read_text()

for line in setup_text.splitlines():
    if (
        "install_requires" in line
        or "torch" in line.lower()
        or "pyro" in line.lower()
        or "numpy" in line.lower()
        or "pandas" in line.lower()
        or "anndata" in line.lower()
    ):
        print(line)


# ============================================================================
# NOTEBOOK INDEX 13: Cell 14. Create isolated official VeloCycle env
# ============================================================================

# ============================================
# Cell 14. Create isolated official VeloCycle env
# ============================================

import subprocess
from pathlib import Path

env_name = "velocycle-official"

# Check existing conda environments
result = subprocess.run(
    ["conda", "env", "list"],
    capture_output=True,
    text=True,
    check=True,
)

print(result.stdout)

if env_name not in result.stdout:
    print(f"Creating conda environment: {env_name}")

    subprocess.run(
        [
            "conda", "create",
            "-n", env_name,
            "python=3.10",
            "-y",
        ],
        check=True,
    )
else:
    print(f"Environment already exists: {env_name}")

# Locate its Python executable
conda_prefix = Path.home() / ".conda" / "envs" / env_name

# Also check common alternative location
if not conda_prefix.exists():
    conda_prefix = Path.home() / "anaconda3" / "envs" / env_name

if not conda_prefix.exists():
    conda_prefix = Path.home() / "miniconda3" / "envs" / env_name

print("\nCandidate environment path:", conda_prefix)
print("Exists:", conda_prefix.exists())

if conda_prefix.exists():
    python_exe = conda_prefix / "bin" / "python"

    out = subprocess.run(
        [str(python_exe), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )

    print("Python executable:", python_exe)
    print("Python version:", out.stdout.strip() or out.stderr.strip())


# ============================================================================
# NOTEBOOK INDEX 14: Cell 15. Resolve official VeloCycle env path
# ============================================================================

# ============================================
# Cell 15. Resolve official VeloCycle env path
# ============================================

from pathlib import Path
import subprocess

velocycle_env = Path(
    "/environment/miniconda3/envs/velocycle-official"
)

python_official = velocycle_env / "bin" / "python"
pip_official = velocycle_env / "bin" / "pip"

print("Environment:", velocycle_env)
print("Exists:", velocycle_env.exists())

print("\nPython:")
print("  path:", python_official)
print("  exists:", python_official.exists())

print("\nPip:")
print("  path:", pip_official)
print("  exists:", pip_official.exists())

if python_official.exists():
    result = subprocess.run(
        [str(python_official), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    print("\nVersion:", result.stdout.strip() or result.stderr.strip())


# ============================================================================
# NOTEBOOK INDEX 15: Cell 16. Install core official VeloCycle stack
# ============================================================================

# ============================================
# Cell 16. Install core official VeloCycle stack
# ============================================

import subprocess

cmd = [
    str(pip_official),
    "install",

    # Core numerical stack used by official environment
    "numpy==1.24.4",
    "scipy==1.10.1",
    "pandas==2.0.3",
    "anndata==0.9.2",

    # PyTorch / Pyro versions from official requirements
    "torch==2.1.1",
    "pyro-api==0.1.2",
    "pyro-ppl==1.8.6",

    # Other dependencies directly used by VeloCycle
    "scikit-learn==1.3.2",
    "statsmodels==0.14.0",
    "pycircstat==0.0.2",
    "tqdm==4.66.1",
]

print("Installing official-compatible core dependencies...")
print()

subprocess.run(
    cmd,
    check=True,
)

print("\nCore dependency installation completed.")


# ============================================================================
# NOTEBOOK INDEX 16: Cell 17b. Test VeloCycle directly from source
# ============================================================================

# ============================================
# Cell 17b. Test VeloCycle directly from source
# without installing the package
# ============================================

import subprocess

check_code = rf"""
import sys

repo = r"{velocycle_repo}"
sys.path.insert(0, repo)

print("Python:", sys.version)
print("Using source repo:", repo)

import numpy as np
import scipy
import pandas as pd
import anndata
import torch
import pyro

print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pd.__version__)
print("anndata:", anndata.__version__)
print("torch:", torch.__version__)
print("pyro:", pyro.__version__)

import velocycle

print("velocycle imported successfully")
print("velocycle path:", velocycle.__file__)

print("CUDA available:", torch.cuda.is_available())
"""

result = subprocess.run(
    [str(python_official), "-c", check_code],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)


# ============================================================================
# NOTEBOOK INDEX 17: Cell 18b. Install missing VeloCycle
# ============================================================================

# ============================================
# Cell 18b. Install missing VeloCycle
# runtime scientific dependencies
# ============================================

import subprocess

runtime_packages = [
    "matplotlib==3.7.4",
    "scanpy==1.9.6",
    "seaborn==0.12.2",
    "numba==0.58.1",
    "umap-learn==0.5.5",
    "pynndescent==0.5.11",
]

result = subprocess.run(
    [
        str(pip_official),
        "install",
        *runtime_packages,
    ],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)


# ============================================================================
# NOTEBOOK INDEX 18: Cell 19. Re-test VeloCycle import
# ============================================================================

# ============================================
# Cell 19. Re-test VeloCycle import
# ============================================

import subprocess

check_code = rf"""
import sys

repo = r"{velocycle_repo}"
sys.path.insert(0, repo)

import numpy as np
import scipy
import pandas as pd
import anndata
import torch
import pyro
import matplotlib
import scanpy

print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pd.__version__)
print("anndata:", anndata.__version__)
print("torch:", torch.__version__)
print("pyro:", pyro.__version__)
print("matplotlib:", matplotlib.__version__)
print("scanpy:", scanpy.__version__)

import velocycle

print("\nvelocycle imported successfully")
print("velocycle path:", velocycle.__file__)
print("CUDA available:", torch.cuda.is_available())
"""

result = subprocess.run(
    [str(python_official), "-c", check_code],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)


# ============================================================================
# NOTEBOOK INDEX 19: Cell 20. Install missing IPython dependency
# ============================================================================

# ============================================
# Cell 20. Install missing IPython dependency
# ============================================

import subprocess

result = subprocess.run(
    [
        str(pip_official),
        "install",
        "ipython==8.12.3",
    ],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)


# ============================================================================
# NOTEBOOK INDEX 20: Cell 21. Final VeloCycle import check
# ============================================================================

# ============================================
# Cell 21. Final VeloCycle import check
# ============================================

import subprocess

check_code = rf"""
import sys

repo = r"{velocycle_repo}"
sys.path.insert(0, repo)

import torch
import pyro
import velocycle

print("torch:", torch.__version__)
print("pyro:", pyro.__version__)
print("velocycle path:", velocycle.__file__)
print("CUDA available:", torch.cuda.is_available())
"""

result = subprocess.run(
    [str(python_official), "-c", check_code],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)


# ============================================================================
# NOTEBOOK INDEX 21: Cell 22. Inspect VeloCycle git history
# ============================================================================

# ============================================
# Cell 22. Inspect VeloCycle git history
# ============================================

import subprocess

def run_git(args):
    result = subprocess.run(
        ["git", "-C", str(velocycle_repo)] + args,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout

print("--- Current commit ---")
print(
    run_git([
        "log", "-1",
        "--format=%H%n%ad%n%s",
        "--date=iso"
    ])
)

print("--- Recent commits touching phase_inference_model.py ---")
print(
    run_git([
        "log",
        "--follow",
        "--format=%h  %ad  %s",
        "--date=short",
        "-20",
        "--",
        "velocycle/phase_inference_model.py",
    ])
)

print("--- Tags ---")
tags = run_git(["tag", "--list"])
print(tags if tags.strip() else "(no tags)")


# ============================================================================
# NOTEBOOK INDEX 22: Cell 23. Syntax-check historical versions
# ============================================================================

# ============================================
# Cell 23. Syntax-check historical versions
# of phase_inference_model.py
# ============================================

import subprocess

candidate_commits = [
    "2659775",
    "91123be",
    "a092e1e",
]

for commit in candidate_commits:

    result = subprocess.run(
        [
            "git",
            "-C", str(velocycle_repo),
            "show",
            f"{commit}:velocycle/phase_inference_model.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    source = result.stdout

    try:
        compile(
            source,
            f"{commit}:phase_inference_model.py",
            "exec",
        )
        status = "VALID"
        error = ""

    except SyntaxError as e:
        status = "INVALID"
        error = f"line {e.lineno}: {e.msg}"

    print(f"{commit}: {status}")

    if error:
        print("   ", error)


# ============================================================================
# NOTEBOOK INDEX 23: Cell 24. Create a clean VeloCycle worktree
# ============================================================================

# ============================================
# Cell 24. Create a clean VeloCycle worktree
# at commit 91123be
# ============================================

import subprocess
from pathlib import Path

velocycle_legacy = base_dir / "velocycle_91123be"

if not velocycle_legacy.exists():
    subprocess.run(
        [
            "git",
            "-C", str(velocycle_repo),
            "worktree",
            "add",
            str(velocycle_legacy),
            "91123be",
        ],
        check=True,
    )
else:
    print("Worktree already exists:", velocycle_legacy)

print("\nLegacy VeloCycle path:")
print(velocycle_legacy)

# Verify commit
result = subprocess.run(
    [
        "git",
        "-C", str(velocycle_legacy),
        "log",
        "-1",
        "--format=%H%n%ad%n%s",
        "--date=iso",
    ],
    capture_output=True,
    text=True,
    check=True,
)

print("\nCommit:")
print(result.stdout)

print("phase_inference_model.py exists:",
      (velocycle_legacy / "velocycle" / "phase_inference_model.py").exists())


# ============================================================================
# NOTEBOOK INDEX 24: Cell 25. Test import using legacy VeloCycle source
# ============================================================================

# ============================================
# Cell 25. Test import using legacy VeloCycle source
# ============================================

import subprocess

check_code = rf"""
import sys

repo = r"{velocycle_legacy}"
sys.path.insert(0, repo)

import torch
import pyro

print("torch:", torch.__version__)
print("pyro:", pyro.__version__)

import velocycle

print("\nvelocycle imported successfully")
print("velocycle path:", velocycle.__file__)
print("CUDA available:", torch.cuda.is_available())
"""

result = subprocess.run(
    [str(python_official), "-c", check_code],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)


# ============================================================================
# NOTEBOOK INDEX 25: Cell 26. Load official 986-condition pickle
# ============================================================================

# ============================================
# Cell 26. Load official 986-condition pickle
# safely onto CPU and inspect top-level objects
# ============================================

import subprocess

pickle_986 = (
    base_dir
    / "pickle_result_outputs"
    / "Replogle_Saunders_PerturbRPE1_75cells_phase_velocity_data_fit_LargeGene_986conditions.pkl.gz"
)

inspect_code = rf"""
import sys
import gzip
import pickle
import io
import torch

# Use the syntax-valid historical VeloCycle source
repo = r"{velocycle_legacy}"
sys.path.insert(0, repo)

import velocycle

pickle_path = r"{pickle_986}"

print("VeloCycle:", velocycle.__file__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Pickle:", pickle_path)
print()

# ------------------------------------------------
# Force tensors stored on cuda:0 to load onto CPU
# ------------------------------------------------
_original_load_from_bytes = torch.storage._load_from_bytes

torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b),
    map_location="cpu"
)

print("Loading gzip pickle onto CPU...")
print("This may take several minutes.")
print()

with gzip.open(pickle_path, "rb") as f:
    obj = pickle.load(f)

print("LOAD SUCCESSFUL")
print()

print("Top-level type:", type(obj))

try:
    print("Top-level length:", len(obj))
except Exception:
    pass

if isinstance(obj, (list, tuple)):
    for i, x in enumerate(obj):
        print()
        print("=" * 60)
        print("Object", i)
        print("type:", type(x))

        if hasattr(x, "__dict__"):
            keys = sorted(x.__dict__.keys())
            print("attributes:")
            for k in keys:
                print("  ", k)
"""

env = {
    **__import__("os").environ,
    "CUDA_VISIBLE_DEVICES": "",
}

result = subprocess.run(
    [str(python_official), "-c", inspect_code],
    capture_output=True,
    text=True,
    env=env,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)


# ============================================================================
# NOTEBOOK INDEX 26: Cell 27. Inspect phase, speed, and kinetics
# ============================================================================

# ============================================
# Cell 27. Inspect phase, speed, and kinetics
# from official VeloCycle result
# ============================================

import subprocess
import os

inspect_code = rf"""
import sys
import gzip
import pickle
import io
import torch
import numpy as np
import pandas as pd

repo = r"{velocycle_legacy}"
sys.path.insert(0, repo)

import velocycle

pickle_path = r"{pickle_986}"

# Force CUDA tensors onto CPU
torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b),
    map_location="cpu"
)

with gzip.open(pickle_path, "rb") as f:
    phase_fit, velocity_fit, data_to_fit = pickle.load(f)


def summarize(name, x):
    print()
    print("=" * 70)
    print(name)
    print("type:", type(x))

    if hasattr(x, "shape"):
        try:
            print("shape:", x.shape)
        except Exception:
            pass

    if isinstance(x, torch.Tensor):
        print("dtype:", x.dtype)
        print("device:", x.device)

    if isinstance(x, pd.DataFrame):
        print("index head:", list(x.index[:5]))
        print("columns head:", list(x.columns[:10]))

    elif isinstance(x, pd.Series):
        print("index head:", list(x.index[:5]))
        print("head values:", x.head().tolist())

    elif isinstance(x, dict):
        print("keys head:", list(x.keys())[:20])

    elif hasattr(x, "means"):
        print("has .means:", True)
        print(".means type:", type(x.means))

        if hasattr(x.means, "shape"):
            print(".means shape:", x.means.shape)

        if isinstance(x.means, pd.DataFrame):
            print(".means index head:", list(x.means.index[:5]))
            print(".means columns head:", list(x.means.columns[:10]))


print("data_to_fit shape:", data_to_fit.shape)
print("obs columns:", list(data_to_fit.obs.columns))
print("var columns:", list(data_to_fit.var.columns))
print("first 10 genes:", data_to_fit.var_names[:10].tolist())

summarize("phase_fit.condition", phase_fit.condition)
summarize("phase_fit.phis_pyro", phase_fit.phis_pyro)
summarize("phase_fit.phase_pyro", phase_fit.phase_pyro)

summarize("velocity_fit.condition", velocity_fit.condition)
summarize("velocity_fit.speed_pyro", velocity_fit.speed_pyro)
summarize("velocity_fit.log_betas", velocity_fit.log_betas)
summarize("velocity_fit.log_gammas", velocity_fit.log_gammas)
"""

env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
}

result = subprocess.run(
    [str(python_official), "-c", inspect_code],
    capture_output=True,
    text=True,
    env=env,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)


# ============================================================================
# NOTEBOOK INDEX 27: Cell 28. Understand official phase representation
# ============================================================================

# ============================================
# Cell 28. Understand official phase representation
# ============================================

from pathlib import Path

phases_source = (
    velocycle_legacy
    / "velocycle"
    / "phases.py"
)

print("--- Relevant Phases source ---")

source = phases_source.read_text()

# Print the Phases class definition
start = source.find("class Phases")
end = source.find("\nclass ", start + 1)

if end == -1:
    end = len(source)

print(source[start:end])


# ============================================================================
# NOTEBOOK INDEX 28: Cell 29b. Validate full VeloCycle phase
# ============================================================================

# ============================================
# Cell 29b. Validate full VeloCycle phase
# against official phase metadata
# ============================================

import subprocess
import os

validate_code = r"""
import sys
import gzip
import pickle
import io
import torch
import numpy as np
import pandas as pd

repo = r"__VELO_REPO__"
sys.path.insert(0, repo)

import velocycle

pickle_path = r"__PICKLE_PATH__"
phase_csv = r"__PHASE_CSV__"

torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b),
    map_location="cpu"
)

with gzip.open(pickle_path, "rb") as f:
    phase_fit, velocity_fit, data_to_fit = pickle.load(f)

# Official full-cell phase
phase_obj = phase_fit.phase_pyro

phi_full = np.asarray(
    phase_obj.directions,
    dtype=float
)

cell_ids = np.asarray(
    phase_obj.phi_xy.columns.astype(str)
)

print("Full phase cells:", len(phi_full))
print("phi range:", phi_full.min(), phi_full.max())

print(
    "Phase columns match data_to_fit.obs_names:",
    np.array_equal(
        cell_ids,
        data_to_fit.obs_names.astype(str).values
    )
)

print(
    "phis_pyro shape:",
    np.asarray(phase_fit.phis_pyro).shape
)

print(
    "phis_pyro == phase_pyro.phi_xy:",
    np.allclose(
        np.asarray(phase_fit.phis_pyro),
        phase_obj.phi_xy.values
    )
)

# Validate against official CSV
meta = pd.read_csv(
    phase_csv,
    index_col=0
)

phi_series = pd.Series(
    phi_full,
    index=cell_ids,
    name="phi_pickle"
)

common = meta.index.intersection(phi_series.index)

phi_a = phi_series.loc[common].values
phi_b = meta.loc[common, "cell_cycle_phi"].values

delta = np.angle(
    np.exp(1j * (phi_a - phi_b))
)

abs_error = np.abs(delta)

print()
print("Matched cells with official CSV:", len(common))
print("Median circular error (rad):", np.median(abs_error))
print("Mean circular error (rad):", np.mean(abs_error))
print("Max circular error (rad):", np.max(abs_error))

print()
print("First 5 comparisons:")
print(
    pd.DataFrame({
        "pickle_phi": phi_a[:5],
        "csv_phi": phi_b[:5],
        "abs_circular_error": abs_error[:5],
    })
)
"""

validate_code = (
    validate_code
    .replace("__VELO_REPO__", str(velocycle_legacy))
    .replace("__PICKLE_PATH__", str(pickle_986))
    .replace("__PHASE_CSV__", str(phase_metadata))
)

env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
}

result = subprocess.run(
    [str(python_official), "-c", validate_code],
    capture_output=True,
    text=True,
    env=env,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)


# ============================================================================
# NOTEBOOK INDEX 29: Cell 30. Check phase consistency between
# ============================================================================

# ============================================
# Cell 30. Check phase consistency between
# phase_fit and velocity_fit
# ============================================

import subprocess
import os

check_code = r"""
import sys
import gzip
import pickle
import io
import torch
import numpy as np

repo = r"__VELO_REPO__"
sys.path.insert(0, repo)

import velocycle

pickle_path = r"__PICKLE_PATH__"

torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b),
    map_location="cpu"
)

with gzip.open(pickle_path, "rb") as f:
    phase_fit, velocity_fit, data_to_fit = pickle.load(f)

p_phase = phase_fit.phase_pyro
p_vel   = velocity_fit.phase_pyro

phi_phase = np.asarray(p_phase.directions, dtype=float)
phi_vel   = np.asarray(p_vel.directions, dtype=float)

ids_phase = np.asarray(p_phase.phi_xy.columns.astype(str))
ids_vel   = np.asarray(p_vel.phi_xy.columns.astype(str))

print("phase_fit cells:", len(phi_phase))
print("velocity_fit cells:", len(phi_vel))

print(
    "Cell IDs identical:",
    np.array_equal(ids_phase, ids_vel)
)

print(
    "phi_xy identical:",
    np.allclose(
        p_phase.phi_xy.values,
        p_vel.phi_xy.values
    )
)

delta = np.angle(
    np.exp(1j * (phi_phase - phi_vel))
)
abs_error = np.abs(delta)

print()
print(
    "Median circular difference (rad):",
    np.median(abs_error)
)
print(
    "Mean circular difference (rad):",
    np.mean(abs_error)
)
print(
    "Max circular difference (rad):",
    np.max(abs_error)
)

print()
print("velocity_fit phase range:",
      phi_vel.min(), phi_vel.max())
"""

check_code = (
    check_code
    .replace("__VELO_REPO__", str(velocycle_legacy))
    .replace("__PICKLE_PATH__", str(pickle_986))
)

env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
}

result = subprocess.run(
    [str(python_official), "-c", check_code],
    capture_output=True,
    text=True,
    env=env,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)


# ============================================================================
# NOTEBOOK INDEX 30: Cell 31. Export compact official VeloCycle
# ============================================================================

# ============================================
# Cell 31. Export compact official VeloCycle
# checkpoint for downstream GRN analysis
# ============================================

import subprocess
import os

extract_path = base_dir / "velocycle_986_official_extract.npz"

extract_code = r"""
import sys
import gzip
import pickle
import io
import torch
import numpy as np

repo = r"__VELO_REPO__"
sys.path.insert(0, repo)

import velocycle

pickle_path = r"__PICKLE_PATH__"
out_path = r"__OUT_PATH__"

torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b),
    map_location="cpu"
)

print("Loading official VeloCycle pickle...")

with gzip.open(pickle_path, "rb") as f:
    phase_fit, velocity_fit, data_to_fit = pickle.load(f)

# Use velocity-fit phase for internal consistency
phase_obj = velocity_fit.phase_pyro

cell_ids = np.asarray(
    phase_obj.phi_xy.columns.astype(str)
)

phi = np.asarray(
    phase_obj.directions,
    dtype=np.float64
)

# Condition-specific angular speed posterior means
speed_df = velocity_fit.speed_pyro.means

condition_names = np.asarray(
    speed_df.columns.astype(str)
)

speed_raw = np.asarray(
    speed_df.loc["nu0"].values,
    dtype=np.float64
)

# Kinetic parameters align with data_to_fit's genes
gene_names = np.asarray(
    data_to_fit.var_names.astype(str)
)

log_betas = np.asarray(
    velocity_fit.log_betas,
    dtype=np.float64
)

log_gammas = np.asarray(
    velocity_fit.log_gammas,
    dtype=np.float64
)

assert len(cell_ids) == len(phi)
assert len(condition_names) == len(speed_raw)
assert len(gene_names) == len(log_betas) == len(log_gammas)

np.savez_compressed(
    out_path,
    cell_ids=cell_ids,
    phi=phi,
    condition_names=condition_names,
    speed_raw=speed_raw,
    gene_names=gene_names,
    log_betas=log_betas,
    log_gammas=log_gammas,
)

print()
print("Saved:", out_path)
print("cells:", len(cell_ids))
print("conditions:", len(condition_names))
print("genes:", len(gene_names))
print("phi range:", phi.min(), phi.max())
print("speed range:", speed_raw.min(), speed_raw.max())
print("log_beta range:", log_betas.min(), log_betas.max())
print("log_gamma range:", log_gammas.min(), log_gammas.max())
"""

extract_code = (
    extract_code
    .replace("__VELO_REPO__", str(velocycle_legacy))
    .replace("__PICKLE_PATH__", str(pickle_986))
    .replace("__OUT_PATH__", str(extract_path))
)

env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
}

result = subprocess.run(
    [str(python_official), "-c", extract_code],
    capture_output=True,
    text=True,
    env=env,
)

print(result.stdout)

if result.stderr:
    print("--- stderr ---")
    print(result.stderr)

print("Return code:", result.returncode)
print("Extract exists:", extract_path.exists())

if extract_path.exists():
    print(
        "Extract size (MB):",
        round(extract_path.stat().st_size / 1024**2, 2)
    )


# ============================================================================
# NOTEBOOK INDEX 31: Cell 32. Inspect AngularSpeed parameterization
# ============================================================================

# ============================================
# Cell 32. Inspect AngularSpeed parameterization
# ============================================

from pathlib import Path

angular_source = (
    velocycle_legacy
    / "velocycle"
    / "angularspeed.py"
)

velocity_source = (
    velocycle_legacy
    / "velocycle"
    / "velocity_inference_model.py"
)

print("========== angularspeed.py ==========")

src = angular_source.read_text()

start = src.find("class AngularSpeed")
end = src.find("\nclass ", start + 1)

if end == -1:
    end = len(src)

print(src[start:end])

print("\n\n========== occurrences of 'speed_pyro' ==========")

vsrc = velocity_source.read_text()

for i, line in enumerate(vsrc.splitlines(), start=1):
    if (
        "speed_pyro" in line
        or "nu0" in line
        or "ν" in line
        or "omega" in line.lower()
    ):
        print(f"{i:4d}: {line}")


# ============================================================================
# NOTEBOOK INDEX 32: Cell 33. Inspect VeloCycle speed estimates
# ============================================================================

# ============================================
# Cell 33. Inspect VeloCycle speed estimates
# for NT and our final 153 perturbations
# ============================================

import numpy as np
import pandas as pd

vc = np.load(extract_path, allow_pickle=True)

condition_names_vc = vc["condition_names"].astype(str)
speed_raw_vc = vc["speed_raw"].astype(float)

speed_series = pd.Series(
    speed_raw_vc,
    index=condition_names_vc,
    name="nu0"
)

# NT
nt_speed = speed_series.loc["non-targeting"]

print("NT nu0:", nt_speed)

print("\nAll 986 conditions:")
print(speed_series.describe(
    percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\nSign counts, all conditions:")
print("positive:", int((speed_series > 0).sum()))
print("zero:", int((speed_series == 0).sum()))
print("negative:", int((speed_series < 0).sum()))

# Our final perturbation set
missing = [
    g for g in final_perturbations_qc
    if g not in speed_series.index
]

speed_qc = speed_series.loc[final_perturbations_qc]

print("\nFinal perturbations:", len(final_perturbations_qc))
print("Missing from VeloCycle speed:", len(missing))

print("\nSelected 153 speed summary:")
print(speed_qc.describe(
    percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\nSign counts, selected perturbations:")
print("positive:", int((speed_qc > 0).sum()))
print("zero:", int((speed_qc == 0).sum()))
print("negative:", int((speed_qc < 0).sum()))

print("\nNegative selected conditions:")
print(speed_qc[speed_qc < 0].sort_values())


# ============================================================================
# NOTEBOOK INDEX 33: Cell 33b. Recover final perturbation list
# ============================================================================

# ============================================
# Cell 33b. Recover final perturbation list
# from saved GRN checkpoint
# ============================================

import numpy as np

checkpoint_path = base_dir / "replogle_grn_checkpoint.npz"

ckpt = np.load(
    checkpoint_path,
    allow_pickle=True
)

print("Checkpoint keys:")
print(ckpt.files)

for key in ckpt.files:
    x = ckpt[key]

    try:
        shape = x.shape
    except Exception:
        shape = None

    print(f"{key:30s} shape={shape}")


# ============================================================================
# NOTEBOOK INDEX 34: Cell 34. Restore final perturbations
# ============================================================================

# ============================================
# Cell 34. Restore final perturbations
# and finish speed QC
# ============================================

final_conditions_qc = ckpt["final_conditions_qc"].astype(str).tolist()
regulator_genes_qc = ckpt["regulator_genes_qc"].astype(str).tolist()

final_perturbations_qc = [
    x for x in final_conditions_qc
    if x != "non-targeting"
]

print("Final conditions:", len(final_conditions_qc))
print("Final perturbations:", len(final_perturbations_qc))
print("Regulator genes:", len(regulator_genes_qc))

print(
    "Perturbations == regulator genes:",
    final_perturbations_qc == regulator_genes_qc
)

# Continue speed QC
missing = [
    g for g in final_perturbations_qc
    if g not in speed_series.index
]

speed_qc = speed_series.loc[final_perturbations_qc]

print("\nMissing from VeloCycle speed:", len(missing))
if missing:
    print(missing)

print("\nSelected 153 speed summary:")
print(
    speed_qc.describe(
        percentiles=[
            0.01, 0.05, 0.10, 0.25,
            0.50, 0.75, 0.90, 0.95, 0.99
        ]
    )
)

print("\nSign counts, selected perturbations:")
print("positive:", int((speed_qc > 0).sum()))
print("zero:", int((speed_qc == 0).sum()))
print("negative:", int((speed_qc < 0).sum()))

print("\nNegative selected conditions:")
print(speed_qc[speed_qc < 0].sort_values())


# ============================================================================
# NOTEBOOK INDEX 35: Cell 35. Check kinetics coverage for the
# ============================================================================

# ============================================
# Cell 35. Check kinetics coverage for the
# final 153 regulator / response genes
# ============================================

gene_names_vc = vc["gene_names"].astype(str)
log_betas_vc = vc["log_betas"].astype(float)
log_gammas_vc = vc["log_gammas"].astype(float)

vc_gene_set = set(gene_names_vc)

covered_genes = [
    g for g in regulator_genes_qc
    if g in vc_gene_set
]

missing_genes = [
    g for g in regulator_genes_qc
    if g not in vc_gene_set
]

print("Final regulator genes:", len(regulator_genes_qc))
print("Covered by VeloCycle kinetics:", len(covered_genes))
print("Missing kinetics:", len(missing_genes))

if missing_genes:
    print("\nMissing genes:")
    print(missing_genes)

# Map gene -> kinetic parameters
beta_series = pd.Series(
    np.exp(log_betas_vc),
    index=gene_names_vc,
    name="beta"
)

gamma_series = pd.Series(
    np.exp(log_gammas_vc),
    index=gene_names_vc,
    name="gamma"
)

if covered_genes:
    print("\nBeta summary for covered genes:")
    print(beta_series.loc[covered_genes].describe())

    print("\nGamma summary for covered genes:")
    print(gamma_series.loc[covered_genes].describe())


# ============================================================================
# NOTEBOOK INDEX 36: Cell 36. Check the 426 VeloCycle kinetic genes
# ============================================================================

# ============================================
# Cell 36. Check the 426 VeloCycle kinetic genes
# against the full RPE1 H5AD
# ============================================

vc_genes = gene_names_vc.tolist()

adata_gene_set = set(adata.var_names.astype(str))

vc_in_adata = [
    g for g in vc_genes
    if g in adata_gene_set
]

vc_missing_adata = [
    g for g in vc_genes
    if g not in adata_gene_set
]

print("VeloCycle kinetic genes:", len(vc_genes))
print("Present in full H5AD:", len(vc_in_adata))
print("Missing from full H5AD:", len(vc_missing_adata))

if vc_missing_adata:
    print("\nMissing genes:")
    print(vc_missing_adata)

# Also inspect overlap structure
reg_set = set(regulator_genes_qc)
vc_set = set(vc_genes)

overlap = [
    g for g in regulator_genes_qc
    if g in vc_set
]

print("\nRegulator set R:", len(regulator_genes_qc))
print("Candidate response set G:", len(vc_genes))
print("R ∩ G:", len(overlap))

print("\nR ∩ G genes:")
print(overlap)


# ============================================================================
# NOTEBOOK INDEX 37: Cell 37. NT expression QC for the 426
# ============================================================================

# ============================================
# Cell 37. NT expression QC for the 426
# candidate response genes
# ============================================

import numpy as np
import pandas as pd
from scipy import sparse

# NT cells
nt_mask = (
    adata.obs["gene"].astype(str).values
    == "non-targeting"
)

# Gene indices in the full H5AD
gene_to_idx = {
    g: i
    for i, g in enumerate(adata.var_names.astype(str))
}

vc_gene_idx = np.array(
    [gene_to_idx[g] for g in vc_genes],
    dtype=int
)

# Sparse slices only; do not copy full adata
S_nt = adata.layers["spliced"][nt_mask, :][:, vc_gene_idx]
U_nt = adata.layers["unspliced"][nt_mask, :][:, vc_gene_idx]

# Mean counts
S_mean = np.asarray(S_nt.mean(axis=0)).ravel()
U_mean = np.asarray(U_nt.mean(axis=0)).ravel()

# Fraction of NT cells with nonzero counts
S_detect = np.asarray(
    (S_nt > 0).mean(axis=0)
).ravel()

U_detect = np.asarray(
    (U_nt > 0).mean(axis=0)
).ravel()

response_qc = pd.DataFrame(
    {
        "S_mean_NT": S_mean,
        "U_mean_NT": U_mean,
        "S_detect_NT": S_detect,
        "U_detect_NT": U_detect,
        "beta": beta_series.loc[vc_genes].values,
        "gamma": gamma_series.loc[vc_genes].values,
    },
    index=vc_genes,
)

print("Candidate response genes:", len(response_qc))

print("\nSpliced mean:")
print(response_qc["S_mean_NT"].describe())

print("\nUnspliced mean:")
print(response_qc["U_mean_NT"].describe())

print("\nSpliced detection fraction:")
print(response_qc["S_detect_NT"].describe())

print("\nUnspliced detection fraction:")
print(response_qc["U_detect_NT"].describe())

# Same thresholds we previously used for regulator eligibility
pass_basic = (
    (response_qc["S_mean_NT"] >= 0.1)
    & (response_qc["U_mean_NT"] >= 0.02)
)

print(
    "\nGenes passing mean S >= 0.1 and mean U >= 0.02:",
    int(pass_basic.sum()),
    "/",
    len(pass_basic),
)

print("\nLowest 15 genes by unspliced mean:")
print(
    response_qc
    .sort_values("U_mean_NT")
    .head(15)[
        [
            "S_mean_NT",
            "U_mean_NT",
            "S_detect_NT",
            "U_detect_NT",
        ]
    ]
)


# ============================================================================
# NOTEBOOK INDEX 38: Cell 38. Inspect VeloCycle normalization
# ============================================================================

# ============================================
# Cell 38. Inspect VeloCycle normalization
# before rebuilding trajectories
# ============================================

print("Full H5AD layers:")
print(list(adata.layers.keys()))

print("\nMedGene H5AD layers:")
print(list(adata_med.layers.keys()))

print("\nMedGene obs normalization-related columns:")
print([
    c for c in adata_med.obs.columns
    if (
        "count" in c.lower()
        or "size" in c.lower()
        or "scale" in c.lower()
    )
])

# Search VeloCycle source for S_sz / U_sz construction
print("\n========== VeloCycle source occurrences ==========")

for pyfile in velocycle_legacy.rglob("*.py"):
    try:
        lines = pyfile.read_text().splitlines()
    except Exception:
        continue

    hits = []

    for i, line in enumerate(lines, start=1):
        if (
            "S_sz" in line
            or "U_sz" in line
            or "n_scounts" in line
            or "n_ucounts" in line
        ):
            hits.append((i, line.strip()))

    if hits:
        print(f"\n--- {pyfile.relative_to(velocycle_legacy)} ---")
        for i, line in hits:
            print(f"{i:4d}: {line}")


# ============================================================================
# NOTEBOOK INDEX 39: Cell 39. Verify how VeloCycle n_scounts /
# ============================================================================

# ============================================
# Cell 39. Verify how VeloCycle n_scounts /
# n_ucounts are defined on a filtered gene set
# ============================================

import numpy as np

# Raw totals within the 120-gene MedGene object
S_sum_120 = np.asarray(
    adata_med.layers["spliced"].sum(axis=1)
).ravel()

U_sum_120 = np.asarray(
    adata_med.layers["unspliced"].sum(axis=1)
).ravel()

n_s_saved = adata_med.obs["n_scounts"].to_numpy(dtype=float)
n_u_saved = adata_med.obs["n_ucounts"].to_numpy(dtype=float)

print(
    "n_scounts == raw spliced sum over 120 genes:",
    np.allclose(n_s_saved, S_sum_120)
)

print(
    "n_ucounts == raw unspliced sum over 120 genes:",
    np.allclose(n_u_saved, U_sum_120)
)

print("\nMax absolute difference:")
print(
    "spliced:",
    np.max(np.abs(n_s_saved - S_sum_120))
)
print(
    "unspliced:",
    np.max(np.abs(n_u_saved - U_sum_120))
)

# Verify stored S_sz against the source-code formula
S_factor = n_s_saved.mean() / n_s_saved

# check a small slice only
rows = np.arange(100)
cols = np.arange(min(20, adata_med.n_vars))

S_raw_small = adata_med.layers["spliced"][rows, :][:, cols]
if hasattr(S_raw_small, "toarray"):
    S_raw_small = S_raw_small.toarray()

S_expected = (
    S_raw_small
    * S_factor[rows, None]
)

S_stored = adata_med.layers["S_sz"][rows, :][:, cols]
if hasattr(S_stored, "toarray"):
    S_stored = S_stored.toarray()

print(
    "\nStored S_sz matches VeloCycle formula:",
    np.allclose(
        S_expected,
        S_stored,
        rtol=1e-5,
        atol=1e-7
    )
)

print("\nMean n_scounts:", n_s_saved.mean())
print("Mean n_ucounts:", n_u_saved.mean())


# ============================================================================
# NOTEBOOK INDEX 40: Cell 40. Test whether VeloCycle n_scounts /
# ============================================================================

# ============================================
# Cell 40. Test whether VeloCycle n_scounts /
# n_ucounts are full-transcriptome totals
# ============================================

import numpy as np

# Check cell ordering first
print(
    "Cell order identical:",
    np.array_equal(
        adata.obs_names.astype(str).values,
        adata_med.obs_names.astype(str).values
    )
)

# Sparse row sums over the full H5AD
S_sum_full = np.asarray(
    adata.layers["spliced"].sum(axis=1)
).ravel()

U_sum_full = np.asarray(
    adata.layers["unspliced"].sum(axis=1)
).ravel()

n_s_saved = adata_med.obs["n_scounts"].to_numpy(dtype=float)
n_u_saved = adata_med.obs["n_ucounts"].to_numpy(dtype=float)

print(
    "\nn_scounts == full-H5AD spliced totals:",
    np.allclose(n_s_saved, S_sum_full)
)

print(
    "n_ucounts == full-H5AD unspliced totals:",
    np.allclose(n_u_saved, U_sum_full)
)

print("\nMax absolute difference:")
print(
    "spliced:",
    np.max(np.abs(n_s_saved - S_sum_full))
)
print(
    "unspliced:",
    np.max(np.abs(n_u_saved - U_sum_full))
)

print("\nMeans:")
print("saved n_scounts:", n_s_saved.mean())
print("full spliced total:", S_sum_full.mean())
print("saved n_ucounts:", n_u_saved.mean())
print("full unspliced total:", U_sum_full.mean())


# ============================================================================
# NOTEBOOK INDEX 41: Cell 41. Attach official VeloCycle phase
# ============================================================================

# ============================================
# Cell 41. Attach official VeloCycle phase
# and re-check 10-bin coverage
# ============================================

import numpy as np
import pandas as pd

# Official compact extract
vc_cell_ids = vc["cell_ids"].astype(str)
vc_phi = vc["phi"].astype(float)

# Sanity: official extract should cover all cells
print("Official phase cells:", len(vc_cell_ids))
print("Full adata cells:", adata.n_obs)

print(
    "Same cell set:",
    set(vc_cell_ids) == set(adata.obs_names.astype(str))
)

# Map cell ID -> phi
phi_series = pd.Series(
    vc_phi,
    index=vc_cell_ids,
    name="phi_vc"
)

# Align to full adata order
phi_aligned = phi_series.loc[
    adata.obs_names.astype(str)
].to_numpy()

# normalized circular phase x in [0, 1)
cycle_time_vc = (
    phi_aligned / (2.0 * np.pi)
) % 1.0

print(
    "\ncycle_time_vc range:",
    cycle_time_vc.min(),
    cycle_time_vc.max()
)

# Final 154-condition mask
cond = adata.obs["gene"].astype(str).to_numpy()

final_mask = np.isin(
    cond,
    final_conditions_qc
)

# 10 phase bins
bin_edges = np.linspace(0.0, 1.0, 11)

bin_id = np.digitize(
    cycle_time_vc,
    bin_edges[1:-1],
    right=False
)

# Count cells per condition x bin
coverage_rows = []

for q in final_conditions_qc:
    q_mask = final_mask & (cond == q)

    counts = np.bincount(
        bin_id[q_mask],
        minlength=10
    )

    coverage_rows.append(
        {
            "condition": q,
            "n_cells": int(q_mask.sum()),
            "occupied_bins": int((counts > 0).sum()),
            "usable_bins_ge5": int((counts >= 5).sum()),
            "min_bin_count": int(counts.min()),
            "max_bin_count": int(counts.max()),
        }
    )

coverage_vc = pd.DataFrame(
    coverage_rows
).set_index("condition")

print("\nOfficial-phase coverage summary:")
print(
    coverage_vc[
        ["occupied_bins", "usable_bins_ge5"]
    ].describe()
)

print("\nUsable-bin count distribution:")
print(
    coverage_vc["usable_bins_ge5"]
    .value_counts()
    .sort_index()
)

print("\nConditions with < 7 usable bins:")
print(
    coverage_vc.loc[
        coverage_vc["usable_bins_ge5"] < 7,
        ["n_cells", "usable_bins_ge5"]
    ]
    .sort_values("usable_bins_ge5")
)


# ============================================================================
# NOTEBOOK INDEX 42: Cell 42. Freeze final condition set under
# ============================================================================

# ============================================
# Cell 42. Freeze final condition set under
# official VeloCycle phase
# ============================================

min_usable_bins = 7

final_conditions_vc = [
    q for q in final_conditions_qc
    if coverage_vc.loc[q, "usable_bins_ge5"] >= min_usable_bins
]

final_perturbations_vc = [
    q for q in final_conditions_vc
    if q != "non-targeting"
]

removed_by_official_phase = [
    q for q in final_conditions_qc
    if q not in final_conditions_vc
]

print("Final conditions:", len(final_conditions_vc))
print("Final perturbations:", len(final_perturbations_vc))

print("\nRemoved after official-phase coverage QC:")
print(removed_by_official_phase)

# Regulator set follows the retained perturbation targets
regulator_genes_vc = final_perturbations_vc.copy()

print("\nRegulator genes:", len(regulator_genes_vc))
print(
    "Perturbations == regulator genes:",
    final_perturbations_vc == regulator_genes_vc
)

# Speed QC for the frozen set
speed_final = speed_series.loc[final_conditions_vc]

print("\nFinal-condition speed summary:")
print(speed_final.describe())

print("\nMinimum-speed conditions:")
print(speed_final.sort_values().head(10))

print(
    "\nAll final speeds positive:",
    bool((speed_final > 0).all())
)

# Relative speed with NT = 1
omega_nt = float(speed_series.loc["non-targeting"])
rho_final = speed_final / omega_nt

print("\nNT speed:", omega_nt)

print("\nRelative-speed summary (NT = 1):")
print(rho_final.describe())


# ============================================================================
# NOTEBOOK INDEX 43: Cell 43. Put VeloCycle speed and kinetics
# ============================================================================

# ============================================
# Cell 43. Put VeloCycle speed and kinetics
# into NT-cycle normalized time units
# ============================================

import numpy as np
import pandas as pd

# NT angular speed
omega_nt = float(speed_series.loc["non-targeting"])

# --------------------------------------------
# 1. Condition-specific relative cycle speed
# --------------------------------------------

rho_final = (
    speed_series.loc[final_conditions_vc]
    / omega_nt
)

# --------------------------------------------
# 2. Kinetic rates in normalized time:
#    1 time unit = one NT cell cycle
# --------------------------------------------

time_scale = 2.0 * np.pi / omega_nt

beta_cycle = (
    beta_series.loc[vc_genes]
    * time_scale
)

gamma_cycle = (
    gamma_series.loc[vc_genes]
    * time_scale
)

# --------------------------------------------
# 3. Duration of one 10-bin phase interval
# --------------------------------------------

delta_x = 1.0 / 10.0

delta_t = delta_x / rho_final

# --------------------------------------------
# Diagnostics
# --------------------------------------------

print("omega_NT:", omega_nt)
print("time-scale factor 2pi / omega_NT:", time_scale)

print("\nRelative speed rho:")
print(rho_final.describe())

print("\nDuration of one 0.1-cycle interval:")
print(delta_t.describe())

print("\nBeta in NT-cycle units:")
print(beta_cycle.describe())

print("\nGamma in NT-cycle units:")
print(gamma_cycle.describe())

print("\nSlowest condition:")
slowest = rho_final.idxmin()
print(
    slowest,
    "rho =", rho_final.loc[slowest],
    "delta_t =", delta_t.loc[slowest]
)

print("\nFastest condition:")
fastest = rho_final.idxmax()
print(
    fastest,
    "rho =", rho_final.loc[fastest],
    "delta_t =", delta_t.loc[fastest]
)

print("\nNT interval duration:")
print(delta_t.loc["non-targeting"])


# ============================================================================
# NOTEBOOK INDEX 44: Cell 44. Build VeloCycle-normalized sparse
# ============================================================================

# ============================================
# Cell 44. Build VeloCycle-normalized sparse
# expression slices for final analysis
# ============================================

import numpy as np
from scipy import sparse

# --------------------------------------------
# 1. Final cells
# --------------------------------------------

cond_all = adata.obs["gene"].astype(str).to_numpy()

final_mask_vc = np.isin(
    cond_all,
    final_conditions_vc
)

final_cell_idx = np.flatnonzero(final_mask_vc)

print("Final cells:", len(final_cell_idx))

# Conditions and official phase for these cells
condition_final_cells = cond_all[final_cell_idx]
cycle_final_cells = cycle_time_vc[final_cell_idx]

# --------------------------------------------
# 2. Gene indices
# --------------------------------------------

gene_to_idx = {
    g: i
    for i, g in enumerate(adata.var_names.astype(str))
}

response_idx = np.array(
    [gene_to_idx[g] for g in vc_genes],
    dtype=int
)

regulator_idx = np.array(
    [gene_to_idx[g] for g in regulator_genes_vc],
    dtype=int
)

print("Response genes:", len(response_idx))
print("Regulator genes:", len(regulator_idx))

# --------------------------------------------
# 3. Raw sparse expression slices
# --------------------------------------------

U_response_raw = adata.layers["unspliced"][
    final_cell_idx, :
][:, response_idx].tocsr()

S_regulator_raw = adata.layers["spliced"][
    final_cell_idx, :
][:, regulator_idx].tocsr()

print("\nRaw matrix shapes:")
print("U_response_raw:", U_response_raw.shape)
print("S_regulator_raw:", S_regulator_raw.shape)

# --------------------------------------------
# 4. VeloCycle size factors
#
# U uses full-transcriptome unspliced totals.
# S uses full-transcriptome spliced totals.
# --------------------------------------------

n_s_full = adata_med.obs["n_scounts"].to_numpy(dtype=float)
n_u_full = adata_med.obs["n_ucounts"].to_numpy(dtype=float)

s_factor_all = n_s_full.mean() / n_s_full
u_factor_all = n_u_full.mean() / n_u_full

s_factor = s_factor_all[final_cell_idx]
u_factor = u_factor_all[final_cell_idx]

print("\nSize-factor ranges:")
print("S:", s_factor.min(), s_factor.max())
print("U:", u_factor.min(), u_factor.max())

# Sparse row-wise scaling
S_regulator_sz = sparse.diags(
    s_factor
) @ S_regulator_raw

U_response_sz = sparse.diags(
    u_factor
) @ U_response_raw

S_regulator_sz = S_regulator_sz.tocsr()
U_response_sz = U_response_sz.tocsr()

print("\nNormalized matrix shapes:")
print("U_response_sz:", U_response_sz.shape)
print("S_regulator_sz:", S_regulator_sz.shape)

# --------------------------------------------
# 5. Intervention sanity check
#
# For every perturbation q, its own regulator
# coordinate should remain exactly zero.
# --------------------------------------------

reg_to_col = {
    g: j
    for j, g in enumerate(regulator_genes_vc)
}

target_nonzero = {}

for q in final_perturbations_vc:
    rows = np.flatnonzero(
        condition_final_cells == q
    )

    j = reg_to_col[q]

    nnz = S_regulator_sz[rows, j].nnz

    if nnz > 0:
        target_nonzero[q] = int(nnz)

print(
    "\nPerturbation targets with nonzero "
    "normalized spliced counts:",
    len(target_nonzero)
)

if target_nonzero:
    print(target_nonzero)

print(
    "\nAll perturbation target coordinates zero:",
    len(target_nonzero) == 0
)


# ============================================================================
# NOTEBOOK INDEX 45: Cell 45. Build final binned trajectories
# ============================================================================

# ============================================
# Cell 45. Build final binned trajectories
# using official phase + VeloCycle scaling
# ============================================

import numpy as np

n_conditions = len(final_conditions_vc)
n_bins = 10
n_response = len(vc_genes)
n_regulators = len(regulator_genes_vc)

U_traj_vc = np.full(
    (n_conditions, n_bins, n_response),
    np.nan,
    dtype=np.float64
)

S_traj_vc = np.full(
    (n_conditions, n_bins, n_regulators),
    np.nan,
    dtype=np.float64
)

N_traj_vc = np.zeros(
    (n_conditions, n_bins),
    dtype=int
)

# Bin IDs for the final-cell subset
bin_final_cells = np.digitize(
    cycle_final_cells,
    bin_edges[1:-1],
    right=False
)

condition_to_row = {
    q: i
    for i, q in enumerate(final_conditions_vc)
}

# --------------------------------------------
# Aggregate condition x phase-bin means
# --------------------------------------------

for q in final_conditions_vc:

    qi = condition_to_row[q]

    q_rows = np.flatnonzero(
        condition_final_cells == q
    )

    q_bins = bin_final_cells[q_rows]

    for b in range(n_bins):

        local_rows = np.flatnonzero(q_bins == b)

        n = len(local_rows)
        N_traj_vc[qi, b] = n

        if n < 5:
            continue

        rows = q_rows[local_rows]

        U_traj_vc[qi, b, :] = np.asarray(
            U_response_sz[rows, :].mean(axis=0)
        ).ravel()

        S_traj_vc[qi, b, :] = np.asarray(
            S_regulator_sz[rows, :].mean(axis=0)
        ).ravel()


# --------------------------------------------
# Diagnostics
# --------------------------------------------

usable_mask_vc = N_traj_vc >= 5

print("U_traj_vc shape:", U_traj_vc.shape)
print("S_traj_vc shape:", S_traj_vc.shape)
print("N_traj_vc shape:", N_traj_vc.shape)

print(
    "\nUsable condition-bin pairs:",
    int(usable_mask_vc.sum()),
    "/",
    usable_mask_vc.size,
    "=",
    usable_mask_vc.mean()
)

usable_per_condition = usable_mask_vc.sum(axis=1)

print("\nUsable bins per condition:")
print(
    "min =", usable_per_condition.min(),
    "median =", np.median(usable_per_condition),
    "mean =", usable_per_condition.mean(),
    "max =", usable_per_condition.max()
)

print("\nDistribution:")
vals, counts = np.unique(
    usable_per_condition,
    return_counts=True
)

for v, c in zip(vals, counts):
    print(f"{v} bins: {c} conditions")

# --------------------------------------------
# Perturbation target trajectory check
# --------------------------------------------

bad_targets = []

for q in final_perturbations_vc:

    qi = condition_to_row[q]
    rj = reg_to_col[q]

    usable = usable_mask_vc[qi]

    vals = S_traj_vc[qi, usable, rj]

    if not np.all(vals == 0):
        bad_targets.append(q)

print(
    "\nPerturbation targets nonzero in "
    "binned S trajectories:",
    len(bad_targets)
)

if bad_targets:
    print(bad_targets)

print(
    "All target trajectories exactly zero:",
    len(bad_targets) == 0
)

# NaN sanity check:
# NaNs should occur only in bins with <5 cells
u_bad_nan = np.any(
    np.isnan(U_traj_vc[usable_mask_vc])
)

s_bad_nan = np.any(
    np.isnan(S_traj_vc[usable_mask_vc])
)

print("\nNaN inside usable U bins:", u_bad_nan)
print("NaN inside usable S bins:", s_bad_nan)


# ============================================================================
# NOTEBOOK INDEX 46: Cell 46. Build interval-level design matrix X
# ============================================================================

# ============================================
# Cell 46. Build interval-level design matrix X
# from adjacent usable phase bins
# ============================================

import numpy as np
import pandas as pd

interval_rows = []
X_rows = []

for q in final_conditions_vc:

    qi = condition_to_row[q]

    rho_q = float(rho_final.loc[q])
    dt_q = 0.1 / rho_q

    usable = usable_mask_vc[qi]

    # Only adjacent intervals:
    # 0->1, 1->2, ..., 8->9
    for a in range(9):

        b = a + 1

        if not (usable[a] and usable[b]):
            continue

        # ------------------------------------
        # Integral of regulator trajectory
        # using trapezoidal rule
        # ------------------------------------

        s_a = S_traj_vc[qi, a, :]
        s_b = S_traj_vc[qi, b, :]

        x_reg = (
            0.5 * dt_q * (s_a + s_b)
        )

        # ------------------------------------
        # Apply perturbation operator M_q
        #
        # For perturbation q, zero its own
        # regulator coordinate.
        # It should already be zero empirically,
        # but enforce M_q explicitly.
        # ------------------------------------

        if q != "non-targeting":
            rj = reg_to_col[q]
            x_reg = x_reg.copy()
            x_reg[rj] = 0.0

        # Unpenalized intercept integral:
        # ∫1 dt = dt
        x_row = np.concatenate(
            ([dt_q], x_reg)
        )

        X_rows.append(x_row)

        interval_rows.append(
            {
                "condition": q,
                "condition_idx": qi,
                "bin_a": a,
                "bin_b": b,
                "rho": rho_q,
                "delta_t": dt_q,
                "n_a": int(N_traj_vc[qi, a]),
                "n_b": int(N_traj_vc[qi, b]),
            }
        )


X_integral = np.vstack(X_rows)

interval_meta = pd.DataFrame(interval_rows)

print("Number of intervals:", len(interval_meta))
print("X_integral shape:", X_integral.shape)

print(
    "\nExpected columns = 1 intercept + regulators:",
    1 + len(regulator_genes_vc)
)

print("\nIntervals per condition:")
print(
    interval_meta.groupby("condition")
    .size()
    .describe()
)

print("\nDelta_t summary:")
print(interval_meta["delta_t"].describe())

# --------------------------------------------
# Sanity checks
# --------------------------------------------

print(
    "\nAny NaN in X:",
    bool(np.isnan(X_integral).any())
)

print(
    "Any inf in X:",
    bool(np.isinf(X_integral).any())
)

print(
    "All intercept columns equal delta_t:",
    np.allclose(
        X_integral[:, 0],
        interval_meta["delta_t"].to_numpy()
    )
)

# Check intervention coordinates again,
# now at the actual regression-design level
bad_Mq = []

for q in final_perturbations_vc:

    rows = np.flatnonzero(
        interval_meta["condition"].to_numpy() == q
    )

    if len(rows) == 0:
        bad_Mq.append((q, "no intervals"))
        continue

    j = 1 + reg_to_col[q]

    if not np.all(X_integral[rows, j] == 0):
        bad_Mq.append((q, "target column nonzero"))

print("\nM_q design violations:", len(bad_Mq))

if bad_Mq:
    print(bad_Mq[:20])

# Preview
print("\nFirst 10 interval metadata rows:")
print(interval_meta.head(10))


# ============================================================================
# NOTEBOOK INDEX 47: Cell 47. Build integral-response matrix Y
# ============================================================================

# ============================================
# Cell 47. Build integral-response matrix Y
# for all 426 response genes
# ============================================

import numpy as np
import pandas as pd

n_intervals = len(interval_meta)
n_response = len(vc_genes)

Y_integral = np.full(
    (n_intervals, n_response),
    np.nan,
    dtype=np.float64
)

# Keep the two components separately for QC
delta_U = np.full_like(Y_integral, np.nan)
beta_int_U = np.full_like(Y_integral, np.nan)

beta_vec = beta_cycle.loc[vc_genes].to_numpy(dtype=float)

for i, row in interval_meta.iterrows():

    qi = int(row["condition_idx"])
    a = int(row["bin_a"])
    b = int(row["bin_b"])
    dt = float(row["delta_t"])

    u_a = U_traj_vc[qi, a, :]
    u_b = U_traj_vc[qi, b, :]

    # Endpoint change
    du = u_b - u_a

    # Trapezoidal integral of u
    int_u = 0.5 * dt * (u_a + u_b)

    # beta_g * integral u_g dt
    beta_term = beta_vec * int_u

    delta_U[i, :] = du
    beta_int_U[i, :] = beta_term

    # Manuscript integral-equation response
    # h_g(q) = 0 in the current perturbation encoding
    Y_integral[i, :] = du + beta_term


# --------------------------------------------
# Basic diagnostics
# --------------------------------------------

print("Y_integral shape:", Y_integral.shape)

print(
    "Expected shape:",
    (len(interval_meta), len(vc_genes))
)

print("\nAny NaN in Y:",
      bool(np.isnan(Y_integral).any()))

print("Any inf in Y:",
      bool(np.isinf(Y_integral).any()))

print("\nY summary:")
print(
    pd.Series(Y_integral.ravel()).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nDelta-U component summary:")
print(
    pd.Series(delta_U.ravel()).describe(
        percentiles=[0.01, 0.50, 0.99]
    )
)

print("\nBeta-integral component summary:")
print(
    pd.Series(beta_int_U.ravel()).describe(
        percentiles=[0.01, 0.50, 0.99]
    )
)

# --------------------------------------------
# Per-gene scale diagnostics
# --------------------------------------------

y_rms = np.sqrt(
    np.mean(Y_integral ** 2, axis=0)
)

du_rms = np.sqrt(
    np.mean(delta_U ** 2, axis=0)
)

beta_rms = np.sqrt(
    np.mean(beta_int_U ** 2, axis=0)
)

response_scale_qc = pd.DataFrame(
    {
        "beta_cycle": beta_vec,
        "Y_rms": y_rms,
        "dU_rms": du_rms,
        "betaU_rms": beta_rms,
    },
    index=vc_genes
)

response_scale_qc["beta_to_dU_rms"] = (
    response_scale_qc["betaU_rms"]
    / np.maximum(
        response_scale_qc["dU_rms"],
        1e-12
    )
)

print("\nGenes with largest Y RMS:")
print(
    response_scale_qc
    .sort_values("Y_rms", ascending=False)
    .head(15)
)

print("\nGenes with largest beta_cycle:")
print(
    response_scale_qc
    .sort_values("beta_cycle", ascending=False)
    .head(15)
)

# --------------------------------------------
# Exact first-row sanity check
# --------------------------------------------

i = 0
qi = int(interval_meta.loc[i, "condition_idx"])
a = int(interval_meta.loc[i, "bin_a"])
b = int(interval_meta.loc[i, "bin_b"])
dt = float(interval_meta.loc[i, "delta_t"])

manual_first = (
    U_traj_vc[qi, b, :]
    - U_traj_vc[qi, a, :]
    + beta_vec
    * 0.5
    * dt
    * (
        U_traj_vc[qi, a, :]
        + U_traj_vc[qi, b, :]
    )
)

print(
    "\nFirst interval formula matches:",
    np.allclose(
        Y_integral[0],
        manual_first
    )
)


# ============================================================================
# NOTEBOOK INDEX 48: Cell 48. Build spliced trajectories for the
# ============================================================================

# ============================================
# Cell 48. Build spliced trajectories for the
# 426 response genes
# ============================================

import numpy as np
from scipy import sparse

# --------------------------------------------
# 1. Raw spliced counts for 426 responses
# --------------------------------------------

S_response_raw = adata.layers["spliced"][
    final_cell_idx, :
][:, response_idx].tocsr()

print("S_response_raw shape:", S_response_raw.shape)

# --------------------------------------------
# 2. Apply the same VeloCycle spliced
#    size normalization
# --------------------------------------------

S_response_sz = sparse.diags(
    s_factor
) @ S_response_raw

S_response_sz = S_response_sz.tocsr()

print("S_response_sz shape:", S_response_sz.shape)

# --------------------------------------------
# 3. Bin by final condition + official phase
# --------------------------------------------

S_response_traj_vc = np.full(
    (
        len(final_conditions_vc),
        10,
        len(vc_genes)
    ),
    np.nan,
    dtype=np.float64
)

for q in final_conditions_vc:

    qi = condition_to_row[q]

    q_rows = np.flatnonzero(
        condition_final_cells == q
    )

    q_bins = bin_final_cells[q_rows]

    for b in range(10):

        local_rows = np.flatnonzero(
            q_bins == b
        )

        if len(local_rows) < 5:
            continue

        rows = q_rows[local_rows]

        S_response_traj_vc[
            qi, b, :
        ] = np.asarray(
            S_response_sz[
                rows, :
            ].mean(axis=0)
        ).ravel()


# --------------------------------------------
# 4. Sanity checks
# --------------------------------------------

print(
    "\nS_response_traj_vc shape:",
    S_response_traj_vc.shape
)

print(
    "Any NaN inside usable bins:",
    bool(
        np.isnan(
            S_response_traj_vc[
                usable_mask_vc
            ]
        ).any()
    )
)

print(
    "Any inf inside usable bins:",
    bool(
        np.isinf(
            S_response_traj_vc[
                usable_mask_vc
            ]
        ).any()
    )
)

# Basic trajectory scale
usable_S = S_response_traj_vc[
    usable_mask_vc
]

print("\nBinned spliced response values:")
print(
    pd.Series(
        usable_S.ravel()
    ).describe(
        percentiles=[
            0.01, 0.05, 0.50,
            0.95, 0.99
        ]
    )
)


# ============================================================================
# NOTEBOOK INDEX 49: Cell 49. Kinetic consistency check using
# ============================================================================

# ============================================
# Cell 49. Kinetic consistency check using
# ds/dt = beta*u - gamma*s
# ============================================

import numpy as np
import pandas as pd

n_intervals = len(interval_meta)
n_response = len(vc_genes)

gamma_vec = gamma_cycle.loc[
    vc_genes
].to_numpy(dtype=float)

delta_S = np.full(
    (n_intervals, n_response),
    np.nan,
    dtype=np.float64
)

kinetic_rhs = np.full_like(
    delta_S,
    np.nan
)

for i, row in interval_meta.iterrows():

    qi = int(row["condition_idx"])
    a = int(row["bin_a"])
    b = int(row["bin_b"])
    dt = float(row["delta_t"])

    # Spliced trajectory
    s_a = S_response_traj_vc[qi, a, :]
    s_b = S_response_traj_vc[qi, b, :]

    # Unspliced trajectory
    u_a = U_traj_vc[qi, a, :]
    u_b = U_traj_vc[qi, b, :]

    # Observed change in spliced abundance
    ds = s_b - s_a

    # Trapezoidal integrals
    int_u = 0.5 * dt * (u_a + u_b)
    int_s = 0.5 * dt * (s_a + s_b)

    rhs = (
        beta_vec * int_u
        - gamma_vec * int_s
    )

    delta_S[i, :] = ds
    kinetic_rhs[i, :] = rhs


# --------------------------------------------
# Global diagnostics
# --------------------------------------------

print("delta_S shape:", delta_S.shape)
print("kinetic_rhs shape:", kinetic_rhs.shape)

print(
    "\nAny NaN:",
    bool(
        np.isnan(delta_S).any()
        or np.isnan(kinetic_rhs).any()
    )
)

print(
    "Any inf:",
    bool(
        np.isinf(delta_S).any()
        or np.isinf(kinetic_rhs).any()
    )
)

# Global correlation
global_corr = np.corrcoef(
    delta_S.ravel(),
    kinetic_rhs.ravel()
)[0, 1]

print(
    "\nGlobal correlation "
    "DeltaS vs kinetic RHS:",
    global_corr
)

# --------------------------------------------
# Per-gene diagnostics
# --------------------------------------------

rows = []

for j, g in enumerate(vc_genes):

    obs = delta_S[:, j]
    pred = kinetic_rhs[:, j]

    corr = np.corrcoef(obs, pred)[0, 1]

    rmse = np.sqrt(
        np.mean((obs - pred) ** 2)
    )

    obs_rms = np.sqrt(
        np.mean(obs ** 2)
    )

    pred_rms = np.sqrt(
        np.mean(pred ** 2)
    )

    rel_rmse = (
        rmse / max(obs_rms, 1e-12)
    )

    rows.append(
        {
            "gene": g,
            "beta_cycle": beta_vec[j],
            "gamma_cycle": gamma_vec[j],
            "corr": corr,
            "obs_dS_rms": obs_rms,
            "rhs_rms": pred_rms,
            "relative_rmse": rel_rmse,
        }
    )

kinetic_qc = pd.DataFrame(
    rows
).set_index("gene")


print("\nPer-gene correlation summary:")
print(
    kinetic_qc["corr"].describe(
        percentiles=[
            0.01, 0.05, 0.10,
            0.25, 0.50, 0.75,
            0.90, 0.95, 0.99
        ]
    )
)

print("\nPer-gene relative RMSE summary:")
print(
    kinetic_qc["relative_rmse"].describe(
        percentiles=[
            0.01, 0.05, 0.10,
            0.25, 0.50, 0.75,
            0.90, 0.95, 0.99
        ]
    )
)

print("\nGenes with largest beta:")
print(
    kinetic_qc
    .sort_values(
        "beta_cycle",
        ascending=False
    )
    .head(15)
)

print("\nGenes with best kinetic agreement:")
print(
    kinetic_qc
    .sort_values(
        "corr",
        ascending=False
    )
    .head(15)
)

print("\nGenes with worst kinetic agreement:")
print(
    kinetic_qc
    .sort_values(
        "corr",
        ascending=True
    )
    .head(15)
)


# ============================================================================
# NOTEBOOK INDEX 50: Cell 50. Inspect exact VeloCycle kinetic
# ============================================================================

# ============================================
# Cell 50. Inspect exact VeloCycle kinetic
# parameterization in the velocity model
# ============================================

from pathlib import Path

# Historical VeloCycle source that matches
# the successfully loaded official pickle
src_root = Path(
    "/home/featurize/work/project1/"
    "velocycle_91123be/velocycle"
)

keywords = [
    "log_betas",
    "log_gammas",
    "beta",
    "gamma",
    "ElogU",
    "ElogS",
    "count_factor",
    "velocity_coef",
    "νω",
    "omega",
]

for pyfile in src_root.rglob("*.py"):

    try:
        lines = pyfile.read_text().splitlines()
    except Exception:
        continue

    hit_idx = []

    for i, line in enumerate(lines):
        if any(k in line for k in keywords):
            hit_idx.append(i)

    if not hit_idx:
        continue

    print("\n" + "=" * 80)
    print(pyfile.relative_to(src_root))
    print("=" * 80)

    # Merge nearby hits into context blocks
    blocks = []

    for i in hit_idx:
        start = max(0, i - 5)
        end = min(len(lines), i + 8)

        if blocks and start <= blocks[-1][1]:
            blocks[-1] = (
                blocks[-1][0],
                max(blocks[-1][1], end)
            )
        else:
            blocks.append((start, end))

    for start, end in blocks:
        print(
            f"\n--- lines {start + 1}-{end} ---"
        )

        for j in range(start, end):
            print(
                f"{j + 1:4d}: {lines[j]}"
            )


# ============================================================================
# NOTEBOOK INDEX 51: Cell 51. Inspect the exact preprocessing /
# ============================================================================

# ============================================
# Cell 51. Inspect the exact preprocessing /
# velocity-fit call in the official Fig. 7
# 986-condition notebook
# ============================================

import json
from pathlib import Path

nb_path = Path(
    "/home/featurize/work/project1/ipynb_notebooks/"
    "Fig7_Replogle_Saunders_PerturbRPE1_"
    "MedGene_986conditions.ipynb"
)

with open(nb_path, "r") as f:
    nb = json.load(f)

keywords = [
    "prep_velocity",
    "velocity_fit",
    "velocity_inference",
    "normalize=",
    "count_factor",
    "S_sz",
    "U_sz",
    "data_to_fit",
]

for ci, cell in enumerate(nb["cells"]):

    if cell.get("cell_type") != "code":
        continue

    src = "".join(cell.get("source", []))

    if any(k in src for k in keywords):
        print("\n" + "=" * 80)
        print(f"CODE CELL {ci}")
        print("=" * 80)
        print(src)


# ============================================================================
# NOTEBOOK INDEX 52: Cell 52. Re-check VeloCycle kinetics
# ============================================================================

# ============================================
# Cell 52. Re-check VeloCycle kinetics
# on the correct RAW count scale
# ============================================

import numpy as np
import pandas as pd

# We should already have raw binned trajectories:
# S_response_traj_raw : condition x bin x gene
# U_response_traj_raw : condition x bin x gene
#
# interval_rows contains the same 1214 usable adjacent-bin intervals
# used previously.

delta_S_raw = []
kinetic_rhs_raw = []

for row in interval_rows.itertuples(index=False):

    q = int(row.condition_idx)
    a = int(row.bin_a)
    b = int(row.bin_b)

    dt = float(row.dt)

    S_a = S_response_traj_raw[q, a, :]
    S_b = S_response_traj_raw[q, b, :]

    U_a = U_response_traj_raw[q, a, :]
    U_b = U_response_traj_raw[q, b, :]

    # observed change in spliced counts
    dS = S_b - S_a

    # trapezoidal integral of beta*U - gamma*S
    rhs = dt * 0.5 * (
        beta_cycle * (U_a + U_b)
        - gamma_cycle * (S_a + S_b)
    )

    delta_S_raw.append(dS)
    kinetic_rhs_raw.append(rhs)

delta_S_raw = np.asarray(delta_S_raw)
kinetic_rhs_raw = np.asarray(kinetic_rhs_raw)

print("delta_S_raw shape:", delta_S_raw.shape)
print("kinetic_rhs_raw shape:", kinetic_rhs_raw.shape)

print("\nAny NaN:",
      np.isnan(delta_S_raw).any()
      or np.isnan(kinetic_rhs_raw).any())

print("Any inf:",
      np.isinf(delta_S_raw).any()
      or np.isinf(kinetic_rhs_raw).any())


# --------------------------------------------
# Global correlation
# --------------------------------------------

global_corr_raw = np.corrcoef(
    delta_S_raw.ravel(),
    kinetic_rhs_raw.ravel()
)[0, 1]

print(
    "\nGlobal correlation DeltaS vs kinetic RHS:",
    global_corr_raw
)


# --------------------------------------------
# Per-gene diagnostics
# --------------------------------------------

rows = []

for j, g in enumerate(response_genes_vc):

    obs = delta_S_raw[:, j]
    pred = kinetic_rhs_raw[:, j]

    if np.std(obs) > 0 and np.std(pred) > 0:
        corr = np.corrcoef(obs, pred)[0, 1]
    else:
        corr = np.nan

    obs_rms = np.sqrt(np.mean(obs**2))
    rhs_rms = np.sqrt(np.mean(pred**2))

    rmse = np.sqrt(np.mean((obs - pred)**2))

    relative_rmse = (
        rmse / obs_rms
        if obs_rms > 0
        else np.nan
    )

    rows.append({
        "gene": g,
        "beta_cycle": beta_cycle[j],
        "gamma_cycle": gamma_cycle[j],
        "corr": corr,
        "obs_dS_rms": obs_rms,
        "rhs_rms": rhs_rms,
        "relative_rmse": relative_rmse,
    })

kinetic_check_raw = (
    pd.DataFrame(rows)
    .set_index("gene")
)

print("\nPer-gene correlation summary:")
print(
    kinetic_check_raw["corr"].describe(
        percentiles=[
            0.01, 0.05, 0.10, 0.25,
            0.50, 0.75, 0.90, 0.95, 0.99
        ]
    )
)

print("\nPer-gene relative RMSE summary:")
print(
    kinetic_check_raw["relative_rmse"].describe(
        percentiles=[
            0.01, 0.05, 0.10, 0.25,
            0.50, 0.75, 0.90, 0.95, 0.99
        ]
    )
)

print("\nGenes with best kinetic agreement:")
print(
    kinetic_check_raw
    .sort_values("corr", ascending=False)
    .head(15)
)

print("\nGenes with largest beta:")
print(
    kinetic_check_raw
    .sort_values("beta_cycle", ascending=False)
    .head(15)
)


# ============================================================================
# NOTEBOOK INDEX 53: Cell 52b. Inspect interval_rows structure
# ============================================================================

# ============================================
# Cell 52b. Inspect interval_rows structure
# ============================================

print("type(interval_rows):", type(interval_rows))
print("number of intervals:", len(interval_rows))

print("\nFirst interval:")
print(interval_rows[0])

print("\nType of first interval:")
print(type(interval_rows[0]))

# If dict-like, show keys
if isinstance(interval_rows[0], dict):
    print("\nKeys:")
    print(interval_rows[0].keys())

# If tuple/list-like, show length
if isinstance(interval_rows[0], (tuple, list)):
    print("\nLength of first interval:")
    print(len(interval_rows[0]))


# ============================================================================
# NOTEBOOK INDEX 54: Cell 52c. Build RAW-count trajectories
# ============================================================================

# ============================================
# Cell 52c. Build RAW-count trajectories
# for the 426 response genes
# ============================================

import numpy as np

U_response_traj_raw = np.full(
    (
        len(final_conditions_vc),
        10,
        len(vc_genes)
    ),
    np.nan,
    dtype=np.float64
)

S_response_traj_raw = np.full_like(
    U_response_traj_raw,
    np.nan
)

for q in final_conditions_vc:

    qi = condition_to_row[q]

    # Row positions within the 57,172 final cells
    q_rows = np.flatnonzero(
        condition_final_cells == q
    )

    q_bins = bin_final_cells[q_rows]

    for b in range(10):

        local_rows = np.flatnonzero(
            q_bins == b
        )

        if len(local_rows) < 5:
            continue

        rows = q_rows[local_rows]

        # RAW unspliced mean
        U_response_traj_raw[
            qi, b, :
        ] = np.asarray(
            U_response_raw[
                rows, :
            ].mean(axis=0)
        ).ravel()

        # RAW spliced mean
        S_response_traj_raw[
            qi, b, :
        ] = np.asarray(
            S_response_raw[
                rows, :
            ].mean(axis=0)
        ).ravel()


print(
    "U_response_traj_raw shape:",
    U_response_traj_raw.shape
)

print(
    "S_response_traj_raw shape:",
    S_response_traj_raw.shape
)

print(
    "\nAny NaN in usable U bins:",
    bool(
        np.isnan(
            U_response_traj_raw[
                usable_mask_vc
            ]
        ).any()
    )
)

print(
    "Any NaN in usable S bins:",
    bool(
        np.isnan(
            S_response_traj_raw[
                usable_mask_vc
            ]
        ).any()
    )
)

print(
    "Any inf in usable U bins:",
    bool(
        np.isinf(
            U_response_traj_raw[
                usable_mask_vc
            ]
        ).any()
    )
)

print(
    "Any inf in usable S bins:",
    bool(
        np.isinf(
            S_response_traj_raw[
                usable_mask_vc
            ]
        ).any()
    )
)

print("\nRaw U mean over usable bins:")
print(
    np.nanmean(
        U_response_traj_raw[
            usable_mask_vc
        ]
    )
)

print("Raw S mean over usable bins:")
print(
    np.nanmean(
        S_response_traj_raw[
            usable_mask_vc
        ]
    )
)


# ============================================================================
# NOTEBOOK INDEX 55: Cell 53. Kinetic consistency check
# ============================================================================

# ============================================
# Cell 53. Kinetic consistency check
# on RAW count trajectories
# ============================================

import numpy as np
import pandas as pd

n_intervals = len(interval_rows)
n_response = len(vc_genes)

delta_S_raw = np.full(
    (n_intervals, n_response),
    np.nan,
    dtype=np.float64
)

kinetic_rhs_raw = np.full_like(
    delta_S_raw,
    np.nan
)

for i, row in enumerate(interval_rows):

    qi = int(row["condition_idx"])
    a = int(row["bin_a"])
    b = int(row["bin_b"])
    dt = float(row["delta_t"])

    s_a = S_response_traj_raw[qi, a, :]
    s_b = S_response_traj_raw[qi, b, :]

    u_a = U_response_traj_raw[qi, a, :]
    u_b = U_response_traj_raw[qi, b, :]

    # Observed change in raw spliced counts
    ds = s_b - s_a

    # Trapezoidal integrals
    int_u = 0.5 * dt * (u_a + u_b)
    int_s = 0.5 * dt * (s_a + s_b)

    rhs = (
        beta_vec * int_u
        - gamma_vec * int_s
    )

    delta_S_raw[i, :] = ds
    kinetic_rhs_raw[i, :] = rhs


# --------------------------------------------
# Basic checks
# --------------------------------------------

print("delta_S_raw shape:", delta_S_raw.shape)
print("kinetic_rhs_raw shape:", kinetic_rhs_raw.shape)

print(
    "\nAny NaN:",
    bool(
        np.isnan(delta_S_raw).any()
        or np.isnan(kinetic_rhs_raw).any()
    )
)

print(
    "Any inf:",
    bool(
        np.isinf(delta_S_raw).any()
        or np.isinf(kinetic_rhs_raw).any()
    )
)


# --------------------------------------------
# Global correlation
# --------------------------------------------

global_corr_raw = np.corrcoef(
    delta_S_raw.ravel(),
    kinetic_rhs_raw.ravel()
)[0, 1]

print(
    "\nGlobal correlation "
    "DeltaS vs kinetic RHS:",
    global_corr_raw
)


# --------------------------------------------
# Per-gene diagnostics
# --------------------------------------------

rows = []

for j, g in enumerate(vc_genes):

    obs = delta_S_raw[:, j]
    pred = kinetic_rhs_raw[:, j]

    if (
        np.std(obs) > 0
        and np.std(pred) > 0
    ):
        corr = np.corrcoef(
            obs,
            pred
        )[0, 1]
    else:
        corr = np.nan

    obs_rms = np.sqrt(
        np.mean(obs ** 2)
    )

    rhs_rms = np.sqrt(
        np.mean(pred ** 2)
    )

    rmse = np.sqrt(
        np.mean(
            (obs - pred) ** 2
        )
    )

    rel_rmse = (
        rmse
        / max(obs_rms, 1e-12)
    )

    rows.append(
        {
            "gene": g,
            "beta_cycle": beta_vec[j],
            "gamma_cycle": gamma_vec[j],
            "corr": corr,
            "obs_dS_rms": obs_rms,
            "rhs_rms": rhs_rms,
            "relative_rmse": rel_rmse,
        }
    )

kinetic_qc_raw = pd.DataFrame(
    rows
).set_index("gene")


print("\nPer-gene correlation summary:")
print(
    kinetic_qc_raw["corr"].describe(
        percentiles=[
            0.01, 0.05, 0.10,
            0.25, 0.50, 0.75,
            0.90, 0.95, 0.99
        ]
    )
)

print("\nPer-gene relative RMSE summary:")
print(
    kinetic_qc_raw[
        "relative_rmse"
    ].describe(
        percentiles=[
            0.01, 0.05, 0.10,
            0.25, 0.50, 0.75,
            0.90, 0.95, 0.99
        ]
    )
)

print("\nGenes with largest beta:")
print(
    kinetic_qc_raw
    .sort_values(
        "beta_cycle",
        ascending=False
    )
    .head(15)
)

print("\nGenes with best kinetic agreement:")
print(
    kinetic_qc_raw
    .sort_values(
        "corr",
        ascending=False
    )
    .head(15)
)


# ============================================================================
# NOTEBOOK INDEX 56: Cell 54. Diagnose VeloCycle's global
# ============================================================================

# ============================================
# Cell 54. Diagnose VeloCycle's global
# kinetic time-scale normalization
# ============================================

import numpy as np
import pandas as pd

# Raw VeloCycle kinetic parameters
beta_raw = np.exp(
    beta_series.loc[vc_genes].to_numpy(dtype=float)
)

gamma_raw = np.exp(
    gamma_series.loc[vc_genes].to_numpy(dtype=float)
)

# Official notebook's global gamma scale
gamma_geom = np.exp(
    np.mean(
        gamma_series.loc[vc_genes].to_numpy(dtype=float)
    )
)

# Official notebook displays omega / gamma_geom
omega_nt_over_gamma = (
    omega_nt / gamma_geom
)

# Our previous conversion
old_time_factor = (
    2.0 * np.pi / omega_nt
)

# Alternative dimensionless kinetics using
# gamma_geom as the global time unit
beta_over_gamma_geom = (
    beta_raw / gamma_geom
)

gamma_over_gamma_geom = (
    gamma_raw / gamma_geom
)

print("Global geometric-mean gamma:")
print(gamma_geom)

print("\nNT omega:")
print(omega_nt)

print("\nOfficial-style NT omega / gamma_geom:")
print(omega_nt_over_gamma)

print("\nPrevious 2pi / omega_NT factor:")
print(old_time_factor)


print("\nRaw beta summary:")
print(
    pd.Series(beta_raw).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nRaw gamma summary:")
print(
    pd.Series(gamma_raw).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nBeta / gamma_geom summary:")
print(
    pd.Series(
        beta_over_gamma_geom
    ).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nGamma / gamma_geom summary:")
print(
    pd.Series(
        gamma_over_gamma_geom
    ).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)


# Inspect the problematic genes directly
inspect_genes = [
    "YBX1",
    "RPL23",
    "RPS27L",
    "RHOA",
    "ACTB",
    "MKI67",
    "CENPE",
    "ASPM",
]

rows = []

for g in inspect_genes:

    j = vc_genes.index(g)

    rows.append({
        "gene": g,
        "beta_raw": beta_raw[j],
        "gamma_raw": gamma_raw[j],
        "beta/gamma_geom":
            beta_over_gamma_geom[j],
        "gamma/gamma_geom":
            gamma_over_gamma_geom[j],
        "old_beta_cycle":
            beta_vec[j],
        "old_gamma_cycle":
            gamma_vec[j],
        "kinetic_corr_raw":
            kinetic_qc_raw.loc[g, "corr"],
        "relative_rmse_raw":
            kinetic_qc_raw.loc[
                g,
                "relative_rmse"
            ],
    })

scale_diagnostic = (
    pd.DataFrame(rows)
    .set_index("gene")
)

print("\nSelected genes:")
print(scale_diagnostic)


# ============================================================================
# NOTEBOOK INDEX 57: Cell 55. Audit kinetic variables directly
# ============================================================================

# ============================================
# Cell 55. Audit kinetic variables directly
# from the official compact checkpoint
# ============================================

import numpy as np
import pandas as pd

# Directly from official extracted VeloCycle output
log_beta_vc = np.asarray(
    vc["log_betas"],
    dtype=np.float64
).squeeze()

log_gamma_vc = np.asarray(
    vc["log_gammas"],
    dtype=np.float64
).squeeze()

print("log_beta_vc shape:", log_beta_vc.shape)
print("log_gamma_vc shape:", log_gamma_vc.shape)

print("\nOfficial log_beta summary:")
print(
    pd.Series(log_beta_vc).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nOfficial log_gamma summary:")
print(
    pd.Series(log_gamma_vc).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

# Exponentiate exactly once
beta_vc_direct = np.exp(log_beta_vc)
gamma_vc_direct = np.exp(log_gamma_vc)

print("\nexp(log_beta) summary:")
print(
    pd.Series(beta_vc_direct).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

print("\nexp(log_gamma) summary:")
print(
    pd.Series(gamma_vc_direct).describe(
        percentiles=[
            0.01, 0.05, 0.25,
            0.50, 0.75, 0.95, 0.99
        ]
    )
)

# Compare against the vectors actually used in Cell 49/53
expected_beta_cycle = (
    (2.0 * np.pi / omega_nt)
    * beta_vc_direct
)

expected_gamma_cycle = (
    (2.0 * np.pi / omega_nt)
    * gamma_vc_direct
)

print("\nCompare to beta_vec used in kinetic test:")
print(
    "max abs difference:",
    np.max(
        np.abs(
            expected_beta_cycle
            - beta_vec
        )
    )
)

print("\nCompare to gamma_vec used in kinetic test:")
print(
    "max abs difference:",
    np.max(
        np.abs(
            expected_gamma_cycle
            - gamma_vec
        )
    )
)

# Correct official global gamma scale
gamma_geom_direct = np.exp(
    np.mean(log_gamma_vc)
)

print("\nCorrect exp(mean(log_gamma)):")
print(gamma_geom_direct)

print("\nOfficial-style omega_NT / gamma_geom:")
print(
    omega_nt / gamma_geom_direct
)


# ============================================================================
# NOTEBOOK INDEX 58: Cell 56. Inspect compact checkpoint contents
# ============================================================================

# ============================================
# Cell 56. Inspect compact checkpoint contents
# and currently available VeloCycle objects
# ============================================

print("Compact checkpoint keys:")
print(vc.files)

print("\nRelevant variables currently in memory:")

names_to_check = [
    "cycle_pyro",
    "cycle_pyro_ref",
    "velocity_fit",
    "phase_fit",
    "data_to_fit",
]

for name in names_to_check:
    if name in globals():
        obj = globals()[name]
        print(
            f"{name:20s}",
            type(obj),
            getattr(obj, "shape", "")
        )
    else:
        print(
            f"{name:20s}",
            "NOT IN MEMORY"
        )

# Inspect any cycle-like arrays in compact checkpoint
print("\nCheckpoint arrays:")

for key in vc.files:
    arr = vc[key]
    print(
        f"{key:20s}",
        "shape =", arr.shape,
        "dtype =", arr.dtype
    )


# ============================================================================
# NOTEBOOK INDEX 59: Cell 57. Locate where VeloCycle stores
# ============================================================================

# ============================================
# Cell 57. Locate where VeloCycle stores
# Fourier/cycle coefficients in fitted objects
# ============================================

from pathlib import Path

src_dir = Path(
    "/home/featurize/work/project1/"
    "velocycle_91123be/velocycle"
)

patterns = [
    "self.cycle_pyro",
    "cycle_pyro =",
    "cycle_pyro=",
    "self.cycle",
    "cycle_obj",
    ".means",
    "ν",
]

for pyfile in sorted(src_dir.glob("*.py")):

    text = pyfile.read_text(
        errors="ignore"
    )

    hits = []

    for lineno, line in enumerate(
        text.splitlines(),
        start=1
    ):
        if any(p in line for p in patterns):
            hits.append(
                (lineno, line)
            )

    if hits:
        print(
            "\n" + "=" * 80
        )
        print(pyfile.name)
        print("=" * 80)

        for lineno, line in hits:
            print(
                f"{lineno:5d}: {line}"
            )


# ============================================================================
# NOTEBOOK INDEX 60: Cell 58. Extract compact latent VeloCycle
# ============================================================================

# ============================================
# Cell 58. Extract compact latent VeloCycle
# quantities from the official 986-condition
# LargeGene pickle
# ============================================

import subprocess
from pathlib import Path

python_official = (
    "/environment/miniconda3/envs/"
    "velocycle-official/bin/python"
)

pkl_path = (
    "/home/featurize/work/project1/"
    "pickle_result_outputs/"
    "Replogle_Saunders_PerturbRPE1_75cells_"
    "phase_velocity_data_fit_LargeGene_986conditions.pkl.gz"
)

out_path = (
    "/home/featurize/work/project1/"
    "velocycle_986_latent_extract.npz"
)

worktree = (
    "/home/featurize/work/project1/"
    "velocycle_91123be"
)

script = r'''
import sys
import io
import gzip
import pickle
import numpy as np
import torch

pkl_path = sys.argv[1]
out_path = sys.argv[2]
worktree = sys.argv[3]

sys.path.insert(0, worktree)

# Force CUDA-saved tensors onto CPU
_original_load_from_bytes = torch.storage._load_from_bytes

torch.storage._load_from_bytes = (
    lambda b:
    torch.load(
        io.BytesIO(b),
        map_location="cpu"
    )
)

print("Opening pickle...")

with gzip.open(pkl_path, "rb") as f:
    phase_fit, velocity_fit, data_to_fit = pickle.load(f)

print("Loaded.")

payload = {}

# ------------------------------------------------
# Velocity-fit Fourier coefficients
# ------------------------------------------------

if hasattr(velocity_fit, "fourier_coef"):
    x = np.asarray(
        velocity_fit.fourier_coef
    )
    payload["velocity_fourier_coef"] = x

    print(
        "velocity_fourier_coef:",
        x.shape,
        x.dtype
    )

if hasattr(velocity_fit, "cycle_pyro"):
    cp = velocity_fit.cycle_pyro

    if hasattr(cp, "means"):
        payload[
            "velocity_cycle_means"
        ] = cp.means.to_numpy()

        payload[
            "velocity_cycle_index"
        ] = np.asarray(
            cp.means.index,
            dtype=object
        )

        payload[
            "velocity_cycle_genes"
        ] = np.asarray(
            cp.means.columns,
            dtype=object
        )

        print(
            "velocity_cycle_means:",
            cp.means.shape
        )


# ------------------------------------------------
# Phase-fit Fourier coefficients
# ------------------------------------------------

if hasattr(phase_fit, "fourier_coef"):
    x = np.asarray(
        phase_fit.fourier_coef
    )
    payload["phase_fourier_coef"] = x

    print(
        "phase_fourier_coef:",
        x.shape,
        x.dtype
    )

if hasattr(phase_fit, "cycle_pyro"):
    cp = phase_fit.cycle_pyro

    if hasattr(cp, "means"):
        payload[
            "phase_cycle_means"
        ] = cp.means.to_numpy()

        payload[
            "phase_cycle_index"
        ] = np.asarray(
            cp.means.index,
            dtype=object
        )

        payload[
            "phase_cycle_genes"
        ] = np.asarray(
            cp.means.columns,
            dtype=object
        )

        print(
            "phase_cycle_means:",
            cp.means.shape
        )


# ------------------------------------------------
# Count factor used by the phase model
# ------------------------------------------------

if (
    hasattr(phase_fit, "metaparams")
    and phase_fit.metaparams is not None
    and hasattr(
        phase_fit.metaparams,
        "count_factor"
    )
):
    cf = (
        phase_fit.metaparams
        .count_factor
        .detach()
        .cpu()
        .numpy()
    )

    payload["count_factor"] = cf

    print(
        "count_factor:",
        cf.shape,
        cf.dtype,
        "min =", np.nanmin(cf),
        "max =", np.nanmax(cf)
    )

else:
    print(
        "phase_fit.metaparams.count_factor "
        "NOT AVAILABLE"
    )


# ------------------------------------------------
# Cell / gene identifiers for alignment
# ------------------------------------------------

payload["cell_ids"] = np.asarray(
    data_to_fit.obs_names,
    dtype=object
)

payload["gene_names"] = np.asarray(
    data_to_fit.var_names,
    dtype=object
)

print(
    "data_to_fit:",
    data_to_fit.shape
)

np.savez_compressed(
    out_path,
    **payload
)

print("\nSaved:")
print(out_path)

print("\nSaved keys:")
print(list(payload.keys()))
'''

result = subprocess.run(
    [
        python_official,
        "-c",
        script,
        pkl_path,
        out_path,
        worktree,
    ],
    text=True,
    capture_output=True,
)

print(result.stdout)

if result.stderr:
    print("STDERR:")
    print(result.stderr)

print(
    "\nReturn code:",
    result.returncode
)

print(
    "Output exists:",
    Path(out_path).exists()
)

if Path(out_path).exists():
    print(
        "Output size MB:",
        Path(out_path).stat().st_size
        / 1024**2
    )


# ============================================================================
# NOTEBOOK INDEX 61: Cell 59. Inspect exact count_factor
# ============================================================================

# ============================================
# Cell 59. Inspect exact count_factor
# construction in phase preprocessing
# ============================================

from pathlib import Path

pyfile = Path(
    "/home/featurize/work/project1/"
    "velocycle_91123be/velocycle/"
    "preprocessing.py"
)

lines = pyfile.read_text(
    errors="ignore"
).splitlines()

# Print the phase-preprocessing block with
# enough surrounding context
for start, end in [
    (95, 205),
]:
    print(
        f"\n{'=' * 80}\n"
        f"{pyfile.name}: lines {start}-{end}\n"
        f"{'=' * 80}"
    )

    for i in range(start, end + 1):
        if i <= len(lines):
            print(
                f"{i:4d}: {lines[i - 1]}"
            )


# ============================================================================
# NOTEBOOK INDEX 62: Cell 60. Inspect the exact Fourier basis
# ============================================================================

# ============================================
# Cell 60. Inspect the exact Fourier basis
# zeta(phi) and its derivative
# ============================================

from pathlib import Path

src_dir = Path(
    "/home/featurize/work/project1/"
    "velocycle_91123be/velocycle"
)

search_terms = [
    "ζ",
    "zeta",
    "fourier",
    "cos(",
    "sin(",
    "ζ_dϕ",
]

for pyfile in sorted(src_dir.glob("*.py")):

    lines = pyfile.read_text(
        errors="ignore"
    ).splitlines()

    hits = []

    for i, line in enumerate(lines, start=1):
        if any(term in line for term in search_terms):
            hits.append(i)

    if not hits:
        continue

    # Print compact neighborhoods around relevant hits
    printed = set()

    for hit in hits:

        start = max(1, hit - 4)
        end = min(len(lines), hit + 8)

        key = (start, end)

        if key in printed:
            continue

        printed.add(key)

        print("\n" + "=" * 80)
        print(
            f"{pyfile.name}: "
            f"lines {start}-{end}"
        )
        print("=" * 80)

        for j in range(start, end + 1):
            print(
                f"{j:5d}: {lines[j - 1]}"
            )


# ============================================================================
# NOTEBOOK INDEX 63: Cell 61. Exact latent kinetic identity check
# ============================================================================

# ============================================
# Cell 61. Exact latent kinetic identity check
# inside the VeloCycle parameterization
# ============================================

import numpy as np
import pandas as pd

latent_path = (
    "/home/featurize/work/project1/"
    "velocycle_986_latent_extract.npz"
)

lat = np.load(
    latent_path,
    allow_pickle=True
)

# --------------------------------------------
# Align genes
# --------------------------------------------

coef = np.asarray(
    lat["velocity_fourier_coef"],
    dtype=np.float64
)  # shape: 3 x 426

coef_genes = np.asarray(
    lat["velocity_cycle_genes"],
    dtype=object
)

vc_gene_names = np.asarray(
    vc["gene_names"],
    dtype=object
)

print("Fourier coef shape:", coef.shape)
print(
    "Gene order identical:",
    np.array_equal(
        coef_genes,
        vc_gene_names
    )
)

# Reorder defensively if needed
coef_lookup = {
    g: j
    for j, g in enumerate(coef_genes)
}

coef = np.column_stack([
    coef[:, coef_lookup[g]]
    for g in vc_gene_names
])


# --------------------------------------------
# Official kinetics: exponentiate ONCE
# --------------------------------------------

beta = np.exp(
    np.asarray(
        vc["log_betas"],
        dtype=np.float64
    )
)

gamma = np.exp(
    np.asarray(
        vc["log_gammas"],
        dtype=np.float64
    )
)


# --------------------------------------------
# Dense phase grid
#
# VeloCycle basis for one harmonic:
#
#   zeta(phi)  = [1, sin(phi), cos(phi)]
#   zeta'(phi) = [0, cos(phi), -sin(phi)]
# --------------------------------------------

phi_grid = np.linspace(
    0.0,
    2.0 * np.pi,
    721,
    endpoint=False
)

zeta = np.vstack([
    np.ones_like(phi_grid),
    np.sin(phi_grid),
    np.cos(phi_grid),
])

zeta_dphi = np.vstack([
    np.zeros_like(phi_grid),
    np.cos(phi_grid),
    -np.sin(phi_grid),
])


# --------------------------------------------
# log S and d(log S)/dphi
#
# count_factor is omitted here deliberately:
# it is phase-independent, so its derivative
# is exactly zero.
# --------------------------------------------

logS = coef.T @ zeta
dlogS_dphi = coef.T @ zeta_dphi

S_latent = np.exp(logS)


# --------------------------------------------
# Check every FINAL condition
# --------------------------------------------

condition_names_all = np.asarray(
    vc["condition_names"],
    dtype=object
)

speed_all = np.asarray(
    vc["speed_raw"],
    dtype=np.float64
)

speed_lookup = {
    q: speed_all[i]
    for i, q in enumerate(condition_names_all)
}

records = []

eps = 1e-5

for q in final_conditions_vc:

    omega = float(
        speed_lookup[q]
    )

    # v = omega * d(logS)/dphi
    v = omega * dlogS_dphi

    inside = (
        v
        + gamma[:, None]
    )

    active = inside > 0.0

    # Exact VeloCycle latent U
    U_latent = (
        S_latent
        / beta[:, None]
        * (
            np.maximum(inside, 0.0)
            + eps
        )
    )

    # Left side of spliced ODE in phase time
    lhs = (
        omega
        * dlogS_dphi
        * S_latent
    )

    # Right side
    rhs = (
        beta[:, None]
        * U_latent
        - gamma[:, None]
        * S_latent
    )

    # On non-clipped points, discrepancy should
    # be only the +1e-5 numerical stabilizer.
    err = rhs - lhs

    scale = np.maximum(
        np.abs(lhs),
        1e-12
    )

    records.append({
        "condition": q,
        "omega": omega,
        "relu_active_fraction":
            active.mean(),
        "relu_clipped_fraction":
            1.0 - active.mean(),
        "max_abs_error_active":
            np.max(
                np.abs(err[active])
            ),
        "median_relative_error_active":
            np.median(
                np.abs(err[active])
                / scale[active]
            ),
    })


latent_identity_qc = (
    pd.DataFrame(records)
    .set_index("condition")
)

print(
    "\nCondition-level latent identity summary:"
)

print(
    latent_identity_qc.describe()
)


# --------------------------------------------
# Gene-level clipping frequency
# across all final conditions and phases
# --------------------------------------------

clip_counts = np.zeros(
    len(vc_gene_names),
    dtype=np.float64
)

total_points = (
    len(final_conditions_vc)
    * len(phi_grid)
)

for q in final_conditions_vc:

    omega = float(
        speed_lookup[q]
    )

    inside = (
        omega
        * dlogS_dphi
        + gamma[:, None]
    )

    clip_counts += (
        inside <= 0.0
    ).sum(axis=1)

gene_clip_fraction = (
    clip_counts
    / total_points
)

latent_gene_qc = pd.DataFrame({
    "gene": vc_gene_names,
    "beta": beta,
    "gamma": gamma,
    "clip_fraction":
        gene_clip_fraction,
}).set_index("gene")


print(
    "\nGenes with largest ReLU clipping fraction:"
)

print(
    latent_gene_qc
    .sort_values(
        "clip_fraction",
        ascending=False
    )
    .head(20)
)


print(
    "\nSelected problematic / good genes:"
)

inspect = [
    "YBX1",
    "RPL23",
    "RPS27L",
    "RHOA",
    "ACTB",
    "MKI67",
    "CENPE",
    "ASPM",
]

print(
    latent_gene_qc.loc[
        [
            g for g in inspect
            if g in latent_gene_qc.index
        ]
    ]
)


# ============================================================================
# NOTEBOOK INDEX 64: Cell 62. Build VeloCycle latent trajectories
# ============================================================================

# ============================================
# Cell 62. Build VeloCycle latent trajectories
# on the 10-bin grid for final conditions
# ============================================

import numpy as np

lat = np.load(
    "/home/featurize/work/project1/"
    "velocycle_986_latent_extract.npz",
    allow_pickle=True
)

coef = np.asarray(
    lat["velocity_fourier_coef"],
    dtype=np.float64
)

coef_genes = np.asarray(
    lat["velocity_cycle_genes"],
    dtype=object
)

# Align to vc gene order
vc_gene_names = np.asarray(
    vc["gene_names"],
    dtype=object
)

coef_lookup = {
    g: j
    for j, g in enumerate(coef_genes)
}

coef = np.column_stack([
    coef[:, coef_lookup[g]]
    for g in vc_gene_names
])

beta = np.exp(
    np.asarray(
        vc["log_betas"],
        dtype=np.float64
    )
)

gamma = np.exp(
    np.asarray(
        vc["log_gammas"],
        dtype=np.float64
    )
)

# 10 bin centers in phase angle
phi_bins = (
    2.0
    * np.pi
    * np.asarray(bin_centers)
)

# VeloCycle one-harmonic basis:
# [1, sin(phi), cos(phi)]
zeta = np.vstack([
    np.ones_like(phi_bins),
    np.sin(phi_bins),
    np.cos(phi_bins),
])

zeta_dphi = np.vstack([
    np.zeros_like(phi_bins),
    np.cos(phi_bins),
    -np.sin(phi_bins),
])

# Gene x bin
logS_latent_base = (
    coef.T @ zeta
)

dlogS_dphi = (
    coef.T @ zeta_dphi
)

# Canonical library-size offset:
# count_factor = 0
S_latent_base = np.exp(
    logS_latent_base
)

n_cond = len(final_conditions_vc)
n_bins = len(bin_centers)
n_genes = len(vc_gene_names)

S_latent_traj = np.full(
    (n_cond, n_bins, n_genes),
    np.nan,
    dtype=np.float64
)

U_latent_traj = np.full_like(
    S_latent_traj,
    np.nan
)

speed_lookup = {
    q: float(speed_series.loc[q])
    for q in final_conditions_vc
}

for qi, q in enumerate(
    final_conditions_vc
):

    omega = speed_lookup[q]

    inside = (
        omega * dlogS_dphi
        + gamma[:, None]
    )

    U_base = (
        S_latent_base
        / beta[:, None]
        * (
            np.maximum(
                inside,
                0.0
            )
            + 1e-5
        )
    )

    S_latent_traj[
        qi, :, :
    ] = S_latent_base.T

    U_latent_traj[
        qi, :, :
    ] = U_base.T


print(
    "S_latent_traj shape:",
    S_latent_traj.shape
)

print(
    "U_latent_traj shape:",
    U_latent_traj.shape
)

print(
    "\nAny NaN:",
    np.isnan(
        S_latent_traj
    ).any()
    or np.isnan(
        U_latent_traj
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        S_latent_traj
    ).any()
    or np.isinf(
        U_latent_traj
    ).any()
)

print(
    "\nS latent range:",
    np.min(S_latent_traj),
    np.median(S_latent_traj),
    np.max(S_latent_traj)
)

print(
    "U latent range:",
    np.min(U_latent_traj),
    np.median(U_latent_traj),
    np.max(U_latent_traj)
)


# ============================================================================
# NOTEBOOK INDEX 65: Cell 62b. Restore fixed 10-bin phase grid
# ============================================================================

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


# ============================================================================
# NOTEBOOK INDEX 66: Cell 63. Build VeloCycle latent trajectories
# ============================================================================

# ============================================
# Cell 63. Build VeloCycle latent trajectories
# on the fixed 10-bin grid
# ============================================

import numpy as np

lat = np.load(
    "/home/featurize/work/project1/"
    "velocycle_986_latent_extract.npz",
    allow_pickle=True
)

coef = np.asarray(
    lat["velocity_fourier_coef"],
    dtype=np.float64
)

coef_genes = np.asarray(
    lat["velocity_cycle_genes"],
    dtype=object
)

vc_gene_names = np.asarray(
    vc["gene_names"],
    dtype=object
)

# Align Fourier coefficients to vc gene order
coef_lookup = {
    g: j
    for j, g in enumerate(coef_genes)
}

coef = np.column_stack([
    coef[:, coef_lookup[g]]
    for g in vc_gene_names
])

beta = np.exp(
    np.asarray(
        vc["log_betas"],
        dtype=np.float64
    )
)

gamma = np.exp(
    np.asarray(
        vc["log_gammas"],
        dtype=np.float64
    )
)

# Phase angles at the 10 bin centers
phi_bins = (
    2.0
    * np.pi
    * bin_centers
)

# VeloCycle basis:
# [1, sin(phi), cos(phi)]
zeta = np.vstack([
    np.ones_like(phi_bins),
    np.sin(phi_bins),
    np.cos(phi_bins),
])

zeta_dphi = np.vstack([
    np.zeros_like(phi_bins),
    np.cos(phi_bins),
    -np.sin(phi_bins),
])

# Gene x bin
logS_latent_base = coef.T @ zeta
dlogS_dphi = coef.T @ zeta_dphi

# Canonical count_factor = 0
S_latent_base = np.exp(
    logS_latent_base
)

n_cond = len(final_conditions_vc)
n_bins = len(bin_centers)
n_genes = len(vc_gene_names)

S_latent_traj = np.empty(
    (n_cond, n_bins, n_genes),
    dtype=np.float64
)

U_latent_traj = np.empty_like(
    S_latent_traj
)

speed_lookup = {
    q: float(speed_series.loc[q])
    for q in final_conditions_vc
}

for qi, q in enumerate(
    final_conditions_vc
):

    omega = speed_lookup[q]

    inside = (
        omega
        * dlogS_dphi
        + gamma[:, None]
    )

    U_base = (
        S_latent_base
        / beta[:, None]
        * (
            np.maximum(
                inside,
                0.0
            )
            + 1e-5
        )
    )

    S_latent_traj[
        qi, :, :
    ] = S_latent_base.T

    U_latent_traj[
        qi, :, :
    ] = U_base.T


print(
    "S_latent_traj shape:",
    S_latent_traj.shape
)

print(
    "U_latent_traj shape:",
    U_latent_traj.shape
)

print(
    "\nAny NaN:",
    np.isnan(S_latent_traj).any()
    or np.isnan(U_latent_traj).any()
)

print(
    "Any inf:",
    np.isinf(S_latent_traj).any()
    or np.isinf(U_latent_traj).any()
)

print(
    "\nS latent range:",
    np.min(S_latent_traj),
    np.median(S_latent_traj),
    np.max(S_latent_traj)
)

print(
    "U latent range:",
    np.min(U_latent_traj),
    np.median(U_latent_traj),
    np.max(U_latent_traj)
)


# ============================================================================
# NOTEBOOK INDEX 67: Cell 64. Reconstruct the exact VeloCycle
# ============================================================================

# ============================================
# Cell 64. Reconstruct the exact VeloCycle
# spliced count_factor for all cells
# ============================================

import numpy as np

# Raw total spliced counts per cell
# Sparse-safe: does NOT densify the full matrix
n_scounts_full = np.asarray(
    adata.layers["spliced"].sum(axis=1)
).ravel().astype(np.float64)

mean_n_scounts = n_scounts_full.mean()

count_factor_full = np.log(
    n_scounts_full
    / mean_n_scounts
)

print(
    "n_scounts_full shape:",
    n_scounts_full.shape
)

print(
    "mean n_scounts:",
    mean_n_scounts
)

print(
    "\ncount_factor summary:"
)

print(
    "min   =", np.min(count_factor_full)
)

print(
    "median=",
    np.median(count_factor_full)
)

print(
    "mean  =",
    np.mean(count_factor_full)
)

print(
    "max   =",
    np.max(count_factor_full)
)

print(
    "\nAny NaN:",
    np.isnan(count_factor_full).any()
)

print(
    "Any inf:",
    np.isinf(count_factor_full).any()
)

# Check alignment with the official compact extract
cell_ids_latent = np.asarray(
    lat["cell_ids"],
    dtype=object
)

print(
    "\nCell order identical to adata:",
    np.array_equal(
        cell_ids_latent,
        np.asarray(
            adata.obs_names,
            dtype=object
        )
    )
)


# ============================================================================
# NOTEBOOK INDEX 68: Cell 65. Estimate condition-specific
# ============================================================================

# ============================================
# Cell 65. Estimate condition-specific
# expression shifts relative to VeloCycle
# latent phase + library-size baseline
# ============================================

import numpy as np
import pandas as pd

# ------------------------------------------------
# Work only on the already frozen final cells
# ------------------------------------------------

idx = np.asarray(
    final_cell_idx,
    dtype=int
)

phi_cells = np.asarray(
    cycle_time_vc[idx],
    dtype=np.float64
) * 2.0 * np.pi

cf_cells = np.asarray(
    count_factor_full[idx],
    dtype=np.float64
)

condition_cells = np.asarray(
    adata.obs["gene"].iloc[idx],
    dtype=object
)

# ------------------------------------------------
# Exact one-harmonic VeloCycle log-S template
#
# log S_ig =
#   nu0_g
# + nu1_sin_g sin(phi_i)
# + nu1_cos_g cos(phi_i)
# + count_factor_i
# ------------------------------------------------

nu0 = coef[0, :]
nu_sin = coef[1, :]
nu_cos = coef[2, :]

sin_phi = np.sin(phi_cells)
cos_phi = np.cos(phi_cells)

# ------------------------------------------------
# Observed raw spliced counts for the 426 genes
#
# Use log(S + 1), matching VeloCycle preprocessing.
# Process condition-by-condition so we never
# densify the full 57k x 426 matrix at once.
# ------------------------------------------------

condition_log_shift = np.full(
    (
        len(final_conditions_vc),
        len(vc_gene_names)
    ),
    np.nan,
    dtype=np.float64
)

condition_cell_counts = np.zeros(
    len(final_conditions_vc),
    dtype=int
)

for qi, q in enumerate(final_conditions_vc):

    local = np.flatnonzero(
        condition_cells == q
    )

    condition_cell_counts[qi] = len(local)

    # This is only condition_cells x 426,
    # not the full transcriptome.
    obs = (
        U_response_raw[local, :] * 0
    )

    # Retrieve spliced response counts
    S_obs = (
        S_response_raw[local, :]
        .toarray()
        .astype(np.float64)
    )

    logS_obs = np.log(
        S_obs + 1.0
    )

    # VeloCycle fitted log-spliced baseline
    logS_base = (
        nu0[None, :]
        + sin_phi[local, None]
          * nu_sin[None, :]
        + cos_phi[local, None]
          * nu_cos[None, :]
        + cf_cells[local, None]
    )

    residual = (
        logS_obs
        - logS_base
    )

    # Robust condition-level shift
    condition_log_shift[
        qi, :
    ] = np.median(
        residual,
        axis=0
    )


print(
    "condition_log_shift shape:",
    condition_log_shift.shape
)

print(
    "Any NaN:",
    np.isnan(
        condition_log_shift
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        condition_log_shift
    ).any()
)

print(
    "\nCells per condition:"
)

print(
    pd.Series(
        condition_cell_counts
    ).describe()
)

print(
    "\nCondition log-shift summary:"
)

print(
    pd.Series(
        condition_log_shift.ravel()
    ).describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
)

# ------------------------------------------------
# Sanity check:
# direct CRISPRi target genes should tend toward
# negative shifts when the target is among the
# 426 response genes.
# ------------------------------------------------

response_lookup = {
    g: j
    for j, g in enumerate(
        vc_gene_names
    )
}

direct_rows = []

for qi, q in enumerate(
    final_conditions_vc
):

    if q == "non-targeting":
        continue

    if q not in response_lookup:
        continue

    gj = response_lookup[q]

    direct_rows.append({
        "condition": q,
        "target_log_shift":
            condition_log_shift[
                qi, gj
            ],
    })

direct_shift_df = pd.DataFrame(
    direct_rows
)

print(
    "\nDirect-target shifts "
    "(targets also among 426 response genes):"
)

print(
    direct_shift_df
    .sort_values(
        "target_log_shift"
    )
    .to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 69: Cell 66. Estimate perturbation-specific
# ============================================================================

# ============================================
# Cell 66. Estimate perturbation-specific
# shifts relative to NT within phase bins
# ============================================

import numpy as np
import pandas as pd

# Raw spliced trajectories already constructed:
# S_response_traj_raw:
#   condition x bin x 426 genes
#
# N_traj_vc:
#   condition x bin cell counts

nt_idx = final_conditions_vc.index(
    "non-targeting"
)

S_nt = np.asarray(
    S_response_traj_raw[nt_idx],
    dtype=np.float64
)  # 10 x 426

# Small pseudocount on raw-count scale
pc = 0.1

# condition x bin x gene
logFC_bin = (
    np.log(
        S_response_traj_raw + pc
    )
    - np.log(
        S_nt[None, :, :] + pc
    )
)

# ------------------------------------------------
# Collapse phase bins robustly to one
# condition-specific gene shift.
#
# Only bins usable in that condition AND NT.
# ------------------------------------------------

condition_logFC_nt = np.full(
    (
        len(final_conditions_vc),
        len(vc_gene_names)
    ),
    np.nan,
    dtype=np.float64
)

for qi, q in enumerate(
    final_conditions_vc
):

    usable = (
        (N_traj_vc[qi] >= 5)
        & (N_traj_vc[nt_idx] >= 5)
    )

    condition_logFC_nt[
        qi, :
    ] = np.nanmedian(
        logFC_bin[
            qi,
            usable,
            :
        ],
        axis=0
    )


print(
    "condition_logFC_nt shape:",
    condition_logFC_nt.shape
)

print(
    "Any NaN:",
    np.isnan(
        condition_logFC_nt
    ).any()
)

print(
    "\nNT shift:"
)

print(
    "max abs =",
    np.nanmax(
        np.abs(
            condition_logFC_nt[
                nt_idx
            ]
        )
    )
)

print(
    "\nAll perturbation-gene shifts:"
)

pert_values = np.delete(
    condition_logFC_nt,
    nt_idx,
    axis=0
).ravel()

print(
    pd.Series(
        pert_values
    ).describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
)


# ------------------------------------------------
# Direct-target sanity check
# ------------------------------------------------

response_lookup = {
    g: j
    for j, g in enumerate(
        vc_gene_names
    )
}

direct_rows = []

for qi, q in enumerate(
    final_conditions_vc
):

    if q == "non-targeting":
        continue

    if q not in response_lookup:
        continue

    gj = response_lookup[q]

    direct_rows.append({
        "condition": q,
        "target_logFC_vs_NT":
            condition_logFC_nt[
                qi, gj
            ],
    })

direct_logFC_df = pd.DataFrame(
    direct_rows
).sort_values(
    "target_logFC_vs_NT"
)

print(
    "\nDirect-target log fold changes vs NT:"
)

print(
    direct_logFC_df.to_string(
        index=False
    )
)

print(
    "\nDirect targets < 0:",
    (
        direct_logFC_df[
            "target_logFC_vs_NT"
        ] < 0
    ).sum(),
    "/",
    len(direct_logFC_df)
)


# ============================================================================
# NOTEBOOK INDEX 70: Cell 67. Build perturbation-specific
# ============================================================================

# ============================================
# Cell 67. Build perturbation-specific
# smoothed spliced response trajectories
# ============================================

import numpy as np

# condition_logFC_nt:
#   condition x 426 genes
#
# S_latent_base.T:
#   10 bins x 426 genes

condition_scale = np.exp(
    condition_logFC_nt
)

S_response_hat = (
    S_latent_base.T[None, :, :]
    * condition_scale[:, None, :]
)

print(
    "S_response_hat shape:",
    S_response_hat.shape
)

print(
    "Any NaN:",
    np.isnan(
        S_response_hat
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        S_response_hat
    ).any()
)

print(
    "\nS_response_hat range:"
)

print(
    "min   =",
    np.min(
        S_response_hat
    )
)

print(
    "median=",
    np.median(
        S_response_hat
    )
)

print(
    "max   =",
    np.max(
        S_response_hat
    )
)

# --------------------------------------------
# Sanity check:
# direct target trajectory should be strongly
# suppressed relative to NT
# --------------------------------------------

response_lookup = {
    g: j
    for j, g in enumerate(
        vc_gene_names
    )
}

condition_lookup = {
    q: i
    for i, q in enumerate(
        final_conditions_vc
    )
}

direct_ratios = []

for q in final_perturbations_vc:

    if q not in response_lookup:
        continue

    qi = condition_lookup[q]
    gi = response_lookup[q]

    ratio = (
        np.mean(
            S_response_hat[
                qi, :, gi
            ]
        )
        /
        np.mean(
            S_response_hat[
                nt_idx, :, gi
            ]
        )
    )

    direct_ratios.append(
        (q, ratio)
    )

print(
    "\nDirect-target mean trajectory ratios "
    "(perturbation / NT):"
)

for q, ratio in direct_ratios:
    print(
        f"{q:10s}  {ratio:.4f}"
    )


# ============================================================================
# NOTEBOOK INDEX 71: Cell 68. Build perturbation-specific
# ============================================================================

# ============================================
# Cell 68. Build perturbation-specific
# smoothed unspliced response trajectories
# ============================================

import numpy as np

U_response_hat = np.empty_like(
    S_response_hat
)

for qi, q in enumerate(
    final_conditions_vc
):

    omega = float(
        speed_lookup[q]
    )

    inside = (
        omega
        * dlogS_dphi
        + gamma[:, None]
    )

    # gene x bin
    kinetic_ratio = (
        np.maximum(
            inside,
            0.0
        )
        + 1e-5
    ) / beta[:, None]

    # S_response_hat is bin x gene,
    # so transpose kinetic_ratio
    U_response_hat[
        qi, :, :
    ] = (
        S_response_hat[
            qi, :, :
        ]
        * kinetic_ratio.T
    )


print(
    "U_response_hat shape:",
    U_response_hat.shape
)

print(
    "Any NaN:",
    np.isnan(
        U_response_hat
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        U_response_hat
    ).any()
)

print(
    "\nU_response_hat range:"
)

print(
    "min   =",
    np.min(
        U_response_hat
    )
)

print(
    "median=",
    np.median(
        U_response_hat
    )
)

print(
    "max   =",
    np.max(
        U_response_hat
    )
)


# --------------------------------------------
# Verify that the perturbation-specific
# trajectories still satisfy the VeloCycle
# kinetic identity analytically
# --------------------------------------------

rel_errors = []

for qi, q in enumerate(
    final_conditions_vc
):

    omega = float(
        speed_lookup[q]
    )

    # ds/dphi = s * d(log s)/dphi
    lhs = (
        omega
        * S_response_hat[qi].T
        * dlogS_dphi
    )

    rhs = (
        beta[:, None]
        * U_response_hat[qi].T
        - gamma[:, None]
        * S_response_hat[qi].T
    )

    inside = (
        omega
        * dlogS_dphi
        + gamma[:, None]
    )

    active = inside > 0

    denom = np.maximum(
        np.abs(lhs),
        1e-12
    )

    rel_errors.append(
        np.median(
            np.abs(
                rhs[active]
                - lhs[active]
            )
            / denom[active]
        )
    )


rel_errors = np.asarray(
    rel_errors
)

print(
    "\nLatent kinetic identity "
    "median relative error:"
)

print(
    "median =",
    np.median(rel_errors)
)

print(
    "max    =",
    np.max(rel_errors)
)


# ============================================================================
# NOTEBOOK INDEX 72: Cell 69. Build raw spliced trajectories
# ============================================================================

# ============================================
# Cell 69. Build raw spliced trajectories
# for the 151 regulator genes
# ============================================

import numpy as np

n_cond = len(final_conditions_vc)
n_bins = len(bin_centers)
n_reg = len(regulator_genes_vc)

S_regulator_traj_raw = np.full(
    (n_cond, n_bins, n_reg),
    np.nan,
    dtype=np.float64
)

N_regulator_traj = np.zeros(
    (n_cond, n_bins),
    dtype=int
)

# final-cell arrays already aligned with
# S_regulator_raw rows
condition_cells = np.asarray(
    adata.obs["gene"].iloc[
        final_cell_idx
    ],
    dtype=object
)

cycle_cells = np.asarray(
    cycle_time_vc[
        final_cell_idx
    ],
    dtype=np.float64
)

bin_cells = np.digitize(
    cycle_cells,
    bin_edges,
    right=False
) - 1

# Protect the rare value exactly equal to 1
bin_cells = np.clip(
    bin_cells,
    0,
    n_bins - 1
)

condition_lookup = {
    q: i
    for i, q in enumerate(
        final_conditions_vc
    )
}

for q in final_conditions_vc:

    qi = condition_lookup[q]

    cond_mask = (
        condition_cells == q
    )

    for b in range(n_bins):

        local = np.flatnonzero(
            cond_mask
            & (bin_cells == b)
        )

        n = len(local)

        N_regulator_traj[
            qi, b
        ] = n

        if n < 5:
            continue

        S_regulator_traj_raw[
            qi, b, :
        ] = np.asarray(
            S_regulator_raw[
                local, :
            ].mean(axis=0)
        ).ravel()


print(
    "S_regulator_traj_raw shape:",
    S_regulator_traj_raw.shape
)

print(
    "N_regulator_traj shape:",
    N_regulator_traj.shape
)

usable = (
    N_regulator_traj >= 5
)

print(
    "\nUsable condition-bin pairs:",
    int(usable.sum()),
    "/",
    usable.size
)

print(
    "Usable fraction:",
    usable.mean()
)

print(
    "\nNaN inside usable bins:",
    np.isnan(
        S_regulator_traj_raw[
            usable
        ]
    ).any()
)

# Direct-target zero check
reg_lookup = {
    g: j
    for j, g in enumerate(
        regulator_genes_vc
    )
}

violations = []

for q in final_perturbations_vc:

    qi = condition_lookup[q]
    gj = reg_lookup[q]

    vals = S_regulator_traj_raw[
        qi, :, gj
    ]

    vals = vals[
        np.isfinite(vals)
    ]

    if np.any(vals != 0):
        violations.append(q)

print(
    "\nDirect-target zero violations:",
    len(violations)
)

if violations:
    print(violations[:20])


# ============================================================================
# NOTEBOOK INDEX 73: Cell 70. Circular Fourier smoothing of
# ============================================================================

# ============================================
# Cell 70. Circular Fourier smoothing of
# the 151 regulator trajectories
# ============================================

import numpy as np

phi_bins = (
    2.0
    * np.pi
    * np.asarray(bin_centers)
)

# 10 x 3
F = np.column_stack([
    np.ones(n_bins),
    np.sin(phi_bins),
    np.cos(phi_bins),
])

S_regulator_hat = np.full(
    (
        len(final_conditions_vc),
        n_bins,
        len(regulator_genes_vc)
    ),
    np.nan,
    dtype=np.float64
)

for qi, q in enumerate(
    final_conditions_vc
):

    usable = (
        N_regulator_traj[
            qi
        ] >= 5
    )

    F_q = F[usable, :]

    # sqrt(n) weighting
    w = np.sqrt(
        N_regulator_traj[
            qi, usable
        ].astype(np.float64)
    )

    Fw = (
        F_q
        * w[:, None]
    )

    for gj, g in enumerate(
        regulator_genes_vc
    ):

        # The QC-filtered direct target
        # is exactly zero by construction.
        if (
            q != "non-targeting"
            and q == g
        ):
            S_regulator_hat[
                qi, :, gj
            ] = 0.0
            continue

        y = S_regulator_traj_raw[
            qi, usable, gj
        ]

        yw = y * w

        coef_qg, *_ = np.linalg.lstsq(
            Fw,
            yw,
            rcond=None
        )

        y_hat = F @ coef_qg

        # Spliced abundance cannot be negative.
        S_regulator_hat[
            qi, :, gj
        ] = np.maximum(
            y_hat,
            0.0
        )


print(
    "S_regulator_hat shape:",
    S_regulator_hat.shape
)

print(
    "Any NaN:",
    np.isnan(
        S_regulator_hat
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        S_regulator_hat
    ).any()
)

print(
    "\nSmoothed regulator range:"
)

print(
    "min   =",
    np.min(
        S_regulator_hat
    )
)

print(
    "median=",
    np.median(
        S_regulator_hat
    )
)

print(
    "max   =",
    np.max(
        S_regulator_hat
    )
)


# --------------------------------------------
# Check direct targets remain exactly zero
# --------------------------------------------

violations = []

for q in final_perturbations_vc:

    qi = condition_lookup[q]
    gj = reg_lookup[q]

    if not np.all(
        S_regulator_hat[
            qi, :, gj
        ] == 0.0
    ):
        violations.append(q)

print(
    "\nDirect-target zero violations:",
    len(violations)
)


# --------------------------------------------
# How closely does the smooth curve reproduce
# observed usable-bin means?
# --------------------------------------------

obs_all = []
fit_all = []

for qi in range(
    len(final_conditions_vc)
):

    usable = (
        N_regulator_traj[
            qi
        ] >= 5
    )

    obs_all.append(
        S_regulator_traj_raw[
            qi, usable, :
        ].ravel()
    )

    fit_all.append(
        S_regulator_hat[
            qi, usable, :
        ].ravel()
    )

obs_all = np.concatenate(
    obs_all
)

fit_all = np.concatenate(
    fit_all
)

global_corr = np.corrcoef(
    obs_all,
    fit_all
)[0, 1]

rmse = np.sqrt(
    np.mean(
        (
            obs_all
            - fit_all
        ) ** 2
    )
)

print(
    "\nObserved-vs-smoothed:"
)

print(
    "global correlation =",
    global_corr
)

print(
    "RMSE =",
    rmse
)


# ============================================================================
# NOTEBOOK INDEX 74: Cell 71. Build FINAL integral design matrix X
# ============================================================================

# ============================================
# Cell 71. Build FINAL integral design matrix X
# from smoothed regulator trajectories
# ============================================

import numpy as np
import pandas as pd

X_rows = []
final_interval_rows = []

for qi, q in enumerate(
    final_conditions_vc
):

    omega_q = float(
        speed_lookup[q]
    )

    rho_q = (
        omega_q
        / omega_nt
    )

    dt_q = (
        0.1
        / rho_q
    )

    for b in range(
        n_bins - 1
    ):

        # Require both adjacent bins
        # to contain >= 5 cells
        if (
            N_regulator_traj[
                qi, b
            ] < 5
            or
            N_regulator_traj[
                qi, b + 1
            ] < 5
        ):
            continue

        s_a = (
            S_regulator_hat[
                qi, b, :
            ].copy()
        )

        s_b = (
            S_regulator_hat[
                qi, b + 1, :
            ].copy()
        )

        # Explicit M_q:
        # zero the directly perturbed
        # regulator coordinate
        if q != "non-targeting":

            gj = reg_lookup[q]

            s_a[gj] = 0.0
            s_b[gj] = 0.0

        # Trapezoidal integral
        integral_s = (
            0.5
            * dt_q
            * (
                s_a
                + s_b
            )
        )

        # First column = intercept integral
        # integral 1 dt = Delta t
        x = np.concatenate([
            [dt_q],
            integral_s
        ])

        X_rows.append(x)

        final_interval_rows.append({
            "condition": q,
            "condition_idx": qi,
            "bin_a": b,
            "bin_b": b + 1,
            "rho": rho_q,
            "delta_t": dt_q,
            "n_a": int(
                N_regulator_traj[
                    qi, b
                ]
            ),
            "n_b": int(
                N_regulator_traj[
                    qi, b + 1
                ]
            ),
        })


X_final = np.vstack(
    X_rows
)

print(
    "X_final shape:",
    X_final.shape
)

print(
    "Number of intervals:",
    len(
        final_interval_rows
    )
)

print(
    "\nAny NaN:",
    np.isnan(
        X_final
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        X_final
    ).any()
)

print(
    "\nIntercept equals delta_t:",
    np.allclose(
        X_final[:, 0],
        np.asarray([
            r["delta_t"]
            for r in final_interval_rows
        ])
    )
)


# --------------------------------------------
# Verify M_q target coordinate = 0
# --------------------------------------------

mq_violations = 0

for i, row in enumerate(
    final_interval_rows
):

    q = row["condition"]

    if q == "non-targeting":
        continue

    gj = reg_lookup[q]

    if X_final[
        i, 1 + gj
    ] != 0.0:
        mq_violations += 1

print(
    "M_q design violations:",
    mq_violations
)


# --------------------------------------------
# Interval counts by condition
# --------------------------------------------

interval_counts = (
    pd.Series([
        r["condition"]
        for r in final_interval_rows
    ])
    .value_counts()
)

print(
    "\nIntervals per condition:"
)

print(
    interval_counts.describe()
)


# --------------------------------------------
# Numerical scale of regulator integrals
# --------------------------------------------

print(
    "\nRegulator integral summary:"
)

print(
    pd.Series(
        X_final[:, 1:].ravel()
    ).describe(
        percentiles=[
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
        ]
    )
)


# ============================================================================
# NOTEBOOK INDEX 75: Cell 72. Build FINAL integral response Y
# ============================================================================

# ============================================
# Cell 72. Build FINAL integral response Y
# from perturbation-specific latent trajectories
# ============================================

import numpy as np
import pandas as pd

# Convert VeloCycle beta to our normalized
# time coordinate:
#
# t = omega_NT * tau / (2*pi)
#
# therefore:
# beta_tilde = (2*pi / omega_NT) * beta

time_scale = (
    2.0
    * np.pi
    / omega_nt
)

beta_tilde = (
    time_scale
    * beta
)

Y_rows = []
delta_U_rows = []
beta_integral_rows = []

for row in final_interval_rows:

    qi = row[
        "condition_idx"
    ]

    a = row[
        "bin_a"
    ]

    b = row[
        "bin_b"
    ]

    dt = row[
        "delta_t"
    ]

    u_a = U_response_hat[
        qi, a, :
    ]

    u_b = U_response_hat[
        qi, b, :
    ]

    delta_u = (
        u_b
        - u_a
    )

    integral_u = (
        0.5
        * dt
        * (
            u_a
            + u_b
        )
    )

    beta_integral = (
        beta_tilde
        * integral_u
    )

    # h_g(q) = 0 here.
    # Direct-target rows will be excluded
    # gene-by-gene during reconstruction.
    y = (
        delta_u
        + beta_integral
    )

    Y_rows.append(y)
    delta_U_rows.append(
        delta_u
    )
    beta_integral_rows.append(
        beta_integral
    )


Y_final = np.vstack(
    Y_rows
)

delta_U_final = np.vstack(
    delta_U_rows
)

beta_integral_final = np.vstack(
    beta_integral_rows
)


print(
    "Y_final shape:",
    Y_final.shape
)

print(
    "Expected shape:",
    (
        X_final.shape[0],
        len(vc_gene_names)
    )
)

print(
    "\nAny NaN:",
    np.isnan(
        Y_final
    ).any()
)

print(
    "Any inf:",
    np.isinf(
        Y_final
    ).any()
)


print(
    "\nY_final summary:"
)

print(
    pd.Series(
        Y_final.ravel()
    ).describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
)


print(
    "\ndelta_U summary:"
)

print(
    pd.Series(
        delta_U_final.ravel()
    ).describe(
        percentiles=[
            0.01,
            0.50,
            0.99,
        ]
    )
)


print(
    "\nbeta-integral summary:"
)

print(
    pd.Series(
        beta_integral_final.ravel()
    ).describe(
        percentiles=[
            0.01,
            0.50,
            0.99,
        ]
    )
)


# --------------------------------------------
# Exact row-level formula check
# --------------------------------------------

check = (
    delta_U_final
    + beta_integral_final
)

print(
    "\nFormula exact:",
    np.allclose(
        Y_final,
        check
    )
)


# ============================================================================
# NOTEBOOK INDEX 76: Cell 73. Numerical rank and conditioning
# ============================================================================

# ============================================
# Cell 73. Numerical rank and conditioning
# of the FINAL integral design matrix
# ============================================

import numpy as np

print(
    "X_final shape:",
    X_final.shape
)

# --------------------------------------------
# Standardize regulator columns only.
#
# Intercept/integral-of-1 column is kept
# separate because its scale has a different
# physical meaning.
# --------------------------------------------

X_reg = np.asarray(
    X_final[:, 1:],
    dtype=np.float64
)

reg_mean = X_reg.mean(
    axis=0
)

reg_std = X_reg.std(
    axis=0,
    ddof=0
)

print(
    "\nRegulator columns with zero std:",
    np.sum(
        reg_std == 0
    )
)

X_reg_z = (
    X_reg
    - reg_mean[None, :]
) / reg_std[None, :]

# --------------------------------------------
# Singular values
# --------------------------------------------

singular_values = np.linalg.svd(
    X_reg_z,
    compute_uv=False
)

rank = np.linalg.matrix_rank(
    X_reg_z
)

condition_number = (
    singular_values[0]
    / singular_values[-1]
)

# Information matrix normalized by n
info_eigenvalues = (
    singular_values ** 2
    / X_reg_z.shape[0]
)

lambda_max = np.max(
    info_eigenvalues
)

lambda_min = np.min(
    info_eigenvalues
)

print(
    "\nNumerical rank:",
    rank,
    "/",
    X_reg_z.shape[1]
)

print(
    "\nSingular values:"
)

print(
    "largest =",
    singular_values[0]
)

print(
    "median  =",
    np.median(
        singular_values
    )
)

print(
    "smallest=",
    singular_values[-1]
)

print(
    "\nCondition number:",
    condition_number
)

print(
    "\nStandardized information matrix:"
)

print(
    "lambda_max =",
    lambda_max
)

print(
    "lambda_min =",
    lambda_min
)

print(
    "condition number =",
    lambda_max
    / lambda_min
)


# --------------------------------------------
# Bottom 10 singular values
# --------------------------------------------

print(
    "\nSmallest 10 singular values:"
)

print(
    singular_values[-10:]
)


# ============================================================================
# NOTEBOOK INDEX 77: Cell 74. Perturbation-assisted identifiability:
# ============================================================================

# ============================================
# Cell 74. Perturbation-assisted identifiability:
# rank growth as conditions are added
# ============================================

import numpy as np
import pandas as pd

# ------------------------------------------------
# Use the SAME global centering/scaling from Cell 73
# so all subsets live in the same coordinate system.
# ------------------------------------------------

Xz_all = (
    X_final[:, 1:]
    - reg_mean[None, :]
) / reg_std[None, :]

row_conditions = np.asarray([
    r["condition"]
    for r in final_interval_rows
], dtype=object)

# Fixed condition order:
# NT first, then the frozen perturbation order
condition_order = list(
    final_conditions_vc
)

results = []

for k in range(
    1,
    len(condition_order) + 1
):

    included = set(
        condition_order[:k]
    )

    mask = np.array([
        q in included
        for q in row_conditions
    ])

    Xk = Xz_all[
        mask, :
    ]

    s = np.linalg.svd(
        Xk,
        compute_uv=False
    )

    rank_k = np.linalg.matrix_rank(
        Xk
    )

    # lambda_min of X'X/n is exactly zero
    # whenever rank < number of regulators.
    if rank_k < Xk.shape[1]:
        lambda_min_k = 0.0
        cond_info_k = np.inf
    else:
        eig = (
            s ** 2
            / Xk.shape[0]
        )

        lambda_min_k = float(
            np.min(eig)
        )

        cond_info_k = float(
            np.max(eig)
            / np.min(eig)
        )

    results.append({
        "n_conditions": k,
        "n_perturbations": k - 1,
        "n_rows": Xk.shape[0],
        "rank": rank_k,
        "lambda_min": lambda_min_k,
        "condition_number_info":
            cond_info_k,
    })


identifiability_path = pd.DataFrame(
    results
)

print(
    "NT only:"
)

print(
    identifiability_path.iloc[0]
)

print(
    "\nAll conditions:"
)

print(
    identifiability_path.iloc[-1]
)


# ------------------------------------------------
# First point at which full rank is reached
# ------------------------------------------------

full_rank_rows = (
    identifiability_path[
        identifiability_path[
            "rank"
        ] == len(
            regulator_genes_vc
        )
    ]
)

if len(full_rank_rows) > 0:

    first_full = (
        full_rank_rows.iloc[0]
    )

    print(
        "\nFirst full-rank point:"
    )

    print(
        first_full
    )

else:

    print(
        "\nFull rank was never reached."
    )


# ------------------------------------------------
# Selected checkpoints
# ------------------------------------------------

checkpoints = [
    1,
    5,
    10,
    20,
    40,
    80,
    120,
    len(condition_order),
]

checkpoints = [
    k
    for k in checkpoints
    if k <= len(
        condition_order
    )
]

print(
    "\nSelected checkpoints:"
)

print(
    identifiability_path[
        identifiability_path[
            "n_conditions"
        ].isin(
            checkpoints
        )
    ].to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 78: Cell 75. Random-order robustness of
# ============================================================================

# ============================================
# Cell 75. Random-order robustness of
# perturbation-assisted rank recovery
# ============================================

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)

n_repeats = 100

# Evaluate at selected perturbation counts.
# Include dense coverage around the expected
# full-rank transition.
k_grid = np.array([
    0,
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
    120,
    151,
], dtype=int)

perturbations = np.asarray(
    final_perturbations_vc,
    dtype=object
)

records = []

for rep in range(n_repeats):

    order = rng.permutation(
        perturbations
    )

    for k in k_grid:

        included = set(
            ["non-targeting"]
            + list(order[:k])
        )

        mask = np.fromiter(
            (
                q in included
                for q in row_conditions
            ),
            dtype=bool,
            count=len(row_conditions)
        )

        Xk = Xz_all[
            mask, :
        ]

        rank_k = np.linalg.matrix_rank(
            Xk
        )

        if rank_k < len(
            regulator_genes_vc
        ):
            lambda_min_k = 0.0

        else:
            s = np.linalg.svd(
                Xk,
                compute_uv=False
            )

            lambda_min_k = (
                s[-1] ** 2
                / Xk.shape[0]
            )

        records.append({
            "repeat": rep,
            "n_perturbations": k,
            "n_rows": Xk.shape[0],
            "rank": rank_k,
            "lambda_min": lambda_min_k,
        })


rank_robustness = pd.DataFrame(
    records
)

summary = (
    rank_robustness
    .groupby(
        "n_perturbations",
        observed=True
    )
    .agg(
        rank_median=(
            "rank",
            "median"
        ),
        rank_q10=(
            "rank",
            lambda x:
                np.quantile(x, 0.10)
        ),
        rank_q90=(
            "rank",
            lambda x:
                np.quantile(x, 0.90)
        ),
        full_rank_fraction=(
            "rank",
            lambda x:
                np.mean(
                    x
                    == len(
                        regulator_genes_vc
                    )
                )
        ),
        lambda_min_median=(
            "lambda_min",
            "median"
        ),
        lambda_min_q10=(
            "lambda_min",
            lambda x:
                np.quantile(x, 0.10)
        ),
        lambda_min_q90=(
            "lambda_min",
            lambda x:
                np.quantile(x, 0.90)
        ),
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False
    )
)


# --------------------------------------------
# For each random ordering, find the first
# perturbation count where full rank appears.
# --------------------------------------------

first_full_rank = []

for rep in range(n_repeats):

    order = rng.permutation(
        perturbations
    )

    first_k = None

    for k in range(
        0,
        len(perturbations) + 1
    ):

        included = set(
            ["non-targeting"]
            + list(order[:k])
        )

        mask = np.fromiter(
            (
                q in included
                for q in row_conditions
            ),
            dtype=bool,
            count=len(row_conditions)
        )

        rank_k = np.linalg.matrix_rank(
            Xz_all[
                mask, :
            ]
        )

        if rank_k == len(
            regulator_genes_vc
        ):
            first_k = k
            break

    first_full_rank.append(
        first_k
    )


first_full_rank = np.asarray(
    first_full_rank,
    dtype=float
)

print(
    "\nFirst full-rank perturbation count:"
)

print(
    pd.Series(
        first_full_rank
    ).describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
)


# ============================================================================
# NOTEBOOK INDEX 79: Cell 76. Build gene-specific valid-row masks
# ============================================================================

# ============================================
# Cell 76. Build gene-specific valid-row masks
# for sparse reconstruction
# ============================================

import numpy as np
import pandas as pd

response_genes = np.asarray(
    vc_gene_names,
    dtype=object
)

regulator_set = set(
    regulator_genes_vc
)

row_conditions = np.asarray([
    row["condition"]
    for row in final_interval_rows
], dtype=object)

valid_row_masks = {}

excluded_summary = []

for g in response_genes:

    # Default:
    # all intervals are usable
    mask = np.ones(
        len(final_interval_rows),
        dtype=bool
    )

    # If response gene g is itself one of
    # the directly perturbed targets,
    # remove condition q = g because h_g(q)
    # is unknown.
    if g in regulator_set:

        direct_mask = (
            row_conditions == g
        )

        mask[
            direct_mask
        ] = False

        n_excluded = int(
            direct_mask.sum()
        )

    else:

        n_excluded = 0

    valid_row_masks[g] = mask

    excluded_summary.append({
        "gene": g,
        "is_perturbed_target":
            g in regulator_set,
        "n_total_rows":
            len(mask),
        "n_excluded_direct":
            n_excluded,
        "n_valid_rows":
            int(mask.sum()),
    })


valid_row_summary = pd.DataFrame(
    excluded_summary
)

print(
    "Response genes:",
    len(response_genes)
)

print(
    "Response genes also directly perturbed:",
    valid_row_summary[
        "is_perturbed_target"
    ].sum()
)

print(
    "\nValid-row count summary:"
)

print(
    valid_row_summary[
        "n_valid_rows"
    ].describe()
)

print(
    "\nDirect-intervention exclusions:"
)

print(
    valid_row_summary[
        valid_row_summary[
            "n_excluded_direct"
        ] > 0
    ][[
        "gene",
        "n_excluded_direct",
        "n_valid_rows",
    ]]
    .sort_values(
        "gene"
    )
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Sanity checks
# --------------------------------------------

assert len(
    valid_row_masks
) == 426

assert all(
    len(mask)
    == X_final.shape[0]
    for mask
    in valid_row_masks.values()
)

print(
    "\nMask bookkeeping checks passed."
)


# ============================================================================
# NOTEBOOK INDEX 80: Cell 77. Prepare condition-grouped CV
# ============================================================================

# ============================================
# Cell 77. Prepare condition-grouped CV
# for one test response gene: TACC3
# ============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

test_gene = "TACC3"

g_idx = np.where(
    response_genes == test_gene
)[0][0]

valid_mask = valid_row_masks[
    test_gene
]

X_test = np.asarray(
    X_final[
        valid_mask, :
    ],
    dtype=np.float64
)

y_test = np.asarray(
    Y_final[
        valid_mask,
        g_idx
    ],
    dtype=np.float64
)

groups_test = row_conditions[
    valid_mask
]

print(
    "Test gene:",
    test_gene
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)

print(
    "Number of condition groups:",
    len(
        np.unique(
            groups_test
        )
    )
)

print(
    "TACC3 condition still present:",
    np.any(
        groups_test == "TACC3"
    )
)


# --------------------------------------------
# 5-fold GroupKFold:
# entire perturbation conditions stay together
# --------------------------------------------

n_splits = 5

gkf = GroupKFold(
    n_splits=n_splits
)

fold_rows = []

fold_indices = []

for fold, (
    train_idx,
    val_idx
) in enumerate(
    gkf.split(
        X_test,
        y_test,
        groups=groups_test
    ),
    start=1
):

    train_conditions = np.unique(
        groups_test[
            train_idx
        ]
    )

    val_conditions = np.unique(
        groups_test[
            val_idx
        ]
    )

    overlap = np.intersect1d(
        train_conditions,
        val_conditions
    )

    fold_indices.append(
        (
            train_idx,
            val_idx
        )
    )

    fold_rows.append({
        "fold": fold,
        "train_rows":
            len(train_idx),
        "val_rows":
            len(val_idx),
        "train_conditions":
            len(train_conditions),
        "val_conditions":
            len(val_conditions),
        "condition_overlap":
            len(overlap),
    })


fold_summary = pd.DataFrame(
    fold_rows
)

print(
    "\nGrouped CV folds:"
)

print(
    fold_summary.to_string(
        index=False
    )
)

print(
    "\nAll folds have zero "
    "condition leakage:",
    (
        fold_summary[
            "condition_overlap"
        ] == 0
    ).all()
)


# --------------------------------------------
# Check every valid row appears exactly once
# as validation data
# --------------------------------------------

validation_counts = np.zeros(
    len(y_test),
    dtype=int
)

for _, val_idx in fold_indices:
    validation_counts[
        val_idx
    ] += 1

print(
    "Every row validated exactly once:",
    np.all(
        validation_counts == 1
    )
)


# ============================================================================
# NOTEBOOK INDEX 81: Cell 78. Grouped-CV Lasso for TACC3
# ============================================================================

# ============================================
# Cell 78. Grouped-CV Lasso for TACC3
# Select lambda without condition leakage
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ------------------------------------------------
# Convert integral equation to rate form:
#
# Y / dt = c_g + A_g * (integral S / dt)
#
# This makes sklearn's unpenalized intercept
# exactly the basal term c_g.
# ------------------------------------------------

dt_test = X_test[:, 0]

Z_test = (
    X_test[:, 1:]
    / dt_test[:, None]
)

r_test = (
    y_test
    / dt_test
)

print(
    "Z_test shape:",
    Z_test.shape
)

print(
    "r_test shape:",
    r_test.shape
)

print(
    "Any NaN/inf:",
    (
        ~np.isfinite(
            Z_test
        )
    ).any()
    or
    (
        ~np.isfinite(
            r_test
        )
    ).any()
)


# ------------------------------------------------
# Lambda grid
#
# sklearn objective:
#
# (1 / (2n)) ||y - Xw||^2
# + alpha ||w||_1
# ------------------------------------------------

alpha_grid = np.logspace(
    -4,
    1,
    40
)

cv_records = []

for alpha in alpha_grid:

    fold_mse = []
    fold_nonzero = []

    for train_idx, val_idx in fold_indices:

        Z_train = Z_test[
            train_idx
        ]

        Z_val = Z_test[
            val_idx
        ]

        r_train = r_test[
            train_idx
        ]

        r_val = r_test[
            val_idx
        ]

        # Fit scaler ONLY on training fold
        scaler = StandardScaler()

        Z_train_z = (
            scaler.fit_transform(
                Z_train
            )
        )

        Z_val_z = (
            scaler.transform(
                Z_val
            )
        )

        model = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z_train_z,
            r_train
        )

        pred = model.predict(
            Z_val_z
        )

        fold_mse.append(
            mean_squared_error(
                r_val,
                pred
            )
        )

        fold_nonzero.append(
            np.count_nonzero(
                model.coef_
            )
        )

    cv_records.append({
        "alpha": alpha,
        "mean_mse":
            np.mean(
                fold_mse
            ),
        "std_mse":
            np.std(
                fold_mse,
                ddof=1
            ),
        "mean_nonzero":
            np.mean(
                fold_nonzero
            ),
    })


cv_tacc3 = pd.DataFrame(
    cv_records
)

best_idx = (
    cv_tacc3[
        "mean_mse"
    ].idxmin()
)

best_row = cv_tacc3.loc[
    best_idx
]

print(
    "\nBest CV alpha:"
)

print(
    best_row
)


print(
    "\nTop 10 alpha values by CV MSE:"
)

print(
    cv_tacc3
    .sort_values(
        "mean_mse"
    )
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nGrid boundary check:"
)

print(
    "best alpha =",
    best_row["alpha"]
)

print(
    "grid min   =",
    alpha_grid.min()
)

print(
    "grid max   =",
    alpha_grid.max()
)

print(
    "best at boundary:",
    (
        best_idx
        == cv_tacc3.index[0]
        or
        best_idx
        == cv_tacc3.index[-1]
    )
)


# ============================================================================
# NOTEBOOK INDEX 82: Cell 79. Final refit for TACC3
# ============================================================================

# ============================================
# Cell 79. Final refit for TACC3
# and recover coefficients on original scale
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

best_alpha = float(
    best_row["alpha"]
)

# --------------------------------------------
# Fit scaler on ALL valid TACC3 rows
# now that alpha has been selected by CV
# --------------------------------------------

scaler_final = StandardScaler()

Z_test_z = scaler_final.fit_transform(
    Z_test
)

model_final = Lasso(
    alpha=best_alpha,
    fit_intercept=True,
    max_iter=20000,
    tol=1e-6,
    selection="cyclic",
)

model_final.fit(
    Z_test_z,
    r_test
)

# --------------------------------------------
# Convert coefficients back to original
# regulator scale
#
# r = c + sum_j A_j Z_j
# --------------------------------------------

A_tacc3 = (
    model_final.coef_
    / scaler_final.scale_
)

c_tacc3 = (
    model_final.intercept_
    - np.sum(
        model_final.coef_
        * scaler_final.mean_
        / scaler_final.scale_
    )
)

pred_rate = (
    c_tacc3
    + Z_test @ A_tacc3
)

train_mse = mean_squared_error(
    r_test,
    pred_rate
)

ss_res = np.sum(
    (
        r_test
        - pred_rate
    ) ** 2
)

ss_tot = np.sum(
    (
        r_test
        - np.mean(r_test)
    ) ** 2
)

train_r2 = (
    1.0
    - ss_res / ss_tot
)

nonzero = np.flatnonzero(
    A_tacc3 != 0
)

print(
    "Gene:",
    test_gene
)

print(
    "Selected alpha:",
    best_alpha
)

print(
    "Basal term c_g:",
    c_tacc3
)

print(
    "\nNonzero regulators:",
    len(nonzero),
    "/",
    len(regulator_genes_vc)
)

print(
    "Training MSE:",
    train_mse
)

print(
    "Training R^2:",
    train_r2
)


# --------------------------------------------
# Ranked inferred regulators
# --------------------------------------------

coef_df = pd.DataFrame({
    "regulator":
        np.asarray(
            regulator_genes_vc
        )[nonzero],
    "A":
        A_tacc3[
            nonzero
        ],
})

coef_df[
    "abs_A"
] = np.abs(
    coef_df["A"]
)

coef_df = (
    coef_df
    .sort_values(
        "abs_A",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

print(
    "\nTop inferred TACC3 regulators:"
)

print(
    coef_df
    .head(20)
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Sign counts
# --------------------------------------------

print(
    "\nPositive edges:",
    np.sum(
        A_tacc3 > 0
    )
)

print(
    "Negative edges:",
    np.sum(
        A_tacc3 < 0
    )
)

print(
    "Zero edges:",
    np.sum(
        A_tacc3 == 0
    )
)


# --------------------------------------------
# Exact original-scale prediction check
# --------------------------------------------

pred_sklearn = model_final.predict(
    Z_test_z
)

print(
    "\nBack-transform prediction exact:",
    np.allclose(
        pred_rate,
        pred_sklearn
    )
)


# ============================================================================
# NOTEBOOK INDEX 83: Cell 80. Inspect response-rate scale
# ============================================================================

# ============================================
# Cell 80. Inspect response-rate scale
# across all 426 genes
# ============================================

import numpy as np
import pandas as pd

dt_all = np.asarray(
    X_final[:, 0],
    dtype=np.float64
)

R_all = (
    Y_final
    / dt_all[:, None]
)

response_scale_rows = []

for gi, g in enumerate(
    response_genes
):

    mask = valid_row_masks[g]

    r = R_all[
        mask, gi
    ]

    response_scale_rows.append({
        "gene": g,
        "n_rows": len(r),
        "mean": np.mean(r),
        "std": np.std(
            r,
            ddof=0
        ),
        "median": np.median(r),
        "rms": np.sqrt(
            np.mean(
                r ** 2
            )
        ),
        "q05": np.quantile(
            r,
            0.05
        ),
        "q95": np.quantile(
            r,
            0.95
        ),
    })


response_scale_df = pd.DataFrame(
    response_scale_rows
)

print(
    "Response-rate scale summary:"
)

print(
    response_scale_df[
        ["std", "rms"]
    ].describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
)


print(
    "\nLargest response-rate SD:"
)

print(
    response_scale_df
    .sort_values(
        "std",
        ascending=False
    )
    .head(15)
    .to_string(
        index=False
    )
)


print(
    "\nSmallest response-rate SD:"
)

print(
    response_scale_df
    .sort_values(
        "std"
    )
    .head(15)
    .to_string(
        index=False
    )
)


tacc3_scale = response_scale_df[
    response_scale_df[
        "gene"
    ] == "TACC3"
]

print(
    "\nTACC3 scale:"
)

print(
    tacc3_scale.to_string(
        index=False
    )
)


print(
    "\nMax / min response SD ratio:",
    response_scale_df[
        "std"
    ].max()
    /
    response_scale_df[
        "std"
    ].min()
)


# ============================================================================
# NOTEBOOK INDEX 84: Cell 81. TACC3 grouped CV with BOTH
# ============================================================================

# ============================================
# Cell 81. TACC3 grouped CV with BOTH
# predictor and response scaling
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

alpha_grid_scaled = np.logspace(
    -4,
    0,
    40
)

cv_scaled_records = []

for alpha in alpha_grid_scaled:

    fold_mse_std = []
    fold_nonzero = []

    for train_idx, val_idx in fold_indices:

        Z_train = Z_test[
            train_idx
        ]

        Z_val = Z_test[
            val_idx
        ]

        r_train = r_test[
            train_idx
        ]

        r_val = r_test[
            val_idx
        ]

        # ------------------------------------
        # Predictor scaling:
        # training fold ONLY
        # ------------------------------------

        x_scaler = StandardScaler()

        Z_train_z = (
            x_scaler.fit_transform(
                Z_train
            )
        )

        Z_val_z = (
            x_scaler.transform(
                Z_val
            )
        )

        # ------------------------------------
        # Response scaling:
        # training fold ONLY
        # ------------------------------------

        r_mean = np.mean(
            r_train
        )

        r_std = np.std(
            r_train,
            ddof=0
        )

        r_train_z = (
            r_train
            - r_mean
        ) / r_std

        r_val_z = (
            r_val
            - r_mean
        ) / r_std

        model = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z_train_z,
            r_train_z
        )

        pred_z = model.predict(
            Z_val_z
        )

        fold_mse_std.append(
            mean_squared_error(
                r_val_z,
                pred_z
            )
        )

        fold_nonzero.append(
            np.count_nonzero(
                model.coef_
            )
        )

    cv_scaled_records.append({
        "alpha": alpha,
        "mean_mse_std":
            np.mean(
                fold_mse_std
            ),
        "std_mse_std":
            np.std(
                fold_mse_std,
                ddof=1
            ),
        "mean_nonzero":
            np.mean(
                fold_nonzero
            ),
    })


cv_tacc3_scaled = pd.DataFrame(
    cv_scaled_records
)

best_scaled_idx = (
    cv_tacc3_scaled[
        "mean_mse_std"
    ].idxmin()
)

best_scaled_row = (
    cv_tacc3_scaled.loc[
        best_scaled_idx
    ]
)


print(
    "Best scaled alpha:"
)

print(
    best_scaled_row
)


print(
    "\nTop 10 scaled alpha values:"
)

print(
    cv_tacc3_scaled
    .sort_values(
        "mean_mse_std"
    )
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nGrid boundary check:"
)

print(
    "best alpha =",
    best_scaled_row[
        "alpha"
    ]
)

print(
    "grid min   =",
    alpha_grid_scaled.min()
)

print(
    "grid max   =",
    alpha_grid_scaled.max()
)

print(
    "best at boundary:",
    (
        best_scaled_idx
        == cv_tacc3_scaled.index[0]
        or
        best_scaled_idx
        == cv_tacc3_scaled.index[-1]
    )
)


# --------------------------------------------
# Compare with the old unscaled-response fit
#
# Rough equivalence:
# alpha_raw / SD(response)
# --------------------------------------------

print(
    "\nTACC3 full-data response SD:",
    np.std(
        r_test,
        ddof=0
    )
)

print(
    "Old raw alpha / response SD:",
    best_alpha
    / np.std(
        r_test,
        ddof=0
    )
)


# ============================================================================
# NOTEBOOK INDEX 85: Cell 82. Build a representative response-gene
# ============================================================================

# ============================================
# Cell 82. Build a representative response-gene
# panel for global-alpha calibration
# ============================================

import numpy as np
import pandas as pd

n_panel = 30

scale_sorted = (
    response_scale_df
    .sort_values("std")
    .reset_index(drop=True)
)

# Evenly spaced ranks across the full
# response-SD distribution
rank_idx = np.linspace(
    0,
    len(scale_sorted) - 1,
    n_panel,
    dtype=int
)

panel_genes = list(
    scale_sorted.loc[
        rank_idx,
        "gene"
    ]
)

# Force TACC3 into the calibration panel
# without introducing a duplicate
if "TACC3" not in panel_genes:
    panel_genes.append(
        "TACC3"
    )

panel_genes = list(
    dict.fromkeys(
        panel_genes
    )
)

panel_df = (
    response_scale_df
    .set_index("gene")
    .loc[panel_genes]
    .reset_index()
    .sort_values("std")
    .reset_index(drop=True)
)

print(
    "Calibration panel size:",
    len(panel_df)
)

print(
    "\nCalibration genes:"
)

print(
    panel_df[
        [
            "gene",
            "std",
            "rms",
            "n_rows",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nPanel response-SD range:"
)

print(
    "min   =",
    panel_df["std"].min()
)

print(
    "median=",
    panel_df["std"].median()
)

print(
    "max   =",
    panel_df["std"].max()
)

print(
    "\nFull response-SD range:"
)

print(
    "min   =",
    response_scale_df[
        "std"
    ].min()
)

print(
    "median=",
    response_scale_df[
        "std"
    ].median()
)

print(
    "max   =",
    response_scale_df[
        "std"
    ].max()
)


# ============================================================================
# NOTEBOOK INDEX 86: Cell 83. Calibrate dimensionless alpha
# ============================================================================

# ============================================
# Cell 83. Calibrate dimensionless alpha
# across the representative 31-gene panel
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

alpha_grid_global = np.logspace(
    -4,
    0,
    40
)

panel_cv_records = []

for gene_num, g in enumerate(
    panel_genes,
    start=1
):

    gi = np.where(
        response_genes == g
    )[0][0]

    valid_mask = valid_row_masks[
        g
    ]

    # Rate-form design
    dt_g = X_final[
        valid_mask,
        0
    ]

    Z_g = (
        X_final[
            valid_mask,
            1:
        ]
        / dt_g[:, None]
    )

    r_g = (
        Y_final[
            valid_mask,
            gi
        ]
        / dt_g
    )

    groups_g = row_conditions[
        valid_mask
    ]

    gkf_g = GroupKFold(
        n_splits=5
    )

    folds_g = list(
        gkf_g.split(
            Z_g,
            r_g,
            groups=groups_g
        )
    )

    for alpha in alpha_grid_global:

        fold_mse = []
        fold_nonzero = []

        for train_idx, val_idx in folds_g:

            Z_train = Z_g[
                train_idx
            ]

            Z_val = Z_g[
                val_idx
            ]

            r_train = r_g[
                train_idx
            ]

            r_val = r_g[
                val_idx
            ]

            # ------------------------------
            # X scaling: training fold only
            # ------------------------------

            x_scaler = StandardScaler()

            Z_train_z = (
                x_scaler.fit_transform(
                    Z_train
                )
            )

            Z_val_z = (
                x_scaler.transform(
                    Z_val
                )
            )

            # ------------------------------
            # y scaling: training fold only
            # ------------------------------

            r_mean = np.mean(
                r_train
            )

            r_std = np.std(
                r_train,
                ddof=0
            )

            r_train_z = (
                r_train
                - r_mean
            ) / r_std

            r_val_z = (
                r_val
                - r_mean
            ) / r_std

            model = Lasso(
                alpha=alpha,
                fit_intercept=True,
                max_iter=20000,
                tol=1e-6,
                selection="cyclic",
            )

            model.fit(
                Z_train_z,
                r_train_z
            )

            pred_z = model.predict(
                Z_val_z
            )

            fold_mse.append(
                mean_squared_error(
                    r_val_z,
                    pred_z
                )
            )

            fold_nonzero.append(
                np.count_nonzero(
                    model.coef_
                )
            )

        panel_cv_records.append({
            "gene": g,
            "alpha": alpha,
            "mean_mse_std":
                np.mean(
                    fold_mse
                ),
            "std_mse_std":
                np.std(
                    fold_mse,
                    ddof=1
                ),
            "mean_nonzero":
                np.mean(
                    fold_nonzero
                ),
        })

    print(
        f"[{gene_num:02d}/{len(panel_genes)}] "
        f"{g} done"
    )


panel_cv = pd.DataFrame(
    panel_cv_records
)


# --------------------------------------------
# Best alpha separately for each panel gene
# --------------------------------------------

best_panel_rows = []

for g in panel_genes:

    tmp = panel_cv[
        panel_cv["gene"] == g
    ]

    idx_best = tmp[
        "mean_mse_std"
    ].idxmin()

    best_panel_rows.append(
        panel_cv.loc[
            idx_best
        ]
    )

best_panel_alpha = pd.DataFrame(
    best_panel_rows
).reset_index(
    drop=True
)


print(
    "\nPer-gene optimal alpha:"
)

print(
    best_panel_alpha[
        [
            "gene",
            "alpha",
            "mean_mse_std",
            "mean_nonzero",
        ]
    ]
    .sort_values(
        "alpha"
    )
    .to_string(
        index=False
    )
)


print(
    "\nOptimal-alpha summary:"
)

print(
    best_panel_alpha[
        "alpha"
    ].describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
)


print(
    "\nGenes whose optimum is "
    "at grid boundary:"
)

boundary = (
    np.isclose(
        best_panel_alpha[
            "alpha"
        ],
        alpha_grid_global.min()
    )
    |
    np.isclose(
        best_panel_alpha[
            "alpha"
        ],
        alpha_grid_global.max()
    )
)

print(
    best_panel_alpha.loc[
        boundary,
        [
            "gene",
            "alpha",
            "mean_mse_std",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nBoundary count:",
    int(
        boundary.sum()
    ),
    "/",
    len(
        best_panel_alpha
    )
)


# ============================================================================
# NOTEBOOK INDEX 87: Cell 84. Select one global dimensionless
# ============================================================================

# ============================================
# Cell 84. Select one global dimensionless
# alpha by aggregate held-out CV loss
# ============================================

import numpy as np
import pandas as pd

global_alpha_curve = (
    panel_cv
    .groupby(
        "alpha",
        as_index=False
    )
    .agg(
        mean_gene_mse=(
            "mean_mse_std",
            "mean"
        ),
        median_gene_mse=(
            "mean_mse_std",
            "median"
        ),
        std_gene_mse=(
            "mean_mse_std",
            "std"
        ),
        mean_nonzero=(
            "mean_nonzero",
            "mean"
        ),
        median_nonzero=(
            "mean_nonzero",
            "median"
        ),
    )
    .sort_values(
        "alpha"
    )
    .reset_index(
        drop=True
    )
)

best_global_idx = (
    global_alpha_curve[
        "mean_gene_mse"
    ].idxmin()
)

best_global_row = (
    global_alpha_curve.loc[
        best_global_idx
    ]
)

global_alpha = float(
    best_global_row[
        "alpha"
    ]
)


print(
    "Global alpha selected by "
    "mean standardized held-out MSE:"
)

print(
    best_global_row
)


print(
    "\nTop 10 global alpha candidates:"
)

print(
    global_alpha_curve
    .sort_values(
        "mean_gene_mse"
    )
    .head(10)
    .to_string(
        index=False
    )
)


# --------------------------------------------
# How much worse is the global alpha than
# each gene's own individually optimal alpha?
# --------------------------------------------

gene_optimum = (
    panel_cv
    .groupby("gene")[
        "mean_mse_std"
    ]
    .min()
)

global_gene_loss = (
    panel_cv[
        np.isclose(
            panel_cv["alpha"],
            global_alpha
        )
    ]
    .set_index("gene")[
        "mean_mse_std"
    ]
)

relative_excess = (
    global_gene_loss
    / gene_optimum
    - 1.0
)

print(
    "\nRelative excess CV loss from using "
    "one global alpha:"
)

print(
    relative_excess.describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nGenes most affected by global alpha:"
)

comparison = pd.DataFrame({
    "optimal_mse":
        gene_optimum,
    "global_mse":
        global_gene_loss,
    "relative_excess":
        relative_excess,
}).sort_values(
    "relative_excess",
    ascending=False
)

print(
    comparison
    .head(10)
    .to_string()
)


print(
    "\nGlobal alpha at grid boundary:",
    (
        best_global_idx == 0
        or
        best_global_idx
        == len(
            global_alpha_curve
        ) - 1
    )
)


# ============================================================================
# NOTEBOOK INDEX 88: Cell 85. Reconstruct the full 426 x 151 GRN
# ============================================================================

# ============================================
# Cell 85. Reconstruct the full 426 x 151 GRN
# using the frozen global dimensionless alpha
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

n_response = len(response_genes)
n_regulator = len(regulator_genes_vc)

A_hat = np.zeros(
    (n_response, n_regulator),
    dtype=np.float64
)

c_hat = np.zeros(
    n_response,
    dtype=np.float64
)

fit_summary_rows = []

for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

    dt_g = np.asarray(
        X_final[
            valid_mask,
            0
        ],
        dtype=np.float64
    )

    Z_g = np.asarray(
        X_final[
            valid_mask,
            1:
        ],
        dtype=np.float64
    ) / dt_g[:, None]

    r_g = np.asarray(
        Y_final[
            valid_mask,
            gi
        ],
        dtype=np.float64
    ) / dt_g

    # ----------------------------------------
    # Predictor scaling on all valid rows
    # ----------------------------------------

    x_scaler = StandardScaler()

    Z_g_z = x_scaler.fit_transform(
        Z_g
    )

    # ----------------------------------------
    # Response scaling
    # ----------------------------------------

    r_mean = np.mean(
        r_g
    )

    r_std = np.std(
        r_g,
        ddof=0
    )

    assert (
        np.isfinite(r_std)
        and r_std > 0
    )

    r_g_z = (
        r_g
        - r_mean
    ) / r_std

    # ----------------------------------------
    # Frozen global-alpha Lasso
    # ----------------------------------------

    model = Lasso(
        alpha=global_alpha,
        fit_intercept=True,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z_g_z,
        r_g_z
    )

    # ----------------------------------------
    # Back-transform to ORIGINAL model scale
    #
    # r = c_g + Z A_g
    # ----------------------------------------

    A_g = (
        r_std
        * model.coef_
        / x_scaler.scale_
    )

    c_g = (
        r_mean
        + r_std
        * model.intercept_
        - np.sum(
            A_g
            * x_scaler.mean_
        )
    )

    A_hat[
        gi, :
    ] = A_g

    c_hat[
        gi
    ] = c_g

    # ----------------------------------------
    # Original-scale prediction sanity check
    # ----------------------------------------

    pred_original = (
        c_g
        + Z_g @ A_g
    )

    pred_scaled_back = (
        r_mean
        + r_std
        * model.predict(
            Z_g_z
        )
    )

    transform_ok = np.allclose(
        pred_original,
        pred_scaled_back,
        rtol=1e-8,
        atol=1e-8
    )

    ss_res = np.sum(
        (
            r_g
            - pred_original
        ) ** 2
    )

    ss_tot = np.sum(
        (
            r_g
            - np.mean(r_g)
        ) ** 2
    )

    train_r2 = (
        1.0
        - ss_res / ss_tot
    )

    fit_summary_rows.append({
        "gene": g,
        "n_rows":
            len(r_g),
        "response_std":
            r_std,
        "n_nonzero":
            int(
                np.count_nonzero(
                    A_g
                )
            ),
        "train_r2":
            train_r2,
        "transform_ok":
            transform_ok,
    })


fit_summary = pd.DataFrame(
    fit_summary_rows
)


print(
    "A_hat shape:",
    A_hat.shape
)

print(
    "c_hat shape:",
    c_hat.shape
)

print(
    "Global alpha:",
    global_alpha
)

print(
    "\nNaN/inf in A_hat:",
    (
        ~np.isfinite(
            A_hat
        )
    ).any()
)

print(
    "NaN/inf in c_hat:",
    (
        ~np.isfinite(
            c_hat
        )
    ).any()
)

print(
    "All back-transforms exact:",
    fit_summary[
        "transform_ok"
    ].all()
)


print(
    "\nNonzero edges per response gene:"
)

print(
    fit_summary[
        "n_nonzero"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


print(
    "\nTotal nonzero edges:",
    np.count_nonzero(
        A_hat
    )
)

print(
    "Total possible edges:",
    A_hat.size
)

print(
    "Network density:",
    np.count_nonzero(
        A_hat
    )
    / A_hat.size
)


print(
    "\nTraining R^2 summary "
    "(diagnostic only):"
)

print(
    fit_summary[
        "train_r2"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


# ============================================================================
# NOTEBOOK INDEX 89: Cell 86. Strict held-out perturbation
# ============================================================================

# ============================================
# Cell 86. Strict held-out perturbation
# prediction for TACC3
#
# - frozen global alpha
# - NT always stays in training
# - validation contains perturbations only
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

test_gene = "TACC3"

gi = np.where(
    response_genes == test_gene
)[0][0]

valid_mask = valid_row_masks[
    test_gene
]

dt_g = np.asarray(
    X_final[
        valid_mask, 0
    ],
    dtype=np.float64
)

Z_g = np.asarray(
    X_final[
        valid_mask, 1:
    ],
    dtype=np.float64
) / dt_g[:, None]

r_g = np.asarray(
    Y_final[
        valid_mask, gi
    ],
    dtype=np.float64
) / dt_g

groups_g = row_conditions[
    valid_mask
]


# --------------------------------------------
# Split perturbation CONDITIONS, not rows.
# NT is never held out.
# --------------------------------------------

pert_conditions = np.asarray([
    q for q in np.unique(groups_g)
    if q != "non-targeting"
], dtype=object)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=2026
)

oof_pred = np.full(
    len(r_g),
    np.nan,
    dtype=np.float64
)

fold_records = []

for fold, (_, val_cond_idx) in enumerate(
    kf.split(pert_conditions),
    start=1
):

    val_conditions = pert_conditions[
        val_cond_idx
    ]

    val_mask_fold = np.isin(
        groups_g,
        val_conditions
    )

    train_mask_fold = (
        ~val_mask_fold
    )

    # NT must always remain in training
    assert np.any(
        groups_g[
            train_mask_fold
        ] == "non-targeting"
    )

    assert not np.any(
        groups_g[
            val_mask_fold
        ] == "non-targeting"
    )

    Z_train = Z_g[
        train_mask_fold
    ]

    Z_val = Z_g[
        val_mask_fold
    ]

    r_train = r_g[
        train_mask_fold
    ]

    r_val = r_g[
        val_mask_fold
    ]

    # ----------------------------------------
    # Training-only predictor scaling
    # ----------------------------------------

    x_scaler = StandardScaler()

    Z_train_z = x_scaler.fit_transform(
        Z_train
    )

    Z_val_z = x_scaler.transform(
        Z_val
    )

    # ----------------------------------------
    # Training-only response scaling
    # ----------------------------------------

    r_mean = np.mean(
        r_train
    )

    r_std = np.std(
        r_train,
        ddof=0
    )

    r_train_z = (
        r_train - r_mean
    ) / r_std

    # ----------------------------------------
    # Frozen global alpha
    # ----------------------------------------

    model = Lasso(
        alpha=global_alpha,
        fit_intercept=True,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z_train_z,
        r_train_z
    )

    pred_z = model.predict(
        Z_val_z
    )

    pred = (
        r_mean
        + r_std * pred_z
    )

    oof_pred[
        val_mask_fold
    ] = pred

    # ----------------------------------------
    # Baseline:
    # predict training mean response rate
    # ----------------------------------------

    baseline = np.full(
        len(r_val),
        r_mean
    )

    sse_model = np.sum(
        (r_val - pred) ** 2
    )

    sse_baseline = np.sum(
        (r_val - baseline) ** 2
    )

    fold_records.append({
        "fold": fold,
        "n_val_conditions":
            len(val_conditions),
        "n_val_rows":
            int(val_mask_fold.sum()),
        "mse":
            np.mean(
                (r_val - pred) ** 2
            ),
        "baseline_mse":
            np.mean(
                (r_val - baseline) ** 2
            ),
        "relative_mse":
            sse_model
            / sse_baseline,
        "n_nonzero":
            np.count_nonzero(
                model.coef_
            ),
    })


fold_oof_tacc3 = pd.DataFrame(
    fold_records
)

# Only perturbation rows receive OOF predictions
eval_mask = np.isfinite(
    oof_pred
)

r_eval = r_g[
    eval_mask
]

pred_eval = oof_pred[
    eval_mask
]

# Global OOF R^2 relative to the observed
# held-out perturbation response distribution
ss_res = np.sum(
    (r_eval - pred_eval) ** 2
)

ss_tot = np.sum(
    (
        r_eval
        - np.mean(r_eval)
    ) ** 2
)

oof_r2 = (
    1.0
    - ss_res / ss_tot
)

print(
    "Test gene:",
    test_gene
)

print(
    "Held-out perturbation conditions:",
    len(pert_conditions)
)

print(
    "OOF rows:",
    eval_mask.sum()
)

print(
    "NT rows predicted:",
    np.sum(
        eval_mask
        & (
            groups_g
            == "non-targeting"
        )
    )
)

print(
    "\nFold results:"
)

print(
    fold_oof_tacc3.to_string(
        index=False
    )
)

print(
    "\nOverall perturbation-held-out OOF R^2:",
    oof_r2
)

print(
    "Mean fold relative MSE:",
    fold_oof_tacc3[
        "relative_mse"
    ].mean()
)

print(
    "Median fold relative MSE:",
    fold_oof_tacc3[
        "relative_mse"
    ].median()
)

print(
    "\nAll perturbation rows predicted once:",
    np.all(
        np.isfinite(
            oof_pred[
                groups_g
                != "non-targeting"
            ]
        )
    )
)


# ============================================================================
# NOTEBOOK INDEX 90: Cell 87. Full 426-gene perturbation-held-out
# ============================================================================

# ============================================
# Cell 87. Full 426-gene perturbation-held-out
# OOF evaluation with frozen global alpha
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

rng_seed = 2026
n_folds = 5

oof_gene_records = []

for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

    dt_g = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Z_g = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    ) / dt_g[:, None]

    r_g = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    ) / dt_g

    groups_g = row_conditions[
        valid_mask
    ]

    # ----------------------------------------
    # Only perturbations are held out.
    # NT remains training reference.
    # ----------------------------------------

    pert_conditions = np.asarray([
        q
        for q in np.unique(groups_g)
        if q != "non-targeting"
    ], dtype=object)

    kf = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=rng_seed
    )

    oof_pred = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    oof_baseline = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    fold_nonzero = []

    for _, val_cond_idx in kf.split(
        pert_conditions
    ):

        val_conditions = pert_conditions[
            val_cond_idx
        ]

        val_fold = np.isin(
            groups_g,
            val_conditions
        )

        train_fold = ~val_fold

        # NT must always be training data
        assert np.any(
            groups_g[
                train_fold
            ] == "non-targeting"
        )

        assert not np.any(
            groups_g[
                val_fold
            ] == "non-targeting"
        )

        Z_train = Z_g[
            train_fold
        ]

        Z_val = Z_g[
            val_fold
        ]

        r_train = r_g[
            train_fold
        ]

        # ------------------------------------
        # Training-only X scaling
        # ------------------------------------

        x_scaler = StandardScaler()

        Z_train_z = (
            x_scaler.fit_transform(
                Z_train
            )
        )

        Z_val_z = (
            x_scaler.transform(
                Z_val
            )
        )

        # ------------------------------------
        # Training-only y scaling
        # ------------------------------------

        r_mean = np.mean(
            r_train
        )

        r_std = np.std(
            r_train,
            ddof=0
        )

        assert (
            np.isfinite(r_std)
            and r_std > 0
        )

        r_train_z = (
            r_train
            - r_mean
        ) / r_std

        # ------------------------------------
        # Frozen global alpha
        # ------------------------------------

        model = Lasso(
            alpha=global_alpha,
            fit_intercept=True,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z_train_z,
            r_train_z
        )

        pred_z = model.predict(
            Z_val_z
        )

        pred = (
            r_mean
            + r_std * pred_z
        )

        oof_pred[
            val_fold
        ] = pred

        oof_baseline[
            val_fold
        ] = r_mean

        fold_nonzero.append(
            np.count_nonzero(
                model.coef_
            )
        )

    # ----------------------------------------
    # Evaluate perturbation rows only
    # ----------------------------------------

    eval_mask = (
        groups_g
        != "non-targeting"
    )

    assert np.all(
        np.isfinite(
            oof_pred[
                eval_mask
            ]
        )
    )

    assert not np.any(
        np.isfinite(
            oof_pred[
                ~eval_mask
            ]
        )
    )

    y_eval = r_g[
        eval_mask
    ]

    p_eval = oof_pred[
        eval_mask
    ]

    b_eval = oof_baseline[
        eval_mask
    ]

    sse_model = np.sum(
        (
            y_eval
            - p_eval
        ) ** 2
    )

    sse_baseline = np.sum(
        (
            y_eval
            - b_eval
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y_eval
            - np.mean(
                y_eval
            )
        ) ** 2
    )

    oof_r2 = (
        1.0
        - sse_model / ss_tot
    )

    relative_mse = (
        sse_model
        / sse_baseline
    )

    oof_gene_records.append({
        "gene": g,
        "n_eval_rows":
            int(
                eval_mask.sum()
            ),
        "n_eval_conditions":
            len(
                pert_conditions
            ),
        "oof_r2":
            oof_r2,
        "relative_mse":
            relative_mse,
        "mean_fold_nonzero":
            np.mean(
                fold_nonzero
            ),
    })

    if (
        (gi + 1) % 25 == 0
        or gi == 0
        or gi + 1 == len(
            response_genes
        )
    ):
        print(
            f"[{gi + 1:03d}/"
            f"{len(response_genes)}] "
            f"{g} done"
        )


oof_gene_summary = pd.DataFrame(
    oof_gene_records
)


print(
    "\nOOF R^2 summary:"
)

print(
    oof_gene_summary[
        "oof_r2"
    ].describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nRelative MSE summary:"
)

print(
    oof_gene_summary[
        "relative_mse"
    ].describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nFraction with OOF R^2 > 0:"
)

print(
    np.mean(
        oof_gene_summary[
            "oof_r2"
        ] > 0
    )
)


print(
    "\nFraction beating fold-specific "
    "mean baseline:"
)

print(
    np.mean(
        oof_gene_summary[
            "relative_mse"
        ] < 1
    )
)


print(
    "\nWorst 10 genes by OOF R^2:"
)

print(
    oof_gene_summary
    .sort_values(
        "oof_r2"
    )
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nBest 10 genes by OOF R^2:"
)

print(
    oof_gene_summary
    .sort_values(
        "oof_r2",
        ascending=False
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 91: Cell 88. Save reconstruction + OOF checkpoint
# ============================================================================

# ============================================
# Cell 88. Save reconstruction + OOF checkpoint
# ============================================

import numpy as np
import pandas as pd
import os

checkpoint_path = (
    "/home/featurize/work/project1/"
    "replogle_grn_reconstruction_checkpoint.npz"
)

np.savez_compressed(
    checkpoint_path,

    # Final reconstructed GRN
    A_hat=A_hat,
    c_hat=c_hat,

    response_genes=np.asarray(
        response_genes,
        dtype=str
    ),

    regulator_genes=np.asarray(
        regulator_genes_vc,
        dtype=str
    ),

    global_alpha=np.asarray(
        global_alpha
    ),

    # Final integral system
    X_final=X_final,
    Y_final=Y_final,

    # OOF summary
    oof_gene_names=np.asarray(
        oof_gene_summary["gene"],
        dtype=str
    ),

    oof_r2=np.asarray(
        oof_gene_summary["oof_r2"],
        dtype=float
    ),

    relative_mse=np.asarray(
        oof_gene_summary["relative_mse"],
        dtype=float
    ),

    mean_fold_nonzero=np.asarray(
        oof_gene_summary[
            "mean_fold_nonzero"
        ],
        dtype=float
    ),
)

print(
    "Saved:",
    checkpoint_path
)

print(
    "Exists:",
    os.path.exists(
        checkpoint_path
    )
)

print(
    "Size MB:",
    os.path.getsize(
        checkpoint_path
    ) / 1024**2
)


# --------------------------------------------
# Reload immediately and verify critical arrays
# --------------------------------------------

check = np.load(
    checkpoint_path,
    allow_pickle=False
)

print(
    "\nSaved keys:"
)

print(
    check.files
)

print(
    "\nA_hat exact:",
    np.array_equal(
        check["A_hat"],
        A_hat
    )
)

print(
    "c_hat exact:",
    np.array_equal(
        check["c_hat"],
        c_hat
    )
)

print(
    "OOF R2 exact:",
    np.array_equal(
        check["oof_r2"],
        oof_gene_summary[
            "oof_r2"
        ].to_numpy()
    )
)

print(
    "Global alpha exact:",
    float(
        check["global_alpha"]
    ) == global_alpha
)


# ============================================================================
# NOTEBOOK INDEX 92: Cell 89. TACC3 condition-permutation null
# ============================================================================

# ============================================
# Cell 89. TACC3 condition-permutation null
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

test_gene = "TACC3"

gi = np.where(
    response_genes == test_gene
)[0][0]

valid_mask = valid_row_masks[
    test_gene
]

dt_g = np.asarray(
    X_final[
        valid_mask, 0
    ],
    dtype=np.float64
)

Z_g = np.asarray(
    X_final[
        valid_mask, 1:
    ],
    dtype=np.float64
) / dt_g[:, None]

r_g = np.asarray(
    Y_final[
        valid_mask, gi
    ],
    dtype=np.float64
) / dt_g

groups_g = row_conditions[
    valid_mask
]

pert_conditions = np.asarray([
    q for q in np.unique(groups_g)
    if q != "non-targeting"
], dtype=object)


# --------------------------------------------
# Build ONE condition-level permutation
# --------------------------------------------

rng = np.random.default_rng(
    2026
)

permuted_conditions = rng.permutation(
    pert_conditions
)

condition_map = dict(
    zip(
        pert_conditions,
        permuted_conditions
    )
)

# NT maps to itself
condition_map[
    "non-targeting"
] = "non-targeting"


# --------------------------------------------
# Construct permuted regulator design:
#
# response rows for condition q receive
# regulator trajectory rows from permuted q'
#
# Match rows by bin interval index.
# --------------------------------------------

interval_keys = np.asarray([
    (
        row["condition"],
        row["bin_a"],
        row["bin_b"]
    )
    for row in np.asarray(
        final_interval_rows,
        dtype=object
    )[valid_mask]
], dtype=object)

lookup = {}

for i, (
    q,
    bin_a,
    bin_b
) in enumerate(
    interval_keys
):
    lookup[
        (
            q,
            int(bin_a),
            int(bin_b)
        )
    ] = i


Z_perm = np.full_like(
    Z_g,
    np.nan
)

usable_perm = np.zeros(
    len(Z_g),
    dtype=bool
)

for i, (
    q,
    bin_a,
    bin_b
) in enumerate(
    interval_keys
):

    q_source = condition_map[q]

    key = (
        q_source,
        int(bin_a),
        int(bin_b)
    )

    if key in lookup:

        j = lookup[key]

        Z_perm[
            i, :
        ] = Z_g[
            j, :
        ]

        usable_perm[
            i
        ] = True


print(
    "Permuted rows retained:",
    usable_perm.sum(),
    "/",
    len(usable_perm)
)

print(
    "Retention fraction:",
    usable_perm.mean()
)

print(
    "NaN/inf in retained Z_perm:",
    (
        ~np.isfinite(
            Z_perm[
                usable_perm
            ]
        )
    ).any()
)


# --------------------------------------------
# Evaluate only rows for which matching
# source condition/bin interval exists
# --------------------------------------------

Z_null = Z_perm[
    usable_perm
]

r_null = r_g[
    usable_perm
]

groups_null = groups_g[
    usable_perm
]

pert_null = np.asarray([
    q for q in np.unique(groups_null)
    if q != "non-targeting"
], dtype=object)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=2026
)

oof_pred_null = np.full(
    len(r_null),
    np.nan,
    dtype=np.float64
)

for _, val_cond_idx in kf.split(
    pert_null
):

    val_conditions = pert_null[
        val_cond_idx
    ]

    val_fold = np.isin(
        groups_null,
        val_conditions
    )

    train_fold = ~val_fold

    Z_train = Z_null[
        train_fold
    ]

    Z_val = Z_null[
        val_fold
    ]

    r_train = r_null[
        train_fold
    ]

    r_val = r_null[
        val_fold
    ]

    x_scaler = StandardScaler()

    Z_train_z = x_scaler.fit_transform(
        Z_train
    )

    Z_val_z = x_scaler.transform(
        Z_val
    )

    r_mean = np.mean(
        r_train
    )

    r_std = np.std(
        r_train,
        ddof=0
    )

    r_train_z = (
        r_train
        - r_mean
    ) / r_std

    model = Lasso(
        alpha=global_alpha,
        fit_intercept=True,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z_train_z,
        r_train_z
    )

    pred_z = model.predict(
        Z_val_z
    )

    oof_pred_null[
        val_fold
    ] = (
        r_mean
        + r_std * pred_z
    )


eval_mask = (
    groups_null
    != "non-targeting"
)

y_eval = r_null[
    eval_mask
]

p_eval = oof_pred_null[
    eval_mask
]

ss_res = np.sum(
    (
        y_eval
        - p_eval
    ) ** 2
)

ss_tot = np.sum(
    (
        y_eval
        - np.mean(y_eval)
    ) ** 2
)

null_oof_r2 = (
    1.0
    - ss_res / ss_tot
)

print(
    "\nObserved TACC3 OOF R^2:",
    float(
        oof_gene_summary.loc[
            oof_gene_summary[
                "gene"
            ] == "TACC3",
            "oof_r2"
        ].iloc[0]
    )
)

print(
    "Permuted-condition OOF R^2:",
    null_oof_r2
)

print(
    "R^2 drop:",
    float(
        oof_gene_summary.loc[
            oof_gene_summary[
                "gene"
            ] == "TACC3",
            "oof_r2"
        ].iloc[0]
    )
    - null_oof_r2
)


# ============================================================================
# NOTEBOOK INDEX 93: Cell 90. Phase-only null for all 426 genes
# ============================================================================

# ============================================
# Cell 90. Phase-only null for all 426 genes
#
# Question:
# How much held-out performance is explained
# WITHOUT using any regulator trajectories?
#
# Model:
# r_g(q, phi) =
#   condition-independent Fourier function
#   of phase only
#
# NT always remains in training.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

phase_only_records = []

for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

    dt_g = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    r_g = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    ) / dt_g

    groups_g = row_conditions[
        valid_mask
    ]

    rows_g = np.asarray(
        final_interval_rows,
        dtype=object
    )[valid_mask]

    # ----------------------------------------
    # Interval midpoint phase
    # bins are 0..9 with centers
    # 0.05, 0.15, ..., 0.95
    #
    # Adjacent interval midpoint:
    # 0.10, 0.20, ..., 0.90
    # ----------------------------------------

    phase_mid = np.asarray([
        0.5 * (
            bin_centers[
                row["bin_a"]
            ]
            +
            bin_centers[
                row["bin_b"]
            ]
        )
        for row in rows_g
    ], dtype=np.float64)

    theta = (
        2.0
        * np.pi
        * phase_mid
    )

    # Phase-only Fourier design.
    # Two harmonics gives enough flexibility
    # to represent shared cell-cycle shape.
    P_g = np.column_stack([
        np.ones(
            len(theta)
        ),
        np.sin(theta),
        np.cos(theta),
        np.sin(
            2.0 * theta
        ),
        np.cos(
            2.0 * theta
        ),
    ])

    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_g
        )
        if q != "non-targeting"
    ], dtype=object)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=2026
    )

    oof_pred_phase = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    for _, val_cond_idx in kf.split(
        pert_conditions
    ):

        val_conditions = pert_conditions[
            val_cond_idx
        ]

        val_fold = np.isin(
            groups_g,
            val_conditions
        )

        train_fold = ~val_fold

        # ------------------------------------
        # Ordinary least squares phase-only
        # model on training conditions
        # ------------------------------------

        coef_phase, _, _, _ = (
            np.linalg.lstsq(
                P_g[
                    train_fold
                ],
                r_g[
                    train_fold
                ],
                rcond=None
            )
        )

        oof_pred_phase[
            val_fold
        ] = (
            P_g[
                val_fold
            ]
            @ coef_phase
        )

    eval_mask = (
        groups_g
        != "non-targeting"
    )

    assert np.all(
        np.isfinite(
            oof_pred_phase[
                eval_mask
            ]
        )
    )

    y_eval = r_g[
        eval_mask
    ]

    pred_eval = oof_pred_phase[
        eval_mask
    ]

    ss_res = np.sum(
        (
            y_eval
            - pred_eval
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y_eval
            - np.mean(
                y_eval
            )
        ) ** 2
    )

    phase_r2 = (
        1.0
        - ss_res / ss_tot
    )

    grn_r2 = float(
        oof_gene_summary.loc[
            oof_gene_summary[
                "gene"
            ] == g,
            "oof_r2"
        ].iloc[0]
    )

    phase_only_records.append({
        "gene": g,
        "grn_oof_r2":
            grn_r2,
        "phase_only_oof_r2":
            phase_r2,
        "incremental_r2":
            grn_r2
            - phase_r2,
    })


phase_only_summary = pd.DataFrame(
    phase_only_records
)


print(
    "Phase-only OOF R^2 summary:"
)

print(
    phase_only_summary[
        "phase_only_oof_r2"
    ].describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nGRN minus phase-only R^2:"
)

print(
    phase_only_summary[
        "incremental_r2"
    ].describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nFraction GRN > phase-only:"
)

print(
    np.mean(
        phase_only_summary[
            "incremental_r2"
        ] > 0
    )
)


print(
    "\nTACC3:"
)

print(
    phase_only_summary[
        phase_only_summary[
            "gene"
        ] == "TACC3"
    ].to_string(
        index=False
    )
)


print(
    "\nGenes with largest GRN gain "
    "over phase-only:"
)

print(
    phase_only_summary
    .sort_values(
        "incremental_r2",
        ascending=False
    )
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 94: Cell 91. Residualize shared phase structure
# ============================================================================

# ============================================
# Cell 91. Residualize shared phase structure
# from response rates and regulator features
# ============================================

import numpy as np

# --------------------------------------------
# Rate-form system
# --------------------------------------------

dt_all = np.asarray(
    X_final[:, 0],
    dtype=np.float64
)

Z_rate = np.asarray(
    X_final[:, 1:],
    dtype=np.float64
) / dt_all[:, None]

R_rate = np.asarray(
    Y_final,
    dtype=np.float64
) / dt_all[:, None]


# --------------------------------------------
# Interval midpoint phase
# --------------------------------------------

phase_mid_all = np.asarray([
    0.5 * (
        bin_centers[
            row["bin_a"]
        ]
        +
        bin_centers[
            row["bin_b"]
        ]
    )
    for row in final_interval_rows
], dtype=np.float64)

theta_all = (
    2.0
    * np.pi
    * phase_mid_all
)


# --------------------------------------------
# Shared phase basis:
# intercept + first two Fourier harmonics
# --------------------------------------------

P_all = np.column_stack([
    np.ones(
        len(theta_all)
    ),
    np.sin(
        theta_all
    ),
    np.cos(
        theta_all
    ),
    np.sin(
        2.0 * theta_all
    ),
    np.cos(
        2.0 * theta_all
    ),
])


# --------------------------------------------
# Residualize regulator features
#
# Z_res = Z - projection onto phase basis
# --------------------------------------------

coef_phase_Z, _, _, _ = np.linalg.lstsq(
    P_all,
    Z_rate,
    rcond=None
)

Z_res = (
    Z_rate
    - P_all @ coef_phase_Z
)


# --------------------------------------------
# Residualize response rates
# independently for every response gene
# --------------------------------------------

coef_phase_R, _, _, _ = np.linalg.lstsq(
    P_all,
    R_rate,
    rcond=None
)

R_res = (
    R_rate
    - P_all @ coef_phase_R
)


print(
    "Z_res shape:",
    Z_res.shape
)

print(
    "R_res shape:",
    R_res.shape
)

print(
    "NaN/inf in Z_res:",
    (
        ~np.isfinite(
            Z_res
        )
    ).any()
)

print(
    "NaN/inf in R_res:",
    (
        ~np.isfinite(
            R_res
        )
    ).any()
)


# --------------------------------------------
# Orthogonality checks
# --------------------------------------------

phase_corr_Z = np.max(
    np.abs(
        P_all.T @ Z_res
    )
)

phase_corr_R = np.max(
    np.abs(
        P_all.T @ R_res
    )
)

print(
    "\nMax |P^T Z_res|:",
    phase_corr_Z
)

print(
    "Max |P^T R_res|:",
    phase_corr_R
)


# Relative reconstruction energy remaining
Z_energy_fraction = (
    np.sum(
        Z_res ** 2
    )
    /
    np.sum(
        Z_rate ** 2
    )
)

R_energy_fraction = (
    np.sum(
        R_res ** 2
    )
    /
    np.sum(
        R_rate ** 2
    )
)

print(
    "\nRegulator residual energy fraction:",
    Z_energy_fraction
)

print(
    "Response residual energy fraction:",
    R_energy_fraction
)


# --------------------------------------------
# TACC3 specifically
# --------------------------------------------

tacc3_idx = np.where(
    response_genes == "TACC3"
)[0][0]

tacc3_total_var = np.var(
    R_rate[
        :, tacc3_idx
    ]
)

tacc3_res_var = np.var(
    R_res[
        :, tacc3_idx
    ]
)

print(
    "\nTACC3 residual variance fraction:",
    tacc3_res_var
    / tacc3_total_var
)


# ============================================================================
# NOTEBOOK INDEX 95: Cell 92. Strict cross-fitted phase-residual
# ============================================================================

# ============================================
# Cell 92. Strict cross-fitted phase-residual
# OOF test for TACC3
#
# Question:
# After removing phase structure using
# TRAINING CONDITIONS ONLY, do regulator
# residuals predict held-out response residuals?
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

test_gene = "TACC3"

gi = np.where(
    response_genes == test_gene
)[0][0]

valid_mask = valid_row_masks[
    test_gene
]

Z_g = Z_rate[
    valid_mask
]

r_g = R_rate[
    valid_mask,
    gi
]

P_g = P_all[
    valid_mask
]

groups_g = row_conditions[
    valid_mask
]


pert_conditions = np.asarray([
    q
    for q in np.unique(
        groups_g
    )
    if q != "non-targeting"
], dtype=object)


kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=2026
)

oof_y_res = np.full(
    len(r_g),
    np.nan
)

oof_pred_res = np.full(
    len(r_g),
    np.nan
)

oof_zero_pred = np.full(
    len(r_g),
    np.nan
)

fold_records = []


for fold, (_, val_cond_idx) in enumerate(
    kf.split(
        pert_conditions
    ),
    start=1
):

    val_conditions = pert_conditions[
        val_cond_idx
    ]

    val_fold = np.isin(
        groups_g,
        val_conditions
    )

    train_fold = ~val_fold

    # ----------------------------------------
    # 1. Learn phase components on TRAIN only
    # ----------------------------------------

    phase_coef_Z, _, _, _ = (
        np.linalg.lstsq(
            P_g[
                train_fold
            ],
            Z_g[
                train_fold
            ],
            rcond=None
        )
    )

    phase_coef_r, _, _, _ = (
        np.linalg.lstsq(
            P_g[
                train_fold
            ],
            r_g[
                train_fold
            ],
            rcond=None
        )
    )


    # ----------------------------------------
    # 2. Residualize train and validation
    # using TRAIN-fitted phase model
    # ----------------------------------------

    Z_train_res = (
        Z_g[
            train_fold
        ]
        -
        P_g[
            train_fold
        ]
        @ phase_coef_Z
    )

    Z_val_res = (
        Z_g[
            val_fold
        ]
        -
        P_g[
            val_fold
        ]
        @ phase_coef_Z
    )

    r_train_res = (
        r_g[
            train_fold
        ]
        -
        P_g[
            train_fold
        ]
        @ phase_coef_r
    )

    r_val_res = (
        r_g[
            val_fold
        ]
        -
        P_g[
            val_fold
        ]
        @ phase_coef_r
    )


    # ----------------------------------------
    # 3. Training-only scaling
    # ----------------------------------------

    x_scaler = StandardScaler()

    Z_train_z = (
        x_scaler.fit_transform(
            Z_train_res
        )
    )

    Z_val_z = (
        x_scaler.transform(
            Z_val_res
        )
    )

    r_scale = np.std(
        r_train_res,
        ddof=0
    )

    assert (
        np.isfinite(r_scale)
        and r_scale > 0
    )

    # Phase residuals already have
    # training mean approximately zero.
    r_train_z = (
        r_train_res
        / r_scale
    )


    # ----------------------------------------
    # 4. Frozen-alpha sparse regression
    #
    # No intercept:
    # phase projection already contains
    # the constant component.
    # ----------------------------------------

    model = Lasso(
        alpha=global_alpha,
        fit_intercept=False,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z_train_z,
        r_train_z
    )

    pred_res = (
        r_scale
        * model.predict(
            Z_val_z
        )
    )


    oof_y_res[
        val_fold
    ] = r_val_res

    oof_pred_res[
        val_fold
    ] = pred_res

    # Null after phase removal:
    # no regulatory contribution
    oof_zero_pred[
        val_fold
    ] = 0.0


    sse_model = np.sum(
        (
            r_val_res
            - pred_res
        ) ** 2
    )

    sse_null = np.sum(
        r_val_res ** 2
    )

    fold_records.append({
        "fold": fold,
        "n_val_conditions":
            len(val_conditions),
        "n_val_rows":
            int(
                val_fold.sum()
            ),
        "relative_mse_vs_phase":
            sse_model
            / sse_null,
        "incremental_r2":
            1.0
            - sse_model
            / sse_null,
        "n_nonzero":
            int(
                np.count_nonzero(
                    model.coef_
                )
            ),
    })


resid_fold_tacc3 = pd.DataFrame(
    fold_records
)


# --------------------------------------------
# Aggregate held-out perturbation result
# --------------------------------------------

eval_mask = (
    groups_g
    != "non-targeting"
)

y_eval = oof_y_res[
    eval_mask
]

p_eval = oof_pred_res[
    eval_mask
]

assert np.all(
    np.isfinite(
        y_eval
    )
)

assert np.all(
    np.isfinite(
        p_eval
    )
)

sse_model = np.sum(
    (
        y_eval
        - p_eval
    ) ** 2
)

sse_phase_only = np.sum(
    y_eval ** 2
)

incremental_r2 = (
    1.0
    - sse_model
    / sse_phase_only
)


print(
    "TACC3 cross-fitted "
    "phase-residual evaluation:"
)

print(
    resid_fold_tacc3.to_string(
        index=False
    )
)

print(
    "\nOverall incremental R^2 "
    "beyond phase-only:",
    incremental_r2
)

print(
    "Overall relative MSE "
    "vs phase-only:",
    sse_model
    / sse_phase_only
)

print(
    "\nMean fold nonzero:",
    resid_fold_tacc3[
        "n_nonzero"
    ].mean()
)

print(
    "All held-out perturbation "
    "residuals predicted:",
    np.all(
        np.isfinite(
            oof_pred_res[
                eval_mask
            ]
        )
    )
)


# ============================================================================
# NOTEBOOK INDEX 96: Cell 93. Full 426-gene cross-fitted
# ============================================================================

# ============================================
# Cell 93. Full 426-gene cross-fitted
# phase-residual evaluation
#
# Diagnostic pass using the CURRENT frozen
# global alpha. No retuning yet.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

residual_oof_records = []

for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

    Z_g = Z_rate[
        valid_mask
    ]

    r_g = R_rate[
        valid_mask,
        gi
    ]

    P_g = P_all[
        valid_mask
    ]

    groups_g = row_conditions[
        valid_mask
    ]

    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_g
        )
        if q != "non-targeting"
    ], dtype=object)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=2026
    )

    oof_y_res = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    oof_pred_res = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    fold_nonzero = []

    for _, val_cond_idx in kf.split(
        pert_conditions
    ):

        val_conditions = pert_conditions[
            val_cond_idx
        ]

        val_fold = np.isin(
            groups_g,
            val_conditions
        )

        train_fold = ~val_fold

        # ------------------------------------
        # Phase models: TRAINING rows only
        # ------------------------------------

        phase_coef_Z, _, _, _ = (
            np.linalg.lstsq(
                P_g[
                    train_fold
                ],
                Z_g[
                    train_fold
                ],
                rcond=None
            )
        )

        phase_coef_r, _, _, _ = (
            np.linalg.lstsq(
                P_g[
                    train_fold
                ],
                r_g[
                    train_fold
                ],
                rcond=None
            )
        )

        Z_train_res = (
            Z_g[
                train_fold
            ]
            -
            P_g[
                train_fold
            ]
            @ phase_coef_Z
        )

        Z_val_res = (
            Z_g[
                val_fold
            ]
            -
            P_g[
                val_fold
            ]
            @ phase_coef_Z
        )

        r_train_res = (
            r_g[
                train_fold
            ]
            -
            P_g[
                train_fold
            ]
            @ phase_coef_r
        )

        r_val_res = (
            r_g[
                val_fold
            ]
            -
            P_g[
                val_fold
            ]
            @ phase_coef_r
        )

        # ------------------------------------
        # Training-only scaling
        # ------------------------------------

        x_scaler = StandardScaler()

        Z_train_z = (
            x_scaler.fit_transform(
                Z_train_res
            )
        )

        Z_val_z = (
            x_scaler.transform(
                Z_val_res
            )
        )

        r_scale = np.std(
            r_train_res,
            ddof=0
        )

        assert (
            np.isfinite(r_scale)
            and r_scale > 0
        )

        r_train_z = (
            r_train_res
            / r_scale
        )

        # ------------------------------------
        # Current frozen alpha
        # diagnostic only
        # ------------------------------------

        model = Lasso(
            alpha=global_alpha,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z_train_z,
            r_train_z
        )

        pred_res = (
            r_scale
            * model.predict(
                Z_val_z
            )
        )

        oof_y_res[
            val_fold
        ] = r_val_res

        oof_pred_res[
            val_fold
        ] = pred_res

        fold_nonzero.append(
            np.count_nonzero(
                model.coef_
            )
        )

    # ----------------------------------------
    # Held-out perturbation rows only
    # ----------------------------------------

    eval_mask = (
        groups_g
        != "non-targeting"
    )

    y_eval = oof_y_res[
        eval_mask
    ]

    p_eval = oof_pred_res[
        eval_mask
    ]

    assert np.all(
        np.isfinite(
            y_eval
        )
    )

    assert np.all(
        np.isfinite(
            p_eval
        )
    )

    sse_model = np.sum(
        (
            y_eval
            - p_eval
        ) ** 2
    )

    sse_phase = np.sum(
        y_eval ** 2
    )

    incremental_r2 = (
        1.0
        - sse_model
        / sse_phase
    )

    residual_oof_records.append({
        "gene": g,
        "n_eval_rows":
            int(
                eval_mask.sum()
            ),
        "n_eval_conditions":
            len(
                pert_conditions
            ),
        "incremental_r2":
            incremental_r2,
        "relative_mse_vs_phase":
            sse_model
            / sse_phase,
        "mean_fold_nonzero":
            np.mean(
                fold_nonzero
            ),
    })

    if (
        (gi + 1) % 25 == 0
        or gi == 0
        or gi + 1
        == len(response_genes)
    ):
        print(
            f"[{gi + 1:03d}/"
            f"{len(response_genes)}] "
            f"{g} done"
        )


residual_oof_summary = pd.DataFrame(
    residual_oof_records
)


print(
    "\nIncremental OOF R^2 "
    "beyond phase-only:"
)

print(
    residual_oof_summary[
        "incremental_r2"
    ].describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nFraction incremental R^2 > 0:"
)

print(
    np.mean(
        residual_oof_summary[
            "incremental_r2"
        ] > 0
    )
)


print(
    "\nMean fold nonzero summary:"
)

print(
    residual_oof_summary[
        "mean_fold_nonzero"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


print(
    "\nWorst 10 genes:"
)

print(
    residual_oof_summary
    .sort_values(
        "incremental_r2"
    )
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nBest 10 genes:"
)

print(
    residual_oof_summary
    .sort_values(
        "incremental_r2",
        ascending=False
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 97: Cell 94. Recalibrate alpha for the
# ============================================================================

# ============================================
# Cell 94. Recalibrate alpha for the
# phase-residualized validation problem
#
# Same fixed 31-gene calibration panel.
# NT always remains in training.
# Phase projection is training-only.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

alpha_grid_resid = np.logspace(
    -4,
    0,
    40
)

resid_alpha_records = []

for pi, g in enumerate(
    calibration_genes
):

    gi = np.where(
        response_genes == g
    )[0][0]

    valid_mask = valid_row_masks[g]

    Z_g = Z_rate[
        valid_mask
    ]

    r_g = R_rate[
        valid_mask,
        gi
    ]

    P_g = P_all[
        valid_mask
    ]

    groups_g = row_conditions[
        valid_mask
    ]

    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_g
        )
        if q != "non-targeting"
    ], dtype=object)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=2026
    )

    # ----------------------------------------
    # Precompute each fold's correctly
    # cross-fitted residualized data
    # ----------------------------------------

    fold_data = []

    for _, val_cond_idx in kf.split(
        pert_conditions
    ):

        val_conditions = pert_conditions[
            val_cond_idx
        ]

        val_fold = np.isin(
            groups_g,
            val_conditions
        )

        train_fold = ~val_fold

        phase_coef_Z, _, _, _ = (
            np.linalg.lstsq(
                P_g[train_fold],
                Z_g[train_fold],
                rcond=None
            )
        )

        phase_coef_r, _, _, _ = (
            np.linalg.lstsq(
                P_g[train_fold],
                r_g[train_fold],
                rcond=None
            )
        )

        Z_train_res = (
            Z_g[train_fold]
            -
            P_g[train_fold]
            @ phase_coef_Z
        )

        Z_val_res = (
            Z_g[val_fold]
            -
            P_g[val_fold]
            @ phase_coef_Z
        )

        r_train_res = (
            r_g[train_fold]
            -
            P_g[train_fold]
            @ phase_coef_r
        )

        r_val_res = (
            r_g[val_fold]
            -
            P_g[val_fold]
            @ phase_coef_r
        )

        x_scaler = StandardScaler()

        Z_train_z = (
            x_scaler.fit_transform(
                Z_train_res
            )
        )

        Z_val_z = (
            x_scaler.transform(
                Z_val_res
            )
        )

        r_scale = np.std(
            r_train_res,
            ddof=0
        )

        assert (
            np.isfinite(r_scale)
            and r_scale > 0
        )

        r_train_z = (
            r_train_res
            / r_scale
        )

        fold_data.append((
            Z_train_z,
            Z_val_z,
            r_train_z,
            r_val_res,
            r_scale
        ))

    # ----------------------------------------
    # Evaluate full alpha grid
    # ----------------------------------------

    for alpha in alpha_grid_resid:

        sse_model = 0.0
        sse_phase = 0.0
        nonzero = []

        for (
            Z_train_z,
            Z_val_z,
            r_train_z,
            r_val_res,
            r_scale
        ) in fold_data:

            model = Lasso(
                alpha=alpha,
                fit_intercept=False,
                max_iter=20000,
                tol=1e-6,
                selection="cyclic",
            )

            model.fit(
                Z_train_z,
                r_train_z
            )

            pred_res = (
                r_scale
                * model.predict(
                    Z_val_z
                )
            )

            sse_model += np.sum(
                (
                    r_val_res
                    - pred_res
                ) ** 2
            )

            sse_phase += np.sum(
                r_val_res ** 2
            )

            nonzero.append(
                np.count_nonzero(
                    model.coef_
                )
            )

        resid_alpha_records.append({
            "gene": g,
            "alpha": alpha,
            "relative_mse":
                sse_model
                / sse_phase,
            "incremental_r2":
                1.0
                - sse_model
                / sse_phase,
            "mean_nonzero":
                np.mean(nonzero),
        })

    print(
        f"[{pi + 1:02d}/"
        f"{len(calibration_genes)}] "
        f"{g} done"
    )


resid_alpha_cv = pd.DataFrame(
    resid_alpha_records
)


# --------------------------------------------
# Equal weight per calibration gene
# --------------------------------------------

resid_alpha_summary = (
    resid_alpha_cv
    .groupby(
        "alpha",
        as_index=False
    )
    .agg(
        mean_relative_mse=(
            "relative_mse",
            "mean"
        ),
        median_relative_mse=(
            "relative_mse",
            "median"
        ),
        mean_incremental_r2=(
            "incremental_r2",
            "mean"
        ),
        mean_nonzero=(
            "mean_nonzero",
            "mean"
        ),
    )
    .sort_values(
        "mean_relative_mse"
    )
)


best_resid_row = (
    resid_alpha_summary
    .iloc[0]
)

global_alpha_resid = float(
    best_resid_row[
        "alpha"
    ]
)


print(
    "\nBest residualized alpha:",
    global_alpha_resid
)

print(
    "\nTop 10 alpha values:"
)

print(
    resid_alpha_summary
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nBest-alpha mean incremental R^2:",
    best_resid_row[
        "mean_incremental_r2"
    ]
)

print(
    "Best-alpha mean nonzero:",
    best_resid_row[
        "mean_nonzero"
    ]
)


# ============================================================================
# NOTEBOOK INDEX 98: Cell 94b. Restore the fixed 31-gene
# ============================================================================

# ============================================
# Cell 94b. Restore the fixed 31-gene
# calibration panel from Cell 82
# ============================================

import numpy as np

calibration_genes = np.asarray([
    "CDKN1B",
    "MLH1",
    "THOC1",
    "BCL7C",
    "EXOC6",
    "MAP3K20",
    "UCHL5",
    "CAMK2G",
    "TXLNG",
    "CSNK2A2",
    "KIF22",
    "DSN1",
    "FANCI",
    "DNA2",
    "NCAPH",
    "PPP6C",
    "NCAPD2",
    "RMI1",
    "KMT2E",
    "VPS4B",
    "TACC3",
    "CACUL1",
    "TFDP2",
    "CDKN3",
    "CSNK1A1",
    "ERCC6",
    "UBE2B",
    "EXT1",
    "HSP90AB1",
    "PLK2",
    "RPS3",
], dtype=object)


print(
    "Calibration genes:",
    len(calibration_genes)
)

print(
    calibration_genes
)


# Verify all are still in the 426 response set
missing = [
    g
    for g in calibration_genes
    if g not in set(response_genes)
]

print(
    "\nMissing from response_genes:",
    missing
)

assert len(calibration_genes) == 31
assert len(missing) == 0

print(
    "\nFixed calibration panel restored."
)


# ============================================================================
# NOTEBOOK INDEX 99: Cell 94c. Calibrate alpha for the
# ============================================================================

# ============================================
# Cell 94c. Calibrate alpha for the
# phase-residualized validation problem
#
# Fixed 31-gene panel from Cell 82.
# NT always remains in training.
# Phase projection is training-only.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


alpha_grid_resid = np.logspace(
    -4,
    0,
    40
)

resid_alpha_records = []


for pi, g in enumerate(
    calibration_genes
):

    gi = np.where(
        response_genes == g
    )[0][0]

    valid_mask = valid_row_masks[g]

    Z_g = Z_rate[
        valid_mask
    ]

    r_g = R_rate[
        valid_mask,
        gi
    ]

    P_g = P_all[
        valid_mask
    ]

    groups_g = row_conditions[
        valid_mask
    ]

    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_g
        )
        if q != "non-targeting"
    ], dtype=object)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=2026
    )

    # ----------------------------------------
    # Construct folds once for this gene.
    # All alpha values see identical folds.
    # ----------------------------------------

    fold_data = []

    for _, val_cond_idx in kf.split(
        pert_conditions
    ):

        val_conditions = pert_conditions[
            val_cond_idx
        ]

        val_fold = np.isin(
            groups_g,
            val_conditions
        )

        train_fold = ~val_fold

        assert np.any(
            groups_g[
                train_fold
            ] == "non-targeting"
        )

        assert not np.any(
            groups_g[
                val_fold
            ] == "non-targeting"
        )

        # ------------------------------------
        # Training-only phase models
        # ------------------------------------

        phase_coef_Z, _, _, _ = (
            np.linalg.lstsq(
                P_g[
                    train_fold
                ],
                Z_g[
                    train_fold
                ],
                rcond=None
            )
        )

        phase_coef_r, _, _, _ = (
            np.linalg.lstsq(
                P_g[
                    train_fold
                ],
                r_g[
                    train_fold
                ],
                rcond=None
            )
        )

        # ------------------------------------
        # Apply training phase model to
        # train and validation
        # ------------------------------------

        Z_train_res = (
            Z_g[
                train_fold
            ]
            -
            P_g[
                train_fold
            ]
            @ phase_coef_Z
        )

        Z_val_res = (
            Z_g[
                val_fold
            ]
            -
            P_g[
                val_fold
            ]
            @ phase_coef_Z
        )

        r_train_res = (
            r_g[
                train_fold
            ]
            -
            P_g[
                train_fold
            ]
            @ phase_coef_r
        )

        r_val_res = (
            r_g[
                val_fold
            ]
            -
            P_g[
                val_fold
            ]
            @ phase_coef_r
        )

        # ------------------------------------
        # Training-only predictor scaling
        # ------------------------------------

        x_scaler = StandardScaler()

        Z_train_z = (
            x_scaler.fit_transform(
                Z_train_res
            )
        )

        Z_val_z = (
            x_scaler.transform(
                Z_val_res
            )
        )

        # ------------------------------------
        # Training-only response scale
        # ------------------------------------

        r_scale = np.std(
            r_train_res,
            ddof=0
        )

        assert (
            np.isfinite(r_scale)
            and r_scale > 0
        )

        r_train_z = (
            r_train_res
            / r_scale
        )

        fold_data.append({
            "Z_train_z":
                Z_train_z,
            "Z_val_z":
                Z_val_z,
            "r_train_z":
                r_train_z,
            "r_val_res":
                r_val_res,
            "r_scale":
                r_scale,
        })


    # ----------------------------------------
    # Evaluate alpha grid
    # ----------------------------------------

    for alpha in alpha_grid_resid:

        sse_model = 0.0
        sse_phase = 0.0

        fold_nonzero = []

        for fd in fold_data:

            model = Lasso(
                alpha=float(alpha),
                fit_intercept=False,
                max_iter=20000,
                tol=1e-6,
                selection="cyclic",
            )

            model.fit(
                fd["Z_train_z"],
                fd["r_train_z"]
            )

            pred_res = (
                fd["r_scale"]
                * model.predict(
                    fd["Z_val_z"]
                )
            )

            sse_model += np.sum(
                (
                    fd["r_val_res"]
                    - pred_res
                ) ** 2
            )

            sse_phase += np.sum(
                fd["r_val_res"] ** 2
            )

            fold_nonzero.append(
                np.count_nonzero(
                    model.coef_
                )
            )

        relative_mse = (
            sse_model
            / sse_phase
        )

        resid_alpha_records.append({
            "gene":
                g,
            "alpha":
                float(alpha),
            "relative_mse":
                relative_mse,
            "incremental_r2":
                1.0
                - relative_mse,
            "mean_nonzero":
                np.mean(
                    fold_nonzero
                ),
        })


    print(
        f"[{pi + 1:02d}/"
        f"{len(calibration_genes)}] "
        f"{g} done"
    )


resid_alpha_cv = pd.DataFrame(
    resid_alpha_records
)


# --------------------------------------------
# Equal weighting across the 31 genes
# --------------------------------------------

resid_alpha_summary = (
    resid_alpha_cv
    .groupby(
        "alpha",
        as_index=False
    )
    .agg(
        mean_relative_mse=(
            "relative_mse",
            "mean"
        ),
        median_relative_mse=(
            "relative_mse",
            "median"
        ),
        mean_incremental_r2=(
            "incremental_r2",
            "mean"
        ),
        mean_nonzero=(
            "mean_nonzero",
            "mean"
        ),
    )
    .sort_values(
        "mean_relative_mse"
    )
    .reset_index(
        drop=True
    )
)


best_resid_row = (
    resid_alpha_summary.iloc[0]
)

global_alpha_resid = float(
    best_resid_row[
        "alpha"
    ]
)


print(
    "\nBest residualized alpha:",
    global_alpha_resid
)

print(
    "\nTop 10 alpha values:"
)

print(
    resid_alpha_summary
    .head(10)
    .to_string(
        index=False
    )
)

print(
    "\nBest-alpha mean incremental R^2:",
    float(
        best_resid_row[
            "mean_incremental_r2"
        ]
    )
)

print(
    "Best-alpha median relative MSE:",
    float(
        best_resid_row[
            "median_relative_mse"
        ]
    )
)

print(
    "Best-alpha mean nonzero:",
    float(
        best_resid_row[
            "mean_nonzero"
        ]
    )
)

print(
    "\nOld diagnostic alpha:",
    global_alpha
)

print(
    "New / old alpha ratio:",
    global_alpha_resid
    / global_alpha
)


# ============================================================================
# NOTEBOOK INDEX 100: Cell 95. Final phase-residualized OOF
# ============================================================================

# ============================================
# Cell 95. Final phase-residualized OOF
# validation with frozen calibrated alpha
#
# Primary validation summary:
# 395 genes NOT used for alpha calibration.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


final_resid_oof_records = []


for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

    Z_g = Z_rate[
        valid_mask
    ]

    r_g = R_rate[
        valid_mask,
        gi
    ]

    P_g = P_all[
        valid_mask
    ]

    groups_g = row_conditions[
        valid_mask
    ]

    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_g
        )
        if q != "non-targeting"
    ], dtype=object)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=2026
    )

    oof_y_res = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    oof_pred_res = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    fold_nonzero = []


    for _, val_cond_idx in kf.split(
        pert_conditions
    ):

        val_conditions = pert_conditions[
            val_cond_idx
        ]

        val_fold = np.isin(
            groups_g,
            val_conditions
        )

        train_fold = ~val_fold

        # ------------------------------------
        # Training-only phase projections
        # ------------------------------------

        phase_coef_Z, _, _, _ = (
            np.linalg.lstsq(
                P_g[
                    train_fold
                ],
                Z_g[
                    train_fold
                ],
                rcond=None
            )
        )

        phase_coef_r, _, _, _ = (
            np.linalg.lstsq(
                P_g[
                    train_fold
                ],
                r_g[
                    train_fold
                ],
                rcond=None
            )
        )

        Z_train_res = (
            Z_g[
                train_fold
            ]
            -
            P_g[
                train_fold
            ]
            @ phase_coef_Z
        )

        Z_val_res = (
            Z_g[
                val_fold
            ]
            -
            P_g[
                val_fold
            ]
            @ phase_coef_Z
        )

        r_train_res = (
            r_g[
                train_fold
            ]
            -
            P_g[
                train_fold
            ]
            @ phase_coef_r
        )

        r_val_res = (
            r_g[
                val_fold
            ]
            -
            P_g[
                val_fold
            ]
            @ phase_coef_r
        )

        # ------------------------------------
        # Training-only scaling
        # ------------------------------------

        x_scaler = StandardScaler()

        Z_train_z = (
            x_scaler.fit_transform(
                Z_train_res
            )
        )

        Z_val_z = (
            x_scaler.transform(
                Z_val_res
            )
        )

        r_scale = np.std(
            r_train_res,
            ddof=0
        )

        assert (
            np.isfinite(r_scale)
            and r_scale > 0
        )

        r_train_z = (
            r_train_res
            / r_scale
        )

        # ------------------------------------
        # Frozen calibrated residual alpha
        # ------------------------------------

        model = Lasso(
            alpha=global_alpha_resid,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z_train_z,
            r_train_z
        )

        pred_res = (
            r_scale
            * model.predict(
                Z_val_z
            )
        )

        oof_y_res[
            val_fold
        ] = r_val_res

        oof_pred_res[
            val_fold
        ] = pred_res

        fold_nonzero.append(
            np.count_nonzero(
                model.coef_
            )
        )


    # ----------------------------------------
    # Held-out perturbation rows only
    # ----------------------------------------

    eval_mask = (
        groups_g
        != "non-targeting"
    )

    y_eval = oof_y_res[
        eval_mask
    ]

    p_eval = oof_pred_res[
        eval_mask
    ]

    assert np.all(
        np.isfinite(
            y_eval
        )
    )

    assert np.all(
        np.isfinite(
            p_eval
        )
    )

    sse_model = np.sum(
        (
            y_eval
            - p_eval
        ) ** 2
    )

    sse_phase = np.sum(
        y_eval ** 2
    )

    relative_mse = (
        sse_model
        / sse_phase
    )

    incremental_r2 = (
        1.0
        - relative_mse
    )

    final_resid_oof_records.append({
        "gene":
            g,
        "used_for_alpha_calibration":
            g in set(
                calibration_genes
            ),
        "n_eval_rows":
            int(
                eval_mask.sum()
            ),
        "n_eval_conditions":
            len(
                pert_conditions
            ),
        "incremental_r2":
            incremental_r2,
        "relative_mse_vs_phase":
            relative_mse,
        "mean_fold_nonzero":
            np.mean(
                fold_nonzero
            ),
    })


    if (
        (gi + 1) % 25 == 0
        or gi == 0
        or gi + 1
        == len(response_genes)
    ):
        print(
            f"[{gi + 1:03d}/"
            f"{len(response_genes)}] "
            f"{g} done"
        )


final_resid_oof_summary = pd.DataFrame(
    final_resid_oof_records
)


# --------------------------------------------
# Primary: non-calibration genes
# --------------------------------------------

noncal_mask = (
    ~final_resid_oof_summary[
        "used_for_alpha_calibration"
    ]
)

noncal = final_resid_oof_summary[
    noncal_mask
]

cal = final_resid_oof_summary[
    ~noncal_mask
]


print(
    "\nPRIMARY: non-calibration genes:"
)

print(
    "n =",
    len(noncal)
)

print(
    noncal[
        "incremental_r2"
    ].describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)

print(
    "\nFraction incremental R^2 > 0:"
)

print(
    np.mean(
        noncal[
            "incremental_r2"
        ] > 0
    )
)

print(
    "\nRelative MSE vs phase-only:"
)

print(
    noncal[
        "relative_mse_vs_phase"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)

print(
    "\nMean fold nonzero:"
)

print(
    noncal[
        "mean_fold_nonzero"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


# --------------------------------------------
# Calibration panel shown separately
# --------------------------------------------

print(
    "\nCALIBRATION PANEL:"
)

print(
    "n =",
    len(cal)
)

print(
    cal[
        "incremental_r2"
    ].describe()
)


# --------------------------------------------
# Full set, secondary summary
# --------------------------------------------

print(
    "\nALL 426 genes:"
)

print(
    final_resid_oof_summary[
        "incremental_r2"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)

print(
    "\nAll-gene fraction > 0:"
)

print(
    np.mean(
        final_resid_oof_summary[
            "incremental_r2"
        ] > 0
    )
)


print(
    "\nBest 10 non-calibration genes:"
)

print(
    noncal
    .sort_values(
        "incremental_r2",
        ascending=False
    )
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nWorst 10 non-calibration genes:"
)

print(
    noncal
    .sort_values(
        "incremental_r2"
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 101: Cell 96. Exact integral-objective prototype
# ============================================================================

# ============================================
# Cell 96. Exact integral-objective prototype
# for TACC3 using FWL residualization
#
# This matches:
#   y = c_g * dt + X_reg A_g
#
# without dividing by dt.
#
# Diagnostic only: no final alpha calibration yet.
# ============================================

import numpy as np
from sklearn.linear_model import Lasso


test_gene = "TACC3"

gi = np.where(
    response_genes == test_gene
)[0][0]

valid_mask = valid_row_masks[
    test_gene
]


# --------------------------------------------
# Original integral system
# --------------------------------------------

d = np.asarray(
    X_final[
        valid_mask, 0
    ],
    dtype=np.float64
)

Xr = np.asarray(
    X_final[
        valid_mask, 1:
    ],
    dtype=np.float64
)

y = np.asarray(
    Y_final[
        valid_mask, gi
    ],
    dtype=np.float64
)


print(
    "Rows:",
    len(y)
)

print(
    "dt min / median / max:",
    np.min(d),
    np.median(d),
    np.max(d)
)

print(
    "max/min dt ratio:",
    np.max(d) / np.min(d)
)


# --------------------------------------------
# FWL projection:
# remove the unpenalized column d = dt
#
# Xp and yp are exactly orthogonal to d.
# --------------------------------------------

dd = np.dot(
    d, d
)

coef_d_X = (
    d @ Xr
) / dd

coef_d_y = (
    d @ y
) / dd


Xp = (
    Xr
    - d[:, None]
    * coef_d_X[None, :]
)

yp = (
    y
    - d * coef_d_y
)


print(
    "\nFWL orthogonality:"
)

print(
    "max |d^T Xp|:",
    np.max(
        np.abs(
            d @ Xp
        )
    )
)

print(
    "|d^T yp|:",
    abs(
        d @ yp
    )
)


# --------------------------------------------
# Scale projected predictors WITHOUT centering.
#
# Centering would introduce a constant direction,
# whereas the original unpenalized direction is dt.
# --------------------------------------------

x_scale = np.sqrt(
    np.mean(
        Xp ** 2,
        axis=0
    )
)

zero_scale = (
    ~np.isfinite(x_scale)
    |
    (x_scale <= 0)
)

print(
    "\nZero/invalid predictor scales:",
    zero_scale.sum()
)

assert zero_scale.sum() == 0


# Dimensionless response scale
y_scale = np.sqrt(
    np.mean(
        yp ** 2
    )
)

assert (
    np.isfinite(y_scale)
    and y_scale > 0
)


Z = (
    Xp
    / x_scale[None, :]
)

yz = (
    yp
    / y_scale
)


# --------------------------------------------
# Use current residualized alpha ONLY as a
# numerical prototype.
#
# This alpha is NOT yet declared final for
# the exact integral estimator.
# --------------------------------------------

model = Lasso(
    alpha=global_alpha_resid,
    fit_intercept=False,
    max_iter=20000,
    tol=1e-6,
    selection="cyclic",
)

model.fit(
    Z,
    yz
)


# --------------------------------------------
# Back-transform A to original integral units
# --------------------------------------------

A_tacc3_integral = (
    y_scale
    * model.coef_
    / x_scale
)


# Exact unpenalized c conditional on A
c_tacc3_integral = (
    d
    @ (
        y
        - Xr
        @ A_tacc3_integral
    )
) / dd


# --------------------------------------------
# Verify equivalence of FWL and original
# residuals at the fitted coefficients
# --------------------------------------------

residual_original = (
    y
    - d * c_tacc3_integral
    - Xr @ A_tacc3_integral
)

residual_projected = (
    yp
    - Xp @ A_tacc3_integral
)


print(
    "\nBack-transform checks:"
)

print(
    "c_tacc3_integral:",
    c_tacc3_integral
)

print(
    "nonzero A:",
    np.count_nonzero(
        A_tacc3_integral
    )
)

print(
    "max residual difference:",
    np.max(
        np.abs(
            residual_original
            - residual_projected
        )
    )
)

print(
    "original integral MSE:",
    np.mean(
        residual_original ** 2
    )
)

print(
    "projected integral MSE:",
    np.mean(
        residual_projected ** 2
    )
)


# --------------------------------------------
# Explicit first-order condition for
# unpenalized c:
#
# d^T residual must be zero.
# --------------------------------------------

print(
    "\n|d^T final residual|:",
    abs(
        d @ residual_original
    )
)


# --------------------------------------------
# Compare with old rate-form TACC3 fit
# only as a diagnostic
# --------------------------------------------

old_tacc3_row = np.where(
    response_genes == "TACC3"
)[0][0]

print(
    "\nOld rate-form nonzero:",
    np.count_nonzero(
        A_hat[
            old_tacc3_row
        ]
    )
)

print(
    "Exact-integral prototype nonzero:",
    np.count_nonzero(
        A_tacc3_integral
    )
)


# ============================================================================
# NOTEBOOK INDEX 102: Cell 97. Alpha calibration for the EXACT
# ============================================================================

# ============================================
# Cell 97. Alpha calibration for the EXACT
# integral estimator (Algorithm 5.1)
#
# Model:
#   Y = c * dt + X_reg A
#
# c is unpenalized via fold-specific FWL.
# L1 penalty applies only to A.
#
# Held-out perturbation conditions;
# NT always remains in training.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso


alpha_grid_integral = np.logspace(
    -4,
    0,
    40
)

integral_alpha_records = []


for pi, g in enumerate(
    calibration_genes
):

    gi = np.where(
        response_genes == g
    )[0][0]

    valid_mask = valid_row_masks[g]

    d_g = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_g = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y_g = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )

    groups_g = row_conditions[
        valid_mask
    ]

    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_g
        )
        if q != "non-targeting"
    ], dtype=object)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=2026
    )

    fold_data = []


    # ----------------------------------------
    # Prepare folds once per gene
    # ----------------------------------------

    for _, val_cond_idx in kf.split(
        pert_conditions
    ):

        val_conditions = pert_conditions[
            val_cond_idx
        ]

        val_fold = np.isin(
            groups_g,
            val_conditions
        )

        train_fold = ~val_fold

        assert np.any(
            groups_g[
                train_fold
            ] == "non-targeting"
        )

        assert not np.any(
            groups_g[
                val_fold
            ] == "non-targeting"
        )


        d_train = d_g[
            train_fold
        ]

        X_train = Xr_g[
            train_fold
        ]

        y_train = y_g[
            train_fold
        ]

        d_val = d_g[
            val_fold
        ]

        X_val = Xr_g[
            val_fold
        ]

        y_val = y_g[
            val_fold
        ]


        # ------------------------------------
        # Training-only FWL projection
        # ------------------------------------

        dd = np.dot(
            d_train,
            d_train
        )

        coef_d_X = (
            d_train @ X_train
        ) / dd

        coef_d_y = (
            d_train @ y_train
        ) / dd


        Xp_train = (
            X_train
            - d_train[:, None]
            * coef_d_X[None, :]
        )

        yp_train = (
            y_train
            - d_train
            * coef_d_y
        )


        # ------------------------------------
        # Scale projected predictors
        # WITHOUT centering
        # ------------------------------------

        x_scale = np.sqrt(
            np.mean(
                Xp_train ** 2,
                axis=0
            )
        )

        assert np.all(
            np.isfinite(
                x_scale
            )
        )

        assert np.all(
            x_scale > 0
        )


        y_scale = np.sqrt(
            np.mean(
                yp_train ** 2
            )
        )

        assert (
            np.isfinite(y_scale)
            and y_scale > 0
        )


        Z_train = (
            Xp_train
            / x_scale[None, :]
        )

        yz_train = (
            yp_train
            / y_scale
        )


        fold_data.append({
            "d_train":
                d_train,
            "X_train":
                X_train,
            "y_train":
                y_train,

            "d_val":
                d_val,
            "X_val":
                X_val,
            "y_val":
                y_val,

            "dd":
                dd,
            "x_scale":
                x_scale,
            "y_scale":
                y_scale,
            "Z_train":
                Z_train,
            "yz_train":
                yz_train,
        })


    # ----------------------------------------
    # Evaluate alpha grid
    # ----------------------------------------

    for alpha in alpha_grid_integral:

        sse_model = 0.0
        sse_baseline = 0.0

        fold_nonzero = []


        for fd in fold_data:

            model = Lasso(
                alpha=float(alpha),
                fit_intercept=False,
                max_iter=20000,
                tol=1e-6,
                selection="cyclic",
            )

            model.fit(
                fd["Z_train"],
                fd["yz_train"]
            )


            # --------------------------------
            # Back-transform A
            # --------------------------------

            A_fold = (
                fd["y_scale"]
                * model.coef_
                / fd["x_scale"]
            )


            # --------------------------------
            # Exact unpenalized c on training
            # --------------------------------

            c_fold = (
                fd["d_train"]
                @ (
                    fd["y_train"]
                    - fd["X_train"]
                    @ A_fold
                )
            ) / fd["dd"]


            # --------------------------------
            # Held-out integral prediction
            # --------------------------------

            pred_val = (
                fd["d_val"]
                * c_fold
                + fd["X_val"]
                @ A_fold
            )


            # --------------------------------
            # Baseline:
            # intercept-only integral model
            # fitted on training rows
            # --------------------------------

            c0_fold = (
                fd["d_train"]
                @ fd["y_train"]
            ) / fd["dd"]

            pred0_val = (
                fd["d_val"]
                * c0_fold
            )


            sse_model += np.sum(
                (
                    fd["y_val"]
                    - pred_val
                ) ** 2
            )

            sse_baseline += np.sum(
                (
                    fd["y_val"]
                    - pred0_val
                ) ** 2
            )

            fold_nonzero.append(
                np.count_nonzero(
                    A_fold
                )
            )


        relative_mse = (
            sse_model
            / sse_baseline
        )

        integral_alpha_records.append({
            "gene":
                g,
            "alpha":
                float(alpha),
            "relative_mse":
                relative_mse,
            "skill_vs_intercept":
                1.0
                - relative_mse,
            "mean_nonzero":
                np.mean(
                    fold_nonzero
                ),
        })


    print(
        f"[{pi + 1:02d}/"
        f"{len(calibration_genes)}] "
        f"{g} done"
    )


integral_alpha_cv = pd.DataFrame(
    integral_alpha_records
)


# --------------------------------------------
# Equal weighting across calibration genes
# --------------------------------------------

integral_alpha_summary = (
    integral_alpha_cv
    .groupby(
        "alpha",
        as_index=False
    )
    .agg(
        mean_relative_mse=(
            "relative_mse",
            "mean"
        ),
        median_relative_mse=(
            "relative_mse",
            "median"
        ),
        mean_skill=(
            "skill_vs_intercept",
            "mean"
        ),
        mean_nonzero=(
            "mean_nonzero",
            "mean"
        ),
    )
    .sort_values(
        "mean_relative_mse"
    )
    .reset_index(
        drop=True
    )
)


best_integral_row = (
    integral_alpha_summary.iloc[0]
)

global_alpha_integral = float(
    best_integral_row[
        "alpha"
    ]
)


print(
    "\nBest exact-integral alpha:",
    global_alpha_integral
)

print(
    "\nTop 10 alpha values:"
)

print(
    integral_alpha_summary
    .head(10)
    .to_string(
        index=False
    )
)

print(
    "\nBest-alpha mean skill:",
    float(
        best_integral_row[
            "mean_skill"
        ]
    )
)

print(
    "Best-alpha median relative MSE:",
    float(
        best_integral_row[
            "median_relative_mse"
        ]
    )
)

print(
    "Best-alpha mean nonzero:",
    float(
        best_integral_row[
            "mean_nonzero"
        ]
    )
)

print(
    "\nRate-form alpha:",
    global_alpha
)

print(
    "Residual-validation alpha:",
    global_alpha_resid
)


# ============================================================================
# NOTEBOOK INDEX 103: Cell 98. FINAL exact-integral GRN fit
# ============================================================================

# ============================================
# Cell 98. FINAL exact-integral GRN fit
#
# Algorithm 5.1 objective:
#
#   Y_g = c_g * dt + X_reg A_g
#
# c_g unpenalized via exact FWL projection.
# L1 penalty only on A_g.
#
# Direct q=g rows remain excluded through
# valid_row_masks.
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


n_response = len(
    response_genes
)

n_regulator = len(
    regulator_genes_vc
)


A_hat_integral = np.zeros(
    (
        n_response,
        n_regulator
    ),
    dtype=np.float64
)

c_hat_integral = np.zeros(
    n_response,
    dtype=np.float64
)


integral_fit_records = []


for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]


    # ----------------------------------------
    # Original integral system
    # ----------------------------------------

    d = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )


    # ----------------------------------------
    # Exact FWL projection of unpenalized dt
    # ----------------------------------------

    dd = np.dot(
        d,
        d
    )

    coef_d_X = (
        d @ Xr
    ) / dd

    coef_d_y = (
        d @ y
    ) / dd


    Xp = (
        Xr
        - d[:, None]
        * coef_d_X[None, :]
    )

    yp = (
        y
        - d * coef_d_y
    )


    # ----------------------------------------
    # RMS scaling only — NO centering
    # ----------------------------------------

    x_scale = np.sqrt(
        np.mean(
            Xp ** 2,
            axis=0
        )
    )

    assert np.all(
        np.isfinite(
            x_scale
        )
    )

    assert np.all(
        x_scale > 0
    )


    y_scale = np.sqrt(
        np.mean(
            yp ** 2
        )
    )

    assert (
        np.isfinite(y_scale)
        and y_scale > 0
    )


    Z = (
        Xp
        / x_scale[None, :]
    )

    yz = (
        yp
        / y_scale
    )


    # ----------------------------------------
    # Frozen exact-integral alpha
    # ----------------------------------------

    model = Lasso(
        alpha=global_alpha_integral,
        fit_intercept=False,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z,
        yz
    )


    # ----------------------------------------
    # Back-transform regulatory coefficients
    # ----------------------------------------

    A_g = (
        y_scale
        * model.coef_
        / x_scale
    )


    # ----------------------------------------
    # Exact unpenalized basal coefficient
    # conditional on A_g
    # ----------------------------------------

    c_g = (
        d
        @ (
            y
            - Xr @ A_g
        )
    ) / dd


    A_hat_integral[
        gi, :
    ] = A_g

    c_hat_integral[
        gi
    ] = c_g


    # ----------------------------------------
    # Diagnostics on original integral scale
    # ----------------------------------------

    pred = (
        d * c_g
        + Xr @ A_g
    )

    residual = (
        y
        - pred
    )


    # Intercept-only baseline
    c0 = (
        d @ y
    ) / dd

    pred0 = (
        d * c0
    )


    sse = np.sum(
        residual ** 2
    )

    sse0 = np.sum(
        (
            y
            - pred0
        ) ** 2
    )


    integral_fit_records.append({
        "gene":
            g,

        "n_rows":
            len(y),

        "n_nonzero":
            int(
                np.count_nonzero(
                    A_g
                )
            ),

        "integral_mse":
            np.mean(
                residual ** 2
            ),

        "skill_vs_intercept":
            1.0
            - sse / sse0,

        "fwl_intercept_score":
            abs(
                d @ residual
            ),
    })


    if (
        (gi + 1) % 25 == 0
        or gi == 0
        or gi + 1
        == n_response
    ):
        print(
            f"[{gi + 1:03d}/"
            f"{n_response}] "
            f"{g} done"
        )


integral_fit_summary = pd.DataFrame(
    integral_fit_records
)


# ============================================
# Global validation checks
# ============================================

print(
    "\nA_hat_integral shape:",
    A_hat_integral.shape
)

print(
    "c_hat_integral shape:",
    c_hat_integral.shape
)

print(
    "NaN/inf in A:",
    (
        ~np.isfinite(
            A_hat_integral
        )
    ).any()
)

print(
    "NaN/inf in c:",
    (
        ~np.isfinite(
            c_hat_integral
        )
    ).any()
)


print(
    "\nNonzero edges / response gene:"
)

print(
    integral_fit_summary[
        "n_nonzero"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


total_nonzero = np.count_nonzero(
    A_hat_integral
)

total_possible = (
    A_hat_integral.size
)

print(
    "\nTotal nonzero edges:",
    total_nonzero,
    "/",
    total_possible
)

print(
    "Network density:",
    total_nonzero
    / total_possible
)


print(
    "\nTraining skill vs intercept-only:"
)

print(
    integral_fit_summary[
        "skill_vs_intercept"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


print(
    "\nFWL first-order-condition check:"
)

print(
    "max |dt^T residual|:",
    integral_fit_summary[
        "fwl_intercept_score"
    ].max()
)


# --------------------------------------------
# Compare exact-integral and old rate network
# --------------------------------------------

old_nz = np.count_nonzero(
    A_hat,
    axis=1
)

new_nz = np.count_nonzero(
    A_hat_integral,
    axis=1
)

print(
    "\nOld rate-form mean nonzero:",
    old_nz.mean()
)

print(
    "Exact-integral mean nonzero:",
    new_nz.mean()
)

print(
    "Median change in edges/gene:",
    np.median(
        new_nz
        - old_nz
    )
)


# ============================================================================
# NOTEBOOK INDEX 104: Cell 99. Save FINAL exact-integral GRN
# ============================================================================

# ============================================
# Cell 99. Save FINAL exact-integral GRN
# + phase-residual validation checkpoint
# ============================================

import numpy as np
import os

final_checkpoint_path = (
    "/home/featurize/work/project1/"
    "replogle_final_exact_integral_grn.npz"
)

np.savez_compressed(
    final_checkpoint_path,

    # ----------------------------------------
    # Final Algorithm 5.1 reconstruction
    # ----------------------------------------

    A_hat_integral=A_hat_integral,
    c_hat_integral=c_hat_integral,

    response_genes=np.asarray(
        response_genes,
        dtype=str
    ),

    regulator_genes=np.asarray(
        regulator_genes_vc,
        dtype=str
    ),

    global_alpha_integral=np.asarray(
        global_alpha_integral
    ),

    # ----------------------------------------
    # Exact integral design / response
    # ----------------------------------------

    X_final=X_final,
    Y_final=Y_final,

    # ----------------------------------------
    # Final fit diagnostics
    # ----------------------------------------

    integral_n_nonzero=np.asarray(
        integral_fit_summary[
            "n_nonzero"
        ],
        dtype=int
    ),

    integral_skill=np.asarray(
        integral_fit_summary[
            "skill_vs_intercept"
        ],
        dtype=float
    ),

    integral_mse=np.asarray(
        integral_fit_summary[
            "integral_mse"
        ],
        dtype=float
    ),

    # ----------------------------------------
    # Strict phase-residualized validation
    # ----------------------------------------

    global_alpha_resid=np.asarray(
        global_alpha_resid
    ),

    validation_genes=np.asarray(
        final_resid_oof_summary[
            "gene"
        ],
        dtype=str
    ),

    validation_incremental_r2=np.asarray(
        final_resid_oof_summary[
            "incremental_r2"
        ],
        dtype=float
    ),

    validation_relative_mse=np.asarray(
        final_resid_oof_summary[
            "relative_mse_vs_phase"
        ],
        dtype=float
    ),

    validation_mean_nonzero=np.asarray(
        final_resid_oof_summary[
            "mean_fold_nonzero"
        ],
        dtype=float
    ),

    validation_calibration_flag=np.asarray(
        final_resid_oof_summary[
            "used_for_alpha_calibration"
        ],
        dtype=bool
    ),
)


print(
    "Saved:",
    final_checkpoint_path
)

print(
    "Exists:",
    os.path.exists(
        final_checkpoint_path
    )
)

print(
    "Size MB:",
    os.path.getsize(
        final_checkpoint_path
    ) / 1024**2
)


# --------------------------------------------
# Immediate reload verification
# --------------------------------------------

check_final = np.load(
    final_checkpoint_path,
    allow_pickle=False
)

print(
    "\nSaved keys:"
)

print(
    check_final.files
)

print(
    "\nA exact:",
    np.array_equal(
        check_final[
            "A_hat_integral"
        ],
        A_hat_integral
    )
)

print(
    "c exact:",
    np.array_equal(
        check_final[
            "c_hat_integral"
        ],
        c_hat_integral
    )
)

print(
    "alpha exact:",
    float(
        check_final[
            "global_alpha_integral"
        ]
    )
    == global_alpha_integral
)

print(
    "validation exact:",
    np.array_equal(
        check_final[
            "validation_incremental_r2"
        ],
        final_resid_oof_summary[
            "incremental_r2"
        ].to_numpy()
    )
)


# ============================================================================
# NOTEBOOK INDEX 105: Cell 100. Correct identifiability design
# ============================================================================

# ============================================
# Cell 100. Correct identifiability design
#
# Full exact-integral design:
#   [dt, integral regulator features]
#
# IMPORTANT:
# - include the dt/intercept column
# - NO centering
# - only invertible column scaling
# ============================================

import numpy as np


X_ident_raw = np.asarray(
    X_final,
    dtype=np.float64
)

n_rows_ident, n_cols_ident = (
    X_ident_raw.shape
)


print(
    "Raw identifiability design shape:",
    X_ident_raw.shape
)

print(
    "Expected columns:",
    1 + len(regulator_genes_vc)
)


# --------------------------------------------
# Raw numerical rank
# --------------------------------------------

rank_raw = np.linalg.matrix_rank(
    X_ident_raw
)

print(
    "\nRaw numerical rank:",
    rank_raw,
    "/",
    n_cols_ident
)


# --------------------------------------------
# Column RMS scaling ONLY
#
# This is an invertible diagonal transform,
# so it preserves structural rank.
# --------------------------------------------

ident_col_scale = np.sqrt(
    np.mean(
        X_ident_raw ** 2,
        axis=0
    )
)

print(
    "\nInvalid/zero column scales:",
    np.sum(
        (~np.isfinite(ident_col_scale))
        |
        (ident_col_scale <= 0)
    )
)

assert np.all(
    np.isfinite(
        ident_col_scale
    )
)

assert np.all(
    ident_col_scale > 0
)


X_ident_scaled = (
    X_ident_raw
    / ident_col_scale[None, :]
)


rank_scaled = np.linalg.matrix_rank(
    X_ident_scaled
)

print(
    "Scaled numerical rank:",
    rank_scaled,
    "/",
    n_cols_ident
)

print(
    "Rank preserved:",
    rank_scaled == rank_raw
)


# --------------------------------------------
# Singular values
# --------------------------------------------

s_ident = np.linalg.svd(
    X_ident_scaled,
    compute_uv=False
)

print(
    "\nSingular values:"
)

print(
    "largest:",
    s_ident[0]
)

print(
    "median:",
    np.median(
        s_ident
    )
)

print(
    "smallest:",
    s_ident[-1]
)

print(
    "singular condition number:",
    s_ident[0]
    / s_ident[-1]
)


print(
    "\nSmallest 10 singular values:"
)

print(
    s_ident[-10:]
)


# --------------------------------------------
# Scaled information matrix
#
# I = X^T X / n
#
# Scaling changes units but not PD/rank.
# --------------------------------------------

I_ident = (
    X_ident_scaled.T
    @ X_ident_scaled
) / n_rows_ident

eig_ident = np.linalg.eigvalsh(
    I_ident
)

lambda_min_ident = eig_ident[0]
lambda_max_ident = eig_ident[-1]

print(
    "\nScaled information matrix:"
)

print(
    "lambda_min:",
    lambda_min_ident
)

print(
    "lambda_max:",
    lambda_max_ident
)

print(
    "condition number:",
    lambda_max_ident
    / lambda_min_ident
)


# --------------------------------------------
# Verify first column really is dt
# --------------------------------------------

dt_from_rows = np.asarray([
    row["delta_t"]
    for row in final_interval_rows
], dtype=np.float64)

print(
    "\nFirst column equals interval dt:",
    np.allclose(
        X_ident_raw[:, 0],
        dt_from_rows,
        rtol=0,
        atol=1e-14
    )
)

print(
    "dt min / median / max:",
    np.min(
        X_ident_raw[:, 0]
    ),
    np.median(
        X_ident_raw[:, 0]
    ),
    np.max(
        X_ident_raw[:, 0]
    )
)


# ============================================================================
# NOTEBOOK INDEX 106: Cell 101. Correct randomized perturbation
# ============================================================================

# ============================================
# Cell 101. Correct randomized perturbation
# identifiability path
#
# - exact integral design
# - include dt column
# - NO centering
# - fixed full-design RMS scaling
# - NT always included
# ============================================

import numpy as np
import pandas as pd


rng = np.random.default_rng(
    2026
)

n_random_orders = 100

k_grid_ident = np.asarray([
    0,
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
    120,
    len(final_perturbations_vc),
], dtype=int)


row_conditions_ident = np.asarray([
    row["condition"]
    for row in final_interval_rows
], dtype=object)


perturbations_ident = np.asarray(
    final_perturbations_vc,
    dtype=object
)


ident_path_records = []
first_full_rank = []


for rep in range(
    n_random_orders
):

    order = rng.permutation(
        perturbations_ident
    )

    first_full = None


    # ----------------------------------------
    # Search first k giving full rank
    # ----------------------------------------

    for k in range(
        len(order) + 1
    ):

        selected_conditions = np.concatenate([
            np.asarray(
                ["non-targeting"],
                dtype=object
            ),
            order[:k]
        ])

        row_mask = np.isin(
            row_conditions_ident,
            selected_conditions
        )

        X_sub = X_ident_scaled[
            row_mask
        ]

        rank_sub = np.linalg.matrix_rank(
            X_sub
        )

        if (
            rank_sub == n_cols_ident
            and first_full is None
        ):
            first_full = k
            break


    if first_full is None:
        first_full = np.nan

    first_full_rank.append(
        first_full
    )


    # ----------------------------------------
    # Fixed checkpoint path
    # ----------------------------------------

    for k in k_grid_ident:

        selected_conditions = np.concatenate([
            np.asarray(
                ["non-targeting"],
                dtype=object
            ),
            order[:k]
        ])

        row_mask = np.isin(
            row_conditions_ident,
            selected_conditions
        )

        X_sub = X_ident_scaled[
            row_mask
        ]

        n_sub_rows = X_sub.shape[0]

        rank_sub = np.linalg.matrix_rank(
            X_sub
        )


        # ------------------------------------
        # lambda_min of empirical information
        # ------------------------------------

        s_sub = np.linalg.svd(
            X_sub,
            compute_uv=False
        )

        lambda_max_sub = (
            s_sub[0] ** 2
            / n_sub_rows
        )

        # If fewer rows than columns or rank
        # deficient, lambda_min is exactly zero
        # in the full p x p information matrix.
        if rank_sub < n_cols_ident:

            lambda_min_sub = 0.0
            cond_sub = np.inf

        else:

            lambda_min_sub = (
                s_sub[-1] ** 2
                / n_sub_rows
            )

            cond_sub = (
                lambda_max_sub
                / lambda_min_sub
            )


        ident_path_records.append({
            "rep":
                rep,

            "n_perturbations":
                int(k),

            "n_conditions":
                int(k + 1),

            "n_rows":
                int(n_sub_rows),

            "rank":
                int(rank_sub),

            "full_rank":
                bool(
                    rank_sub
                    == n_cols_ident
                ),

            "lambda_min":
                float(
                    lambda_min_sub
                ),

            "condition_number":
                float(
                    cond_sub
                ),
        })


ident_path_df = pd.DataFrame(
    ident_path_records
)

first_full_rank = np.asarray(
    first_full_rank,
    dtype=float
)


# ============================================
# Summaries
# ============================================

print(
    "First-full-rank perturbation count:"
)

print(
    pd.Series(
        first_full_rank
    ).describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
)


print(
    "\nCheckpoint summary:"
)


summary_rows = []

for k in k_grid_ident:

    sub = ident_path_df[
        ident_path_df[
            "n_perturbations"
        ] == k
    ]

    positive_lambda = sub[
        "lambda_min"
    ].to_numpy()

    summary_rows.append({
        "k":
            int(k),

        "median_rank":
            float(
                np.median(
                    sub["rank"]
                )
            ),

        "rank_q10":
            float(
                np.quantile(
                    sub["rank"],
                    0.10
                )
            ),

        "rank_q90":
            float(
                np.quantile(
                    sub["rank"],
                    0.90
                )
            ),

        "full_rank_fraction":
            float(
                np.mean(
                    sub["full_rank"]
                )
            ),

        "lambda_min_median":
            float(
                np.median(
                    positive_lambda
                )
            ),

        "lambda_min_q10":
            float(
                np.quantile(
                    positive_lambda,
                    0.10
                )
            ),

        "lambda_min_q90":
            float(
                np.quantile(
                    positive_lambda,
                    0.90
                )
            ),
    })


ident_path_summary = pd.DataFrame(
    summary_rows
)

print(
    ident_path_summary.to_string(
        index=False
    )
)


print(
    "\nFull-data endpoint check:"
)

full_endpoint = ident_path_summary[
    ident_path_summary[
        "k"
    ] == len(
        final_perturbations_vc
    )
].iloc[0]

print(
    "full-rank fraction:",
    full_endpoint[
        "full_rank_fraction"
    ]
)

print(
    "lambda_min median:",
    full_endpoint[
        "lambda_min_median"
    ]
)

print(
    "Cell 100 lambda_min:",
    lambda_min_ident
)


# ============================================================================
# NOTEBOOK INDEX 107: Cell 102. Save corrected identifiability
# ============================================================================

# ============================================
# Cell 102. Save corrected identifiability
# results together with final GRN outputs
# ============================================

import numpy as np
import os


final_checkpoint_v2_path = (
    "/home/featurize/work/project1/"
    "replogle_final_exact_integral_grn_v2.npz"
)


np.savez_compressed(
    final_checkpoint_v2_path,

    # ----------------------------------------
    # Final exact-integral GRN
    # ----------------------------------------

    A_hat_integral=A_hat_integral,
    c_hat_integral=c_hat_integral,

    response_genes=np.asarray(
        response_genes,
        dtype=str
    ),

    regulator_genes=np.asarray(
        regulator_genes_vc,
        dtype=str
    ),

    global_alpha_integral=np.asarray(
        global_alpha_integral
    ),

    X_final=X_final,
    Y_final=Y_final,

    integral_n_nonzero=np.asarray(
        integral_fit_summary[
            "n_nonzero"
        ],
        dtype=int
    ),

    integral_skill=np.asarray(
        integral_fit_summary[
            "skill_vs_intercept"
        ],
        dtype=float
    ),

    integral_mse=np.asarray(
        integral_fit_summary[
            "integral_mse"
        ],
        dtype=float
    ),

    # ----------------------------------------
    # Strict phase-residual validation
    # ----------------------------------------

    global_alpha_resid=np.asarray(
        global_alpha_resid
    ),

    validation_genes=np.asarray(
        final_resid_oof_summary[
            "gene"
        ],
        dtype=str
    ),

    validation_incremental_r2=np.asarray(
        final_resid_oof_summary[
            "incremental_r2"
        ],
        dtype=float
    ),

    validation_relative_mse=np.asarray(
        final_resid_oof_summary[
            "relative_mse_vs_phase"
        ],
        dtype=float
    ),

    validation_mean_nonzero=np.asarray(
        final_resid_oof_summary[
            "mean_fold_nonzero"
        ],
        dtype=float
    ),

    validation_calibration_flag=np.asarray(
        final_resid_oof_summary[
            "used_for_alpha_calibration"
        ],
        dtype=bool
    ),

    # ----------------------------------------
    # Corrected identifiability analysis
    # ----------------------------------------

    ident_col_scale=ident_col_scale,

    ident_full_rank=np.asarray(
        rank_scaled
    ),

    ident_full_lambda_min=np.asarray(
        lambda_min_ident
    ),

    ident_full_lambda_max=np.asarray(
        lambda_max_ident
    ),

    ident_full_condition_number=np.asarray(
        lambda_max_ident
        / lambda_min_ident
    ),

    ident_first_full_rank=first_full_rank,

    ident_k_grid=k_grid_ident,

    ident_path_k=np.asarray(
        ident_path_df[
            "n_perturbations"
        ],
        dtype=int
    ),

    ident_path_rep=np.asarray(
        ident_path_df[
            "rep"
        ],
        dtype=int
    ),

    ident_path_rank=np.asarray(
        ident_path_df[
            "rank"
        ],
        dtype=int
    ),

    ident_path_full_rank=np.asarray(
        ident_path_df[
            "full_rank"
        ],
        dtype=bool
    ),

    ident_path_lambda_min=np.asarray(
        ident_path_df[
            "lambda_min"
        ],
        dtype=float
    ),

    ident_path_condition_number=np.asarray(
        ident_path_df[
            "condition_number"
        ],
        dtype=float
    ),
)


print(
    "Saved:",
    final_checkpoint_v2_path
)

print(
    "Exists:",
    os.path.exists(
        final_checkpoint_v2_path
    )
)

print(
    "Size MB:",
    os.path.getsize(
        final_checkpoint_v2_path
    ) / 1024**2
)


# --------------------------------------------
# Immediate reload checks
# --------------------------------------------

check_v2 = np.load(
    final_checkpoint_v2_path,
    allow_pickle=False
)

print(
    "\nA exact:",
    np.array_equal(
        check_v2[
            "A_hat_integral"
        ],
        A_hat_integral
    )
)

print(
    "Validation exact:",
    np.array_equal(
        check_v2[
            "validation_incremental_r2"
        ],
        final_resid_oof_summary[
            "incremental_r2"
        ].to_numpy()
    )
)

print(
    "First-full-rank exact:",
    np.array_equal(
        check_v2[
            "ident_first_full_rank"
        ],
        first_full_rank
    )
)

print(
    "Full lambda_min exact:",
    float(
        check_v2[
            "ident_full_lambda_min"
        ]
    )
    == lambda_min_ident
)


# ============================================================================
# NOTEBOOK INDEX 108: Cell 103. Pilot edge-stability analysis
# ============================================================================

# ============================================
# Cell 103. Pilot edge-stability analysis
#
# Exact integral estimator only.
#
# Protocol:
# - subsample perturbation CONDITIONS
# - NT always retained
# - 80% perturbation conditions per replicate
# - frozen global_alpha_integral
# - exact FWL treatment of dt
#
# Pilot:
# - 10 response genes
# - 20 replicates
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


rng_stab = np.random.default_rng(
    2026
)

n_stability_reps_pilot = 20

stability_fraction = 0.80


# --------------------------------------------
# Deterministic pilot genes:
# spread across response_genes + TACC3
# --------------------------------------------

pilot_idx = np.unique(
    np.round(
        np.linspace(
            0,
            len(response_genes) - 1,
            9
        )
    ).astype(int)
)

pilot_genes = list(
    response_genes[
        pilot_idx
    ]
)

if "TACC3" not in pilot_genes:
    pilot_genes.append(
        "TACC3"
    )

pilot_genes = np.asarray(
    pilot_genes,
    dtype=object
)


print(
    "Pilot genes:",
    len(pilot_genes)
)

print(
    pilot_genes
)


# --------------------------------------------
# Outputs
# --------------------------------------------

pilot_selection_count = np.zeros(
    (
        len(pilot_genes),
        len(regulator_genes_vc)
    ),
    dtype=int
)

pilot_positive_count = np.zeros_like(
    pilot_selection_count
)

pilot_negative_count = np.zeros_like(
    pilot_selection_count
)

pilot_edge_counts = np.zeros(
    (
        len(pilot_genes),
        n_stability_reps_pilot
    ),
    dtype=int
)


for pgi, g in enumerate(
    pilot_genes
):

    gi = np.where(
        response_genes == g
    )[0][0]

    valid_mask = valid_row_masks[g]

    d_all = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_all = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y_all = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )

    groups_all = row_conditions[
        valid_mask
    ]


    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_all
        )
        if q != "non-targeting"
    ], dtype=object)


    n_keep = int(
        np.floor(
            stability_fraction
            * len(pert_conditions)
        )
    )


    for rep in range(
        n_stability_reps_pilot
    ):

        kept_pert = rng_stab.choice(
            pert_conditions,
            size=n_keep,
            replace=False
        )

        kept_conditions = np.concatenate([
            np.asarray(
                ["non-targeting"],
                dtype=object
            ),
            kept_pert
        ])

        row_mask = np.isin(
            groups_all,
            kept_conditions
        )


        d = d_all[
            row_mask
        ]

        Xr = Xr_all[
            row_mask
        ]

        y = y_all[
            row_mask
        ]


        # ------------------------------------
        # Exact FWL projection
        # ------------------------------------

        dd = d @ d

        coef_d_X = (
            d @ Xr
        ) / dd

        coef_d_y = (
            d @ y
        ) / dd


        Xp = (
            Xr
            - d[:, None]
            * coef_d_X[None, :]
        )

        yp = (
            y
            - d * coef_d_y
        )


        # ------------------------------------
        # RMS scaling only
        # ------------------------------------

        x_scale = np.sqrt(
            np.mean(
                Xp ** 2,
                axis=0
            )
        )

        y_scale = np.sqrt(
            np.mean(
                yp ** 2
            )
        )

        assert np.all(
            np.isfinite(
                x_scale
            )
        )

        assert np.all(
            x_scale > 0
        )

        assert (
            np.isfinite(y_scale)
            and y_scale > 0
        )


        Z = (
            Xp
            / x_scale[None, :]
        )

        yz = (
            yp
            / y_scale
        )


        model = Lasso(
            alpha=global_alpha_integral,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z,
            yz
        )


        A_rep = (
            y_scale
            * model.coef_
            / x_scale
        )


        selected = (
            A_rep != 0
        )

        positive = (
            A_rep > 0
        )

        negative = (
            A_rep < 0
        )


        pilot_selection_count[
            pgi
        ] += selected.astype(int)

        pilot_positive_count[
            pgi
        ] += positive.astype(int)

        pilot_negative_count[
            pgi
        ] += negative.astype(int)

        pilot_edge_counts[
            pgi,
            rep
        ] = np.count_nonzero(
            A_rep
        )


    print(
        f"[{pgi + 1:02d}/"
        f"{len(pilot_genes)}] "
        f"{g} done"
    )


# ============================================
# Summaries
# ============================================

pilot_selection_freq = (
    pilot_selection_count
    / n_stability_reps_pilot
)

pilot_positive_freq = (
    pilot_positive_count
    / n_stability_reps_pilot
)

pilot_negative_freq = (
    pilot_negative_count
    / n_stability_reps_pilot
)


print(
    "\nEdges per replicate:"
)

print(
    pd.Series(
        pilot_edge_counts.ravel()
    ).describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


print(
    "\nSelection-frequency distribution:"
)

print(
    pd.Series(
        pilot_selection_freq.ravel()
    ).describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


print(
    "\nEdges with selection frequency >= 0.8:",
    np.sum(
        pilot_selection_freq >= 0.8
    )
)

print(
    "Edges with selection frequency == 1:",
    np.sum(
        pilot_selection_freq == 1.0
    )
)


# --------------------------------------------
# Sign consistency among highly stable edges
# --------------------------------------------

stable_mask = (
    pilot_selection_freq >= 0.8
)

sign_consistency = np.maximum(
    pilot_positive_freq,
    pilot_negative_freq
)

if np.any(
    stable_mask
):

    print(
        "\nStable-edge sign consistency:"
    )

    print(
        pd.Series(
            sign_consistency[
                stable_mask
            ]
        ).describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
    )

else:

    print(
        "\nNo edges reached selection frequency >= 0.8"
    )


# ============================================================================
# NOTEBOOK INDEX 109: Cell 104. Full edge-stability analysis
# ============================================================================

# ============================================
# Cell 104. Full edge-stability analysis
#
# 426 response genes
# 151 candidate regulators
# 100 condition-subsampling replicates
#
# Protocol:
# - retain 80% perturbation conditions
# - NT always retained
# - exact integral estimator
# - frozen global_alpha_integral
# - direct q=g exclusion preserved
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


n_stability_reps = 100
stability_fraction = 0.80

rng_stab_full = np.random.default_rng(
    2026
)


n_response = len(
    response_genes
)

n_regulator = len(
    regulator_genes_vc
)


selection_count = np.zeros(
    (
        n_response,
        n_regulator
    ),
    dtype=np.uint16
)

positive_count = np.zeros_like(
    selection_count
)

negative_count = np.zeros_like(
    selection_count
)

edge_count_by_rep = np.zeros(
    (
        n_response,
        n_stability_reps
    ),
    dtype=np.uint16
)


for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

    d_all = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_all = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y_all = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )

    groups_all = row_conditions[
        valid_mask
    ]


    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_all
        )
        if q != "non-targeting"
    ], dtype=object)


    n_keep = int(
        np.floor(
            stability_fraction
            * len(pert_conditions)
        )
    )


    for rep in range(
        n_stability_reps
    ):

        kept_pert = rng_stab_full.choice(
            pert_conditions,
            size=n_keep,
            replace=False
        )

        kept_conditions = np.concatenate([
            np.asarray(
                ["non-targeting"],
                dtype=object
            ),
            kept_pert
        ])

        row_mask = np.isin(
            groups_all,
            kept_conditions
        )


        d = d_all[
            row_mask
        ]

        Xr = Xr_all[
            row_mask
        ]

        y = y_all[
            row_mask
        ]


        # ------------------------------------
        # Exact FWL projection
        # ------------------------------------

        dd = d @ d

        coef_d_X = (
            d @ Xr
        ) / dd

        coef_d_y = (
            d @ y
        ) / dd


        Xp = (
            Xr
            - d[:, None]
            * coef_d_X[None, :]
        )

        yp = (
            y
            - d * coef_d_y
        )


        # ------------------------------------
        # RMS scaling only, no centering
        # ------------------------------------

        x_scale = np.sqrt(
            np.mean(
                Xp ** 2,
                axis=0
            )
        )

        y_scale = np.sqrt(
            np.mean(
                yp ** 2
            )
        )

        assert np.all(
            np.isfinite(
                x_scale
            )
        )

        assert np.all(
            x_scale > 0
        )

        assert (
            np.isfinite(y_scale)
            and y_scale > 0
        )


        Z = (
            Xp
            / x_scale[None, :]
        )

        yz = (
            yp
            / y_scale
        )


        model = Lasso(
            alpha=global_alpha_integral,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z,
            yz
        )


        A_rep = (
            y_scale
            * model.coef_
            / x_scale
        )


        selected = (
            A_rep != 0
        )

        selection_count[
            gi
        ] += selected

        positive_count[
            gi
        ] += (
            A_rep > 0
        )

        negative_count[
            gi
        ] += (
            A_rep < 0
        )

        edge_count_by_rep[
            gi,
            rep
        ] = np.count_nonzero(
            A_rep
        )


    if (
        (gi + 1) % 25 == 0
        or gi == 0
        or gi + 1 == n_response
    ):

        print(
            f"[{gi + 1:03d}/"
            f"{n_response}] "
            f"{g} done"
        )


# ============================================
# Frequencies
# ============================================

selection_freq = (
    selection_count.astype(
        np.float64
    )
    / n_stability_reps
)

positive_freq = (
    positive_count.astype(
        np.float64
    )
    / n_stability_reps
)

negative_freq = (
    negative_count.astype(
        np.float64
    )
    / n_stability_reps
)

sign_consistency = np.maximum(
    positive_freq,
    negative_freq
)


# ============================================
# Global summaries
# ============================================

print(
    "\nEdges per subsampled fit:"
)

print(
    pd.Series(
        edge_count_by_rep.ravel()
    ).describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


print(
    "\nSelection-frequency distribution:"
)

print(
    pd.Series(
        selection_freq.ravel()
    ).describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


for threshold in [
    0.50,
    0.80,
    0.90,
    0.95,
    1.00,
]:

    mask = (
        selection_freq
        >= threshold
    )

    print(
        f"\nSelection frequency >= "
        f"{threshold:.2f}:",
        int(
            mask.sum()
        ),
        "edges"
    )

    print(
        "fraction of all candidate edges:",
        float(
            mask.mean()
        )
    )


# ============================================
# Primary stable-edge threshold: >= 0.8
# ============================================

stable80 = (
    selection_freq >= 0.80
)

print(
    "\nStable >=0.8 edges per response gene:"
)

stable_per_gene = np.sum(
    stable80,
    axis=1
)

print(
    pd.Series(
        stable_per_gene
    ).describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


print(
    "\nSign consistency among >=0.8 edges:"
)

print(
    pd.Series(
        sign_consistency[
            stable80
        ]
    ).describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


# ============================================
# Agreement with final full-data network
# ============================================

full_selected = (
    A_hat_integral != 0
)

print(
    "\nAmong full-data selected edges:"
)

print(
    "Total:",
    int(
        full_selected.sum()
    )
)

print(
    "selection freq >=0.8:",
    int(
        np.sum(
            full_selected
            & stable80
        )
    )
)

print(
    "fraction stable >=0.8:",
    float(
        np.mean(
            selection_freq[
                full_selected
            ]
            >= 0.8
        )
    )
)


full_sign = np.sign(
    A_hat_integral
)

stable_sign = np.where(
    positive_freq >= negative_freq,
    1,
    -1
)

stable_full_mask = (
    full_selected
    & stable80
)

print(
    "\nFull-data sign agreement among "
    "stable selected edges:"
)

print(
    np.mean(
        full_sign[
            stable_full_mask
        ]
        ==
        stable_sign[
            stable_full_mask
        ]
    )
)


# ============================================================================
# NOTEBOOK INDEX 110: Cell 105. Save full edge-stability results
# ============================================================================

# ============================================
# Cell 105. Save full edge-stability results
# into final checkpoint v3
# ============================================

import numpy as np
import os


final_checkpoint_v3_path = (
    "/home/featurize/work/project1/"
    "replogle_final_exact_integral_grn_v3.npz"
)


np.savez_compressed(
    final_checkpoint_v3_path,

    # ========================================
    # Final exact-integral GRN
    # ========================================

    A_hat_integral=A_hat_integral,
    c_hat_integral=c_hat_integral,

    response_genes=np.asarray(
        response_genes,
        dtype=str
    ),

    regulator_genes=np.asarray(
        regulator_genes_vc,
        dtype=str
    ),

    global_alpha_integral=np.asarray(
        global_alpha_integral
    ),

    X_final=X_final,
    Y_final=Y_final,

    integral_n_nonzero=np.asarray(
        integral_fit_summary[
            "n_nonzero"
        ],
        dtype=int
    ),

    integral_skill=np.asarray(
        integral_fit_summary[
            "skill_vs_intercept"
        ],
        dtype=float
    ),

    integral_mse=np.asarray(
        integral_fit_summary[
            "integral_mse"
        ],
        dtype=float
    ),


    # ========================================
    # Strict phase-residualized validation
    # ========================================

    global_alpha_resid=np.asarray(
        global_alpha_resid
    ),

    validation_genes=np.asarray(
        final_resid_oof_summary[
            "gene"
        ],
        dtype=str
    ),

    validation_incremental_r2=np.asarray(
        final_resid_oof_summary[
            "incremental_r2"
        ],
        dtype=float
    ),

    validation_relative_mse=np.asarray(
        final_resid_oof_summary[
            "relative_mse_vs_phase"
        ],
        dtype=float
    ),

    validation_mean_nonzero=np.asarray(
        final_resid_oof_summary[
            "mean_fold_nonzero"
        ],
        dtype=float
    ),

    validation_calibration_flag=np.asarray(
        final_resid_oof_summary[
            "used_for_alpha_calibration"
        ],
        dtype=bool
    ),


    # ========================================
    # Corrected identifiability analysis
    # ========================================

    ident_col_scale=ident_col_scale,

    ident_full_rank=np.asarray(
        rank_scaled
    ),

    ident_full_lambda_min=np.asarray(
        lambda_min_ident
    ),

    ident_full_lambda_max=np.asarray(
        lambda_max_ident
    ),

    ident_full_condition_number=np.asarray(
        lambda_max_ident
        / lambda_min_ident
    ),

    ident_first_full_rank=first_full_rank,

    ident_k_grid=k_grid_ident,

    ident_path_k=np.asarray(
        ident_path_df[
            "n_perturbations"
        ],
        dtype=int
    ),

    ident_path_rep=np.asarray(
        ident_path_df[
            "rep"
        ],
        dtype=int
    ),

    ident_path_rank=np.asarray(
        ident_path_df[
            "rank"
        ],
        dtype=int
    ),

    ident_path_full_rank=np.asarray(
        ident_path_df[
            "full_rank"
        ],
        dtype=bool
    ),

    ident_path_lambda_min=np.asarray(
        ident_path_df[
            "lambda_min"
        ],
        dtype=float
    ),

    ident_path_condition_number=np.asarray(
        ident_path_df[
            "condition_number"
        ],
        dtype=float
    ),


    # ========================================
    # Full stability analysis
    # ========================================

    stability_n_reps=np.asarray(
        n_stability_reps
    ),

    stability_fraction=np.asarray(
        stability_fraction
    ),

    stability_selection_count=selection_count,

    stability_positive_count=positive_count,

    stability_negative_count=negative_count,

    stability_selection_freq=selection_freq,

    stability_positive_freq=positive_freq,

    stability_negative_freq=negative_freq,

    stability_sign_consistency=sign_consistency,

    stability_edge_count_by_rep=edge_count_by_rep,
)


print(
    "Saved:",
    final_checkpoint_v3_path
)

print(
    "Exists:",
    os.path.exists(
        final_checkpoint_v3_path
    )
)

print(
    "Size MB:",
    os.path.getsize(
        final_checkpoint_v3_path
    ) / 1024**2
)


# --------------------------------------------
# Reload checks
# --------------------------------------------

check_v3 = np.load(
    final_checkpoint_v3_path,
    allow_pickle=False
)

print(
    "\nA exact:",
    np.array_equal(
        check_v3[
            "A_hat_integral"
        ],
        A_hat_integral
    )
)

print(
    "Selection frequency exact:",
    np.array_equal(
        check_v3[
            "stability_selection_freq"
        ],
        selection_freq
    )
)

print(
    "Positive frequency exact:",
    np.array_equal(
        check_v3[
            "stability_positive_freq"
        ],
        positive_freq
    )
)

print(
    "Negative frequency exact:",
    np.array_equal(
        check_v3[
            "stability_negative_freq"
        ],
        negative_freq
    )
)

print(
    "Identifiability exact:",
    np.array_equal(
        check_v3[
            "ident_first_full_rank"
        ],
        first_full_rank
    )
)


# --------------------------------------------
# Key stability invariants
# --------------------------------------------

stable80_check = (
    check_v3[
        "stability_selection_freq"
    ] >= 0.8
)

full_selected_check = (
    check_v3[
        "A_hat_integral"
    ] != 0
)

print(
    "\nStable >=0.8 edges:",
    int(
        stable80_check.sum()
    )
)

print(
    "All stable edges in full network:",
    bool(
        np.all(
            full_selected_check[
                stable80_check
            ]
        )
    )
)

print(
    "Fraction full-network edges stable >=0.8:",
    float(
        stable80_check[
            full_selected_check
        ].mean()
    )
)


# ============================================================================
# NOTEBOOK INDEX 111: Cell 106. Build final edge-level table
# ============================================================================

# ============================================
# Cell 106. Build final edge-level table
#
# One row = one response gene <- regulator edge
# Includes:
# - exact-integral coefficient
# - full-data selection
# - stability frequency
# - positive/negative frequency
# - sign consistency
# - stable >= 0.8 flag
# ============================================

import numpy as np
import pandas as pd


edge_records = []


for gi, response in enumerate(
    response_genes
):

    for rj, regulator in enumerate(
        regulator_genes_vc
    ):

        coef = float(
            A_hat_integral[
                gi, rj
            ]
        )

        sel_freq = float(
            selection_freq[
                gi, rj
            ]
        )

        pos_freq = float(
            positive_freq[
                gi, rj
            ]
        )

        neg_freq = float(
            negative_freq[
                gi, rj
            ]
        )

        sign_cons = float(
            sign_consistency[
                gi, rj
            ]
        )

        full_selected = (
            coef != 0.0
        )

        stable80 = (
            sel_freq >= 0.80
        )

        if coef > 0:
            sign = 1
        elif coef < 0:
            sign = -1
        else:
            sign = 0

        edge_records.append({
            "response_gene":
                str(response),

            "regulator_gene":
                str(regulator),

            "coefficient":
                coef,

            "abs_coefficient":
                abs(coef),

            "sign":
                sign,

            "full_selected":
                full_selected,

            "selection_frequency":
                sel_freq,

            "positive_frequency":
                pos_freq,

            "negative_frequency":
                neg_freq,

            "sign_consistency":
                sign_cons,

            "stable_80":
                stable80,

            "stable_90":
                sel_freq >= 0.90,

            "stable_95":
                sel_freq >= 0.95,

            "stable_100":
                sel_freq == 1.00,

            "self_edge":
                str(response)
                == str(regulator),
        })


edge_table = pd.DataFrame(
    edge_records
)


print(
    "Edge table shape:",
    edge_table.shape
)

print(
    "Expected rows:",
    len(response_genes)
    * len(regulator_genes_vc)
)


print(
    "\nFull selected:",
    int(
        edge_table[
            "full_selected"
        ].sum()
    )
)

print(
    "Stable >=0.8:",
    int(
        edge_table[
            "stable_80"
        ].sum()
    )
)

print(
    "Stable >=0.9:",
    int(
        edge_table[
            "stable_90"
        ].sum()
    )
)

print(
    "Stable >=0.95:",
    int(
        edge_table[
            "stable_95"
        ].sum()
    )
)

print(
    "Stable ==1.0:",
    int(
        edge_table[
            "stable_100"
        ].sum()
    )
)


# --------------------------------------------
# Important invariant:
# all >=0.8 stable edges should be selected
# in the full-data network
# --------------------------------------------

print(
    "\nAll stable >=0.8 edges "
    "selected in full network:",
    bool(
        edge_table.loc[
            edge_table[
                "stable_80"
            ],
            "full_selected"
        ].all()
    )
)


# --------------------------------------------
# Stable-core sign agreement
# --------------------------------------------

stable_edges = edge_table[
    edge_table[
        "stable_80"
    ]
].copy()

stable_sign_agreement = np.where(
    stable_edges[
        "sign"
    ].to_numpy() > 0,
    stable_edges[
        "positive_frequency"
    ].to_numpy(),
    stable_edges[
        "negative_frequency"
    ].to_numpy()
)

print(
    "\nStable-core sign agreement:"
)

print(
    pd.Series(
        stable_sign_agreement
    ).describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


# --------------------------------------------
# Self-edge bookkeeping
# Only possible where response gene is also
# one of the 151 regulator genes.
# --------------------------------------------

self_edges = edge_table[
    edge_table[
        "self_edge"
    ]
]

print(
    "\nPossible self edges:",
    len(self_edges)
)

print(
    "Selected self edges:",
    int(
        self_edges[
            "full_selected"
        ].sum()
    )
)

print(
    "Stable >=0.8 self edges:",
    int(
        self_edges[
            "stable_80"
        ].sum()
    )
)


# --------------------------------------------
# Preview strongest stable edges
# by absolute coefficient
# --------------------------------------------

print(
    "\nTop 20 stable edges "
    "by |coefficient|:"
)

print(
    stable_edges
    .sort_values(
        "abs_coefficient",
        ascending=False
    )
    .head(20)
    [
        [
            "response_gene",
            "regulator_gene",
            "coefficient",
            "selection_frequency",
            "sign_consistency",
            "self_edge",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 112: Cell 107. Regulator-level network summary
# ============================================================================

# ============================================
# Cell 107. Regulator-level network summary
#
# Rank regulators primarily by:
# - number of stable outgoing edges
# - fraction of selected edges that are stable
#
# Do NOT rank primarily by raw coefficient size.
# ============================================

import numpy as np
import pandas as pd


regulator_summary_records = []


for regulator in regulator_genes_vc:

    sub = edge_table[
        edge_table[
            "regulator_gene"
        ] == regulator
    ]

    selected = sub[
        "full_selected"
    ].to_numpy()

    stable = sub[
        "stable_80"
    ].to_numpy()

    coef = sub[
        "coefficient"
    ].to_numpy(
        dtype=float
    )

    sel_freq = sub[
        "selection_frequency"
    ].to_numpy(
        dtype=float
    )


    n_selected = int(
        selected.sum()
    )

    n_stable = int(
        stable.sum()
    )

    n_stable_positive = int(
        np.sum(
            stable
            & (coef > 0)
        )
    )

    n_stable_negative = int(
        np.sum(
            stable
            & (coef < 0)
        )
    )


    if n_selected > 0:

        stable_fraction_selected = (
            n_stable
            / n_selected
        )

        mean_freq_selected = float(
            np.mean(
                sel_freq[
                    selected
                ]
            )
        )

    else:

        stable_fraction_selected = np.nan
        mean_freq_selected = np.nan


    if n_stable > 0:

        mean_freq_stable = float(
            np.mean(
                sel_freq[
                    stable
                ]
            )
        )

        median_freq_stable = float(
            np.median(
                sel_freq[
                    stable
                ]
            )
        )

    else:

        mean_freq_stable = np.nan
        median_freq_stable = np.nan


    regulator_summary_records.append({
        "regulator_gene":
            str(regulator),

        "n_selected_outgoing":
            n_selected,

        "n_stable80_outgoing":
            n_stable,

        "stable_fraction_selected":
            stable_fraction_selected,

        "mean_selection_freq_selected":
            mean_freq_selected,

        "mean_selection_freq_stable":
            mean_freq_stable,

        "median_selection_freq_stable":
            median_freq_stable,

        "n_stable_positive":
            n_stable_positive,

        "n_stable_negative":
            n_stable_negative,

        "stable_positive_fraction":
            (
                n_stable_positive
                / n_stable
                if n_stable > 0
                else np.nan
            ),

        "stable_negative_fraction":
            (
                n_stable_negative
                / n_stable
                if n_stable > 0
                else np.nan
            ),
    })


regulator_summary = pd.DataFrame(
    regulator_summary_records
)


# --------------------------------------------
# Global distribution
# --------------------------------------------

print(
    "Selected outgoing edges per regulator:"
)

print(
    regulator_summary[
        "n_selected_outgoing"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


print(
    "\nStable >=0.8 outgoing edges "
    "per regulator:"
)

print(
    regulator_summary[
        "n_stable80_outgoing"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


print(
    "\nStable fraction among "
    "full-selected outgoing edges:"
)

print(
    regulator_summary[
        "stable_fraction_selected"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


# --------------------------------------------
# Regulators with most stable outgoing edges
# --------------------------------------------

print(
    "\nTop 25 regulators by "
    "stable outgoing edge count:"
)

print(
    regulator_summary
    .sort_values(
        [
            "n_stable80_outgoing",
            "stable_fraction_selected",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .head(25)
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Regulators with fewest stable outgoing edges
# --------------------------------------------

print(
    "\nBottom 15 regulators by "
    "stable outgoing edge count:"
)

print(
    regulator_summary
    .sort_values(
        [
            "n_stable80_outgoing",
            "stable_fraction_selected",
        ],
        ascending=[
            True,
            True,
        ]
    )
    .head(15)
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Sanity check: totals must match edge table
# --------------------------------------------

print(
    "\nTotal selected outgoing edges:",
    int(
        regulator_summary[
            "n_selected_outgoing"
        ].sum()
    )
)

print(
    "Expected:",
    int(
        edge_table[
            "full_selected"
        ].sum()
    )
)

print(
    "\nTotal stable outgoing edges:",
    int(
        regulator_summary[
            "n_stable80_outgoing"
        ].sum()
    )
)

print(
    "Expected:",
    int(
        edge_table[
            "stable_80"
        ].sum()
    )
)


# ============================================================================
# NOTEBOOK INDEX 113: Cell 108. Regulator hub confounding diagnostic
# ============================================================================

# ============================================
# Cell 108. Regulator hub confounding diagnostic
#
# Question:
# Is stable outgoing degree mainly explained by
# regulator trajectory scale / variability?
#
# Uses final S_regulator_hat only.
# No model refitting.
# ============================================

import numpy as np
import pandas as pd


regulator_diagnostic_records = []


for rj, regulator in enumerate(
    regulator_genes_vc
):

    values = np.asarray(
        S_regulator_hat[
            :, :, rj
        ],
        dtype=np.float64
    ).ravel()


    # ----------------------------------------
    # Basic trajectory scale
    # ----------------------------------------

    mean_level = float(
        np.mean(values)
    )

    median_level = float(
        np.median(values)
    )

    rms_level = float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )

    sd_level = float(
        np.std(values)
    )


    # ----------------------------------------
    # Condition-level mean trajectory
    # ----------------------------------------

    condition_means = np.mean(
        S_regulator_hat[
            :, :, rj
        ],
        axis=1
    )

    condition_sd = float(
        np.std(
            condition_means
        )
    )

    condition_range = float(
        np.max(
            condition_means
        )
        -
        np.min(
            condition_means
        )
    )


    # ----------------------------------------
    # Phase variability within conditions
    # ----------------------------------------

    within_condition_sd = np.std(
        S_regulator_hat[
            :, :, rj
        ],
        axis=1
    )

    mean_phase_sd = float(
        np.mean(
            within_condition_sd
        )
    )


    regulator_diagnostic_records.append({
        "regulator_gene":
            str(regulator),

        "trajectory_mean":
            mean_level,

        "trajectory_median":
            median_level,

        "trajectory_rms":
            rms_level,

        "trajectory_sd":
            sd_level,

        "condition_mean_sd":
            condition_sd,

        "condition_mean_range":
            condition_range,

        "mean_phase_sd":
            mean_phase_sd,
    })


regulator_diagnostics = pd.DataFrame(
    regulator_diagnostic_records
)


# --------------------------------------------
# Merge with network summary
# --------------------------------------------

regulator_hub_diagnostic = (
    regulator_summary
    .merge(
        regulator_diagnostics,
        on="regulator_gene",
        how="left",
        validate="one_to_one"
    )
)


print(
    "Diagnostic table shape:",
    regulator_hub_diagnostic.shape
)


# ============================================
# Spearman correlations
# ============================================

diagnostic_cols = [
    "trajectory_mean",
    "trajectory_rms",
    "trajectory_sd",
    "condition_mean_sd",
    "condition_mean_range",
    "mean_phase_sd",
]


print(
    "\nSpearman correlation with "
    "stable outgoing degree:"
)

corr_stable = (
    regulator_hub_diagnostic[
        [
            "n_stable80_outgoing"
        ]
        + diagnostic_cols
    ]
    .corr(
        method="spearman"
    )
    .loc[
        diagnostic_cols,
        "n_stable80_outgoing"
    ]
    .sort_values(
        ascending=False
    )
)

print(
    corr_stable
)


print(
    "\nSpearman correlation with "
    "full selected outgoing degree:"
)

corr_selected = (
    regulator_hub_diagnostic[
        [
            "n_selected_outgoing"
        ]
        + diagnostic_cols
    ]
    .corr(
        method="spearman"
    )
    .loc[
        diagnostic_cols,
        "n_selected_outgoing"
    ]
    .sort_values(
        ascending=False
    )
)

print(
    corr_selected
)


# ============================================
# Inspect top stable hubs with diagnostics
# ============================================

print(
    "\nTop 25 stable hubs + "
    "trajectory diagnostics:"
)

print(
    regulator_hub_diagnostic
    .sort_values(
        "n_stable80_outgoing",
        ascending=False
    )
    .head(25)
    [
        [
            "regulator_gene",
            "n_stable80_outgoing",
            "stable_fraction_selected",
            "trajectory_mean",
            "trajectory_rms",
            "trajectory_sd",
            "condition_mean_sd",
            "condition_mean_range",
            "mean_phase_sd",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================
# Rank correlations among network metrics
# ============================================

print(
    "\nNetwork-metric Spearman correlations:"
)

print(
    regulator_hub_diagnostic[
        [
            "n_selected_outgoing",
            "n_stable80_outgoing",
            "stable_fraction_selected",
            "mean_selection_freq_selected",
        ]
    ]
    .corr(
        method="spearman"
    )
)


# ============================================================================
# NOTEBOOK INDEX 114: Cell 109. Regulator-feature redundancy
# ============================================================================

# ============================================
# Cell 109. Regulator-feature redundancy
# diagnostic
#
# Question:
# Are some regulator columns strongly
# collinear in the exact integral design?
#
# Uses X_final regulator columns.
# NO centering of the model design itself.
#
# Correlation here is diagnostic only.
# ============================================

import numpy as np
import pandas as pd


Xr_diag = np.asarray(
    X_final[:, 1:],
    dtype=np.float64
)


# --------------------------------------------
# Pearson correlation is used only as a
# redundancy diagnostic.
# --------------------------------------------

corr_reg = np.corrcoef(
    Xr_diag,
    rowvar=False
)

np.fill_diagonal(
    corr_reg,
    np.nan
)

abs_corr_reg = np.abs(
    corr_reg
)


# ============================================
# Per-regulator redundancy metrics
# ============================================

max_abs_corr = np.nanmax(
    abs_corr_reg,
    axis=1
)

mean_abs_corr = np.nanmean(
    abs_corr_reg,
    axis=1
)

n_corr_080 = np.sum(
    abs_corr_reg >= 0.80,
    axis=1
)

n_corr_090 = np.sum(
    abs_corr_reg >= 0.90,
    axis=1
)

n_corr_095 = np.sum(
    abs_corr_reg >= 0.95,
    axis=1
)


redundancy_df = pd.DataFrame({
    "regulator_gene":
        np.asarray(
            regulator_genes_vc,
            dtype=str
        ),

    "max_abs_corr":
        max_abs_corr,

    "mean_abs_corr":
        mean_abs_corr,

    "n_abs_corr_ge_080":
        n_corr_080,

    "n_abs_corr_ge_090":
        n_corr_090,

    "n_abs_corr_ge_095":
        n_corr_095,
})


regulator_hub_redundancy = (
    regulator_hub_diagnostic
    .merge(
        redundancy_df,
        on="regulator_gene",
        how="left",
        validate="one_to_one"
    )
)


# ============================================
# Global pairwise correlation distribution
# ============================================

upper = np.triu_indices(
    len(regulator_genes_vc),
    k=1
)

pair_abs_corr = abs_corr_reg[
    upper
]


print(
    "Pairwise |correlation| distribution:"
)

print(
    pd.Series(
        pair_abs_corr
    ).describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


for threshold in [
    0.80,
    0.90,
    0.95,
]:

    print(
        f"\nPairs with |corr| >= "
        f"{threshold:.2f}:",
        int(
            np.sum(
                pair_abs_corr
                >= threshold
            )
        )
    )


# ============================================
# Relationship with stable outdegree
# ============================================

print(
    "\nSpearman correlations with "
    "stable outgoing degree:"
)

cols = [
    "max_abs_corr",
    "mean_abs_corr",
    "n_abs_corr_ge_080",
    "n_abs_corr_ge_090",
    "n_abs_corr_ge_095",
]

print(
    regulator_hub_redundancy[
        [
            "n_stable80_outgoing"
        ]
        + cols
    ]
    .corr(
        method="spearman"
    )
    .loc[
        cols,
        "n_stable80_outgoing"
    ]
    .sort_values(
        ascending=False
    )
)


# ============================================
# Top stable hubs + redundancy
# ============================================

print(
    "\nTop 25 stable hubs + redundancy:"
)

print(
    regulator_hub_redundancy
    .sort_values(
        "n_stable80_outgoing",
        ascending=False
    )
    .head(25)
    [
        [
            "regulator_gene",
            "n_stable80_outgoing",
            "stable_fraction_selected",
            "max_abs_corr",
            "mean_abs_corr",
            "n_abs_corr_ge_080",
            "n_abs_corr_ge_090",
            "n_abs_corr_ge_095",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================
# Most highly correlated regulator pairs
# ============================================

pair_records = []

for i in range(
    len(regulator_genes_vc)
):

    for j in range(
        i + 1,
        len(regulator_genes_vc)
    ):

        pair_records.append({
            "regulator_1":
                str(
                    regulator_genes_vc[i]
                ),

            "regulator_2":
                str(
                    regulator_genes_vc[j]
                ),

            "correlation":
                float(
                    corr_reg[i, j]
                ),

            "abs_correlation":
                float(
                    abs_corr_reg[i, j]
                ),
        })


regulator_pair_corr = pd.DataFrame(
    pair_records
)


print(
    "\nTop 20 most correlated "
    "regulator pairs:"
)

print(
    regulator_pair_corr
    .sort_values(
        "abs_correlation",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 115: Cell 110. Diagnose the 12 possible self-edges
# ============================================================================

# ============================================
# Cell 110. Diagnose the 12 possible self-edges
#
# Goal:
# inspect whether stable self-edges are
# unusually strong / unusually stable and
# whether they are associated with regulator
# redundancy.
#
# No model refitting.
# ============================================

import numpy as np
import pandas as pd


self_edge_diag = (
    edge_table[
        edge_table[
            "self_edge"
        ]
    ]
    .merge(
        redundancy_df,
        on="regulator_gene",
        how="left",
        validate="one_to_one"
    )
    .copy()
)


# --------------------------------------------
# Rank each self-edge among all outgoing edges
# of the same regulator.
#
# Rank 1 = largest |coefficient|
# Rank 1 = highest selection frequency
# --------------------------------------------

coef_ranks = []
freq_ranks = []

for _, row in self_edge_diag.iterrows():

    regulator = row[
        "regulator_gene"
    ]

    response = row[
        "response_gene"
    ]

    outgoing = edge_table[
        edge_table[
            "regulator_gene"
        ] == regulator
    ].copy()


    coef_rank = (
        outgoing[
            "abs_coefficient"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .loc[
            outgoing[
                "response_gene"
            ] == response
        ]
        .iloc[0]
    )


    freq_rank = (
        outgoing[
            "selection_frequency"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .loc[
            outgoing[
                "response_gene"
            ] == response
        ]
        .iloc[0]
    )


    coef_ranks.append(
        int(coef_rank)
    )

    freq_ranks.append(
        int(freq_rank)
    )


self_edge_diag[
    "abs_coef_outgoing_rank"
] = coef_ranks

self_edge_diag[
    "selection_freq_outgoing_rank"
] = freq_ranks


# --------------------------------------------
# Compare self-edge stability against
# non-self selected edges
# --------------------------------------------

selected_self = edge_table[
    edge_table[
        "self_edge"
    ]
    &
    edge_table[
        "full_selected"
    ]
]

selected_nonself = edge_table[
    (~edge_table[
        "self_edge"
    ])
    &
    edge_table[
        "full_selected"
    ]
]


print(
    "Possible self edges:",
    len(
        self_edge_diag
    )
)

print(
    "Selected self edges:",
    len(
        selected_self
    )
)

print(
    "Stable >=0.8 self edges:",
    int(
        selected_self[
            "stable_80"
        ].sum()
    )
)


print(
    "\nSelf-edge selection frequencies:"
)

print(
    self_edge_diag[
        "selection_frequency"
    ].describe()
)


print(
    "\nSelected non-self edge "
    "selection frequencies:"
)

print(
    selected_nonself[
        "selection_frequency"
    ].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nDetailed self-edge diagnostics:"
)

print(
    self_edge_diag[
        [
            "response_gene",
            "regulator_gene",
            "coefficient",
            "selection_frequency",
            "positive_frequency",
            "negative_frequency",
            "sign_consistency",
            "abs_coef_outgoing_rank",
            "selection_freq_outgoing_rank",
            "max_abs_corr",
            "n_abs_corr_ge_080",
        ]
    ]
    .sort_values(
        "selection_frequency",
        ascending=False
    )
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Sign composition
# --------------------------------------------

print(
    "\nSelected self-edge signs:"
)

print(
    selected_self[
        "sign"
    ].value_counts()
    .sort_index()
)


# --------------------------------------------
# Compare stability proportions
# --------------------------------------------

self_stable_fraction = float(
    selected_self[
        "stable_80"
    ].mean()
)

nonself_stable_fraction = float(
    selected_nonself[
        "stable_80"
    ].mean()
)

print(
    "\nStable fraction among "
    "selected self edges:",
    self_stable_fraction
)

print(
    "Stable fraction among "
    "selected non-self edges:",
    nonself_stable_fraction
)

print(
    "Ratio:",
    self_stable_fraction
    / nonself_stable_fraction
)


# ============================================================================
# NOTEBOOK INDEX 116: Cell 111. Diagnose self-signal coupling
# ============================================================================

# ============================================
# Cell 111. Diagnose self-signal coupling
#
# For the 12 genes that are both:
# - response genes
# - regulator genes
#
# Compare condition-level response trajectory
# against the same-gene regulator trajectory.
#
# IMPORTANT:
# - exclude q = g condition
# - diagnostic only
# - no model refitting
# ============================================

import numpy as np
import pandas as pd


self_coupling_records = []


condition_names_arr = np.asarray(
    final_conditions_vc,
    dtype=object
)


overlap_genes = np.intersect1d(
    np.asarray(
        response_genes,
        dtype=object
    ),
    np.asarray(
        regulator_genes_vc,
        dtype=object
    )
)


print(
    "Overlap genes:",
    len(overlap_genes)
)

print(
    overlap_genes
)


for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]

    rj = np.where(
        np.asarray(
            regulator_genes_vc,
            dtype=object
        ) == g
    )[0][0]


    # ----------------------------------------
    # Condition-level means
    # ----------------------------------------

    response_mean = np.mean(
        S_response_hat[
            :, :, gi
        ],
        axis=1
    )

    regulator_mean = np.mean(
        S_regulator_hat[
            :, :, rj
        ],
        axis=1
    )


    # Exclude the direct perturbation q = g
    keep = (
        condition_names_arr
        != g
    )

    x = regulator_mean[
        keep
    ]

    y = response_mean[
        keep
    ]


    pearson = float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )

    spearman = float(
        pd.Series(x).corr(
            pd.Series(y),
            method="spearman"
        )
    )


    # ----------------------------------------
    # Compare NT-relative condition effects
    # instead of absolute expression levels
    # ----------------------------------------

    nt_idx = np.where(
        condition_names_arr
        == "non-targeting"
    )[0][0]

    eps = 1e-12

    response_log_ratio = np.log(
        (
            response_mean
            + eps
        )
        /
        (
            response_mean[
                nt_idx
            ]
            + eps
        )
    )

    regulator_log_ratio = np.log(
        (
            regulator_mean
            + eps
        )
        /
        (
            regulator_mean[
                nt_idx
            ]
            + eps
        )
    )


    x_lr = regulator_log_ratio[
        keep
    ]

    y_lr = response_log_ratio[
        keep
    ]


    pearson_logratio = float(
        np.corrcoef(
            x_lr,
            y_lr
        )[0, 1]
    )

    spearman_logratio = float(
        pd.Series(
            x_lr
        ).corr(
            pd.Series(
                y_lr
            ),
            method="spearman"
        )
    )


    self_coupling_records.append({
        "gene":
            str(g),

        "pearson_level":
            pearson,

        "spearman_level":
            spearman,

        "pearson_nt_logratio":
            pearson_logratio,

        "spearman_nt_logratio":
            spearman_logratio,

        "self_coefficient":
            float(
                A_hat_integral[
                    gi, rj
                ]
            ),

        "self_selection_frequency":
            float(
                selection_freq[
                    gi, rj
                ]
            ),
    })


self_coupling_df = pd.DataFrame(
    self_coupling_records
)


print(
    "\nSelf-signal coupling:"
)

print(
    self_coupling_df
    .sort_values(
        "spearman_nt_logratio",
        ascending=False
    )
    .to_string(
        index=False
    )
)


print(
    "\nSummary:"
)

print(
    self_coupling_df[
        [
            "pearson_level",
            "spearman_level",
            "pearson_nt_logratio",
            "spearman_nt_logratio",
        ]
    ].describe()
)


# ============================================================================
# NOTEBOOK INDEX 117: Cell 112. Refit the 12 overlap response genes
# ============================================================================

# ============================================
# Cell 112. Refit the 12 overlap response genes
# with self predictor explicitly excluded
#
# Primary exact-integral estimator:
# - same global_alpha_integral
# - same valid rows
# - same exact FWL treatment of dt
# - same RMS scaling
# - ONLY difference:
#     predictor g is removed when response = g
#
# Other 414 response genes remain unchanged.
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


A_hat_integral_noself = (
    A_hat_integral.copy()
)

c_hat_integral_noself = (
    c_hat_integral.copy()
)


noself_refit_records = []


for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]

    self_rj = np.where(
        np.asarray(
            regulator_genes_vc,
            dtype=object
        ) == g
    )[0][0]


    valid_mask = valid_row_masks[g]


    d = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_full = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )


    # ----------------------------------------
    # Explicitly remove same-gene predictor
    # ----------------------------------------

    predictor_keep = np.ones(
        len(regulator_genes_vc),
        dtype=bool
    )

    predictor_keep[
        self_rj
    ] = False


    Xr = Xr_full[
        :,
        predictor_keep
    ]


    # ----------------------------------------
    # Exact FWL projection for unpenalized
    # dt / basal term
    # ----------------------------------------

    dd = d @ d

    coef_d_X = (
        d @ Xr
    ) / dd

    coef_d_y = (
        d @ y
    ) / dd


    Xp = (
        Xr
        - d[:, None]
        * coef_d_X[None, :]
    )

    yp = (
        y
        - d * coef_d_y
    )


    # ----------------------------------------
    # RMS scaling
    # ----------------------------------------

    x_scale = np.sqrt(
        np.mean(
            Xp ** 2,
            axis=0
        )
    )

    y_scale = np.sqrt(
        np.mean(
            yp ** 2
        )
    )


    assert np.all(
        np.isfinite(
            x_scale
        )
    )

    assert np.all(
        x_scale > 0
    )

    assert (
        np.isfinite(y_scale)
        and y_scale > 0
    )


    Z = (
        Xp
        / x_scale[None, :]
    )

    yz = (
        yp
        / y_scale
    )


    model = Lasso(
        alpha=global_alpha_integral,
        fit_intercept=False,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z,
        yz
    )


    A_reduced = (
        y_scale
        * model.coef_
        / x_scale
    )


    # ----------------------------------------
    # Reconstruct full 151-vector,
    # forcing self coefficient exactly zero
    # ----------------------------------------

    A_new = np.zeros(
        len(regulator_genes_vc),
        dtype=np.float64
    )

    A_new[
        predictor_keep
    ] = A_reduced

    A_new[
        self_rj
    ] = 0.0


    # ----------------------------------------
    # Recover unpenalized basal coefficient
    # ----------------------------------------

    c_new = (
        d
        @ (
            y
            - Xr_full @ A_new
        )
    ) / dd


    # ----------------------------------------
    # Diagnostics
    # ----------------------------------------

    pred_new = (
        d * c_new
        + Xr_full @ A_new
    )

    residual_new = (
        y - pred_new
    )

    baseline_c = (
        d @ y
    ) / dd

    baseline_pred = (
        d * baseline_c
    )

    mse_new = float(
        np.mean(
            residual_new ** 2
        )
    )

    mse_baseline = float(
        np.mean(
            (
                y
                - baseline_pred
            ) ** 2
        )
    )

    skill_new = (
        1.0
        - mse_new
        / mse_baseline
    )


    A_old = (
        A_hat_integral[
            gi
        ]
    )

    old_selected = (
        A_old != 0
    )

    new_selected = (
        A_new != 0
    )


    changed_selection = np.sum(
        old_selected
        != new_selected
    )


    # Save corrected row
    A_hat_integral_noself[
        gi
    ] = A_new

    c_hat_integral_noself[
        gi
    ] = c_new


    noself_refit_records.append({
        "gene":
            str(g),

        "old_self_coef":
            float(
                A_old[
                    self_rj
                ]
            ),

        "new_self_coef":
            float(
                A_new[
                    self_rj
                ]
            ),

        "old_n_nonzero":
            int(
                np.count_nonzero(
                    A_old
                )
            ),

        "new_n_nonzero":
            int(
                np.count_nonzero(
                    A_new
                )
            ),

        "changed_selection":
            int(
                changed_selection
            ),

        "old_skill":
            float(
                integral_fit_summary.loc[
                    integral_fit_summary[
                        "gene"
                    ] == g,
                    "skill_vs_intercept"
                ].iloc[0]
            ),

        "new_skill":
            float(
                skill_new
            ),

        "skill_change":
            float(
                skill_new
                -
                integral_fit_summary.loc[
                    integral_fit_summary[
                        "gene"
                    ] == g,
                    "skill_vs_intercept"
                ].iloc[0]
            ),

        "dt_residual_orthogonality":
            float(
                abs(
                    d @ residual_new
                )
            ),
    })


noself_refit_summary = pd.DataFrame(
    noself_refit_records
)


print(
    "Corrected overlap genes:",
    len(
        noself_refit_summary
    )
)


print(
    "\nRefit summary:"
)

print(
    noself_refit_summary
    .sort_values(
        "skill_change"
    )
    .to_string(
        index=False
    )
)


# ============================================
# Global network changes
# ============================================

old_selected_all = (
    A_hat_integral != 0
)

new_selected_all = (
    A_hat_integral_noself != 0
)


print(
    "\nOld total edges:",
    int(
        old_selected_all.sum()
    )
)

print(
    "New total edges:",
    int(
        new_selected_all.sum()
    )
)

print(
    "Net edge-count change:",
    int(
        new_selected_all.sum()
        - old_selected_all.sum()
    )
)


print(
    "\nTotal selection-status changes:",
    int(
        np.sum(
            old_selected_all
            != new_selected_all
        )
    )
)


# ============================================
# Verify only the 12 overlap response rows
# changed
# ============================================

overlap_mask_response = np.isin(
    np.asarray(
        response_genes,
        dtype=object
    ),
    overlap_genes
)

print(
    "\nNon-overlap rows unchanged exactly:",
    np.array_equal(
        A_hat_integral_noself[
            ~overlap_mask_response
        ],
        A_hat_integral[
            ~overlap_mask_response
        ]
    )
)


# ============================================
# Verify all possible self coefficients = 0
# ============================================

self_values_after = []

for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]

    rj = np.where(
        np.asarray(
            regulator_genes_vc,
            dtype=object
        ) == g
    )[0][0]

    self_values_after.append(
        A_hat_integral_noself[
            gi, rj
        ]
    )


self_values_after = np.asarray(
    self_values_after
)


print(
    "\nAll 12 self coefficients exactly zero:",
    bool(
        np.all(
            self_values_after == 0
        )
    )
)

print(
    "Max |self coefficient|:",
    float(
        np.max(
            np.abs(
                self_values_after
            )
        )
    )
)


# ============================================================================
# NOTEBOOK INDEX 118: Cell 113. Update stability for the 12
# ============================================================================

# ============================================
# Cell 113. Update stability for the 12
# self-excluded overlap response genes
#
# Other 414 response rows are copied exactly
# from Cell 104.
#
# For each overlap response g:
# - exclude direct q = g condition via valid mask
# - exclude predictor g itself
# - 80% perturbation-condition subsampling
# - 100 replicates
# - frozen global_alpha_integral
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


# --------------------------------------------
# Start from original full stability results
# --------------------------------------------

selection_count_noself = (
    selection_count.copy()
)

positive_count_noself = (
    positive_count.copy()
)

negative_count_noself = (
    negative_count.copy()
)

edge_count_by_rep_noself = (
    edge_count_by_rep.copy()
)


# Fresh fixed seed for corrected 12-row analysis
rng_stab_noself = np.random.default_rng(
    2027
)


for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]

    self_rj = np.where(
        np.asarray(
            regulator_genes_vc,
            dtype=object
        ) == g
    )[0][0]


    valid_mask = valid_row_masks[g]


    d_all = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_full_all = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y_all = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )

    groups_all = row_conditions[
        valid_mask
    ]


    # ----------------------------------------
    # Explicit self-predictor exclusion
    # ----------------------------------------

    predictor_keep = np.ones(
        len(regulator_genes_vc),
        dtype=bool
    )

    predictor_keep[
        self_rj
    ] = False


    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_all
        )
        if q != "non-targeting"
    ], dtype=object)


    n_keep = int(
        np.floor(
            stability_fraction
            * len(pert_conditions)
        )
    )


    # Reset this response row
    selection_count_noself[
        gi, :
    ] = 0

    positive_count_noself[
        gi, :
    ] = 0

    negative_count_noself[
        gi, :
    ] = 0

    edge_count_by_rep_noself[
        gi, :
    ] = 0


    for rep in range(
        n_stability_reps
    ):

        kept_pert = rng_stab_noself.choice(
            pert_conditions,
            size=n_keep,
            replace=False
        )

        kept_conditions = np.concatenate([
            np.asarray(
                ["non-targeting"],
                dtype=object
            ),
            kept_pert
        ])

        row_mask = np.isin(
            groups_all,
            kept_conditions
        )


        d = d_all[
            row_mask
        ]

        Xr_full = Xr_full_all[
            row_mask
        ]

        Xr = Xr_full[
            :,
            predictor_keep
        ]

        y = y_all[
            row_mask
        ]


        # ------------------------------------
        # Exact FWL projection
        # ------------------------------------

        dd = d @ d

        coef_d_X = (
            d @ Xr
        ) / dd

        coef_d_y = (
            d @ y
        ) / dd


        Xp = (
            Xr
            - d[:, None]
            * coef_d_X[None, :]
        )

        yp = (
            y
            - d * coef_d_y
        )


        # ------------------------------------
        # RMS scaling
        # ------------------------------------

        x_scale = np.sqrt(
            np.mean(
                Xp ** 2,
                axis=0
            )
        )

        y_scale = np.sqrt(
            np.mean(
                yp ** 2
            )
        )


        assert np.all(
            np.isfinite(
                x_scale
            )
        )

        assert np.all(
            x_scale > 0
        )

        assert (
            np.isfinite(y_scale)
            and y_scale > 0
        )


        Z = (
            Xp
            / x_scale[None, :]
        )

        yz = (
            yp
            / y_scale
        )


        model = Lasso(
            alpha=global_alpha_integral,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z,
            yz
        )


        A_reduced = (
            y_scale
            * model.coef_
            / x_scale
        )


        A_rep = np.zeros(
            len(regulator_genes_vc),
            dtype=np.float64
        )

        A_rep[
            predictor_keep
        ] = A_reduced

        A_rep[
            self_rj
        ] = 0.0


        selected = (
            A_rep != 0
        )


        selection_count_noself[
            gi
        ] += selected.astype(
            np.uint16
        )

        positive_count_noself[
            gi
        ] += (
            A_rep > 0
        ).astype(
            np.uint16
        )

        negative_count_noself[
            gi
        ] += (
            A_rep < 0
        ).astype(
            np.uint16
        )


        edge_count_by_rep_noself[
            gi,
            rep
        ] = np.count_nonzero(
            A_rep
        )


    print(
        g,
        "done"
    )


# ============================================
# Updated frequencies
# ============================================

selection_freq_noself = (
    selection_count_noself.astype(
        np.float64
    )
    / n_stability_reps
)

positive_freq_noself = (
    positive_count_noself.astype(
        np.float64
    )
    / n_stability_reps
)

negative_freq_noself = (
    negative_count_noself.astype(
        np.float64
    )
    / n_stability_reps
)

sign_consistency_noself = np.maximum(
    positive_freq_noself,
    negative_freq_noself
)


# ============================================
# Sanity checks
# ============================================

non_overlap_mask = ~np.isin(
    np.asarray(
        response_genes,
        dtype=object
    ),
    overlap_genes
)


print(
    "\nNon-overlap stability rows unchanged:",
    np.array_equal(
        selection_freq_noself[
            non_overlap_mask
        ],
        selection_freq[
            non_overlap_mask
        ]
    )
)


self_freq_after = []

for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]

    rj = np.where(
        np.asarray(
            regulator_genes_vc,
            dtype=object
        ) == g
    )[0][0]

    self_freq_after.append(
        selection_freq_noself[
            gi, rj
        ]
    )


self_freq_after = np.asarray(
    self_freq_after
)


print(
    "All corrected self-edge frequencies zero:",
    bool(
        np.all(
            self_freq_after == 0
        )
    )
)


# ============================================
# Updated primary-network stability summary
# ============================================

full_selected_noself = (
    A_hat_integral_noself != 0
)

stable80_noself = (
    selection_freq_noself >= 0.80
)


print(
    "\nCorrected total full-data edges:",
    int(
        full_selected_noself.sum()
    )
)

print(
    "Corrected stable >=0.8 edges:",
    int(
        stable80_noself.sum()
    )
)

print(
    "Stable >=0.8 among full-selected:",
    int(
        np.sum(
            full_selected_noself
            & stable80_noself
        )
    )
)

print(
    "Fraction of full-selected edges stable >=0.8:",
    float(
        np.mean(
            selection_freq_noself[
                full_selected_noself
            ]
            >= 0.8
        )
    )
)


# ============================================
# Per-overlap-gene comparison
# ============================================

overlap_stability_records = []


for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]


    old_row_selected = (
        A_hat_integral[
            gi
        ] != 0
    )

    new_row_selected = (
        A_hat_integral_noself[
            gi
        ] != 0
    )


    overlap_stability_records.append({
        "gene":
            str(g),

        "old_n_selected":
            int(
                old_row_selected.sum()
            ),

        "new_n_selected":
            int(
                new_row_selected.sum()
            ),

        "old_n_stable80":
            int(
                np.sum(
                    old_row_selected
                    &
                    (
                        selection_freq[
                            gi
                        ] >= 0.8
                    )
                )
            ),

        "new_n_stable80":
            int(
                np.sum(
                    new_row_selected
                    &
                    (
                        selection_freq_noself[
                            gi
                        ] >= 0.8
                    )
                )
            ),

        "new_mean_freq_selected":
            float(
                np.mean(
                    selection_freq_noself[
                        gi,
                        new_row_selected
                    ]
                )
            ),
    })


overlap_stability_summary = pd.DataFrame(
    overlap_stability_records
)


print(
    "\nOverlap-gene stability after "
    "self exclusion:"
)

print(
    overlap_stability_summary
    .to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 119: Cell 114. Freeze corrected primary network
# ============================================================================

# ============================================
# Cell 114. Freeze corrected primary network
# and rebuild the canonical edge table
#
# From this cell onward:
#
# PRIMARY network:
#   A_primary = A_hat_integral_noself
#
# PRIMARY stability:
#   selection_freq_primary
#     = selection_freq_noself
#
# Self predictors are structurally excluded
# for the 12 response/regulator overlap genes.
# ============================================

import numpy as np
import pandas as pd


# ============================================
# Canonical primary objects
# ============================================

A_primary = (
    A_hat_integral_noself.copy()
)

c_primary = (
    c_hat_integral_noself.copy()
)

selection_freq_primary = (
    selection_freq_noself.copy()
)

positive_freq_primary = (
    positive_freq_noself.copy()
)

negative_freq_primary = (
    negative_freq_noself.copy()
)

sign_consistency_primary = (
    sign_consistency_noself.copy()
)


# ============================================
# Rebuild edge table
# ============================================

edge_records_primary = []


response_arr = np.asarray(
    response_genes,
    dtype=object
)

regulator_arr = np.asarray(
    regulator_genes_vc,
    dtype=object
)


for gi, response in enumerate(
    response_arr
):

    for rj, regulator in enumerate(
        regulator_arr
    ):

        coef = float(
            A_primary[
                gi, rj
            ]
        )

        sel_freq = float(
            selection_freq_primary[
                gi, rj
            ]
        )

        pos_freq = float(
            positive_freq_primary[
                gi, rj
            ]
        )

        neg_freq = float(
            negative_freq_primary[
                gi, rj
            ]
        )

        sign_cons = float(
            sign_consistency_primary[
                gi, rj
            ]
        )

        is_self = (
            str(response)
            == str(regulator)
        )

        edge_records_primary.append({
            "response_gene":
                str(response),

            "regulator_gene":
                str(regulator),

            "coefficient":
                coef,

            "abs_coefficient":
                abs(coef),

            "sign":
                int(
                    np.sign(coef)
                ),

            "full_selected":
                bool(
                    coef != 0
                ),

            "selection_frequency":
                sel_freq,

            "positive_frequency":
                pos_freq,

            "negative_frequency":
                neg_freq,

            "sign_consistency":
                sign_cons,

            "stable_80":
                bool(
                    sel_freq >= 0.80
                ),

            "stable_90":
                bool(
                    sel_freq >= 0.90
                ),

            "stable_95":
                bool(
                    sel_freq >= 0.95
                ),

            "stable_100":
                bool(
                    sel_freq == 1.0
                ),

            "self_edge":
                bool(
                    is_self
                ),
        })


edge_table_primary = pd.DataFrame(
    edge_records_primary
)


# ============================================
# Core checks
# ============================================

print(
    "Primary edge table shape:",
    edge_table_primary.shape
)

print(
    "Expected:",
    (
        len(response_genes)
        * len(regulator_genes_vc),
        15
    )
)


print(
    "\nFull-data selected:",
    int(
        edge_table_primary[
            "full_selected"
        ].sum()
    )
)


print(
    "Stable >=0.8:",
    int(
        edge_table_primary[
            "stable_80"
        ].sum()
    )
)

print(
    "Stable >=0.9:",
    int(
        edge_table_primary[
            "stable_90"
        ].sum()
    )
)

print(
    "Stable >=0.95:",
    int(
        edge_table_primary[
            "stable_95"
        ].sum()
    )
)

print(
    "Stable ==1:",
    int(
        edge_table_primary[
            "stable_100"
        ].sum()
    )
)


# ============================================
# Stable edges should be selected in the
# full-data primary fit
# ============================================

stable_not_selected = edge_table_primary[
    edge_table_primary[
        "stable_80"
    ]
    &
    ~edge_table_primary[
        "full_selected"
    ]
]


print(
    "\nStable >=0.8 but not selected "
    "in full-data fit:",
    len(
        stable_not_selected
    )
)


# ============================================
# Verify no self edge survives
# ============================================

self_primary = edge_table_primary[
    edge_table_primary[
        "self_edge"
    ]
]


print(
    "\nPossible self edges:",
    len(
        self_primary
    )
)

print(
    "Selected self edges:",
    int(
        self_primary[
            "full_selected"
        ].sum()
    )
)

print(
    "Stable self edges:",
    int(
        self_primary[
            "stable_80"
        ].sum()
    )
)


# ============================================
# Sign agreement for stable primary edges
# ============================================

stable_primary = edge_table_primary[
    edge_table_primary[
        "stable_80"
    ]
    &
    edge_table_primary[
        "full_selected"
    ]
].copy()


stable_primary[
    "full_sign_frequency"
] = np.where(
    stable_primary[
        "coefficient"
    ].to_numpy()
    > 0,

    stable_primary[
        "positive_frequency"
    ].to_numpy(),

    stable_primary[
        "negative_frequency"
    ].to_numpy()
)


print(
    "\nStable primary sign agreement:"
)

print(
    stable_primary[
        "full_sign_frequency"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


# ============================================
# Final explicit consistency checks
# ============================================

print(
    "\nA_primary equals corrected matrix:",
    np.array_equal(
        A_primary,
        A_hat_integral_noself
    )
)

print(
    "Selection-frequency matrix equals "
    "corrected stability matrix:",
    np.array_equal(
        selection_freq_primary,
        selection_freq_noself
    )
)

print(
    "All primary coefficients finite:",
    bool(
        np.all(
            np.isfinite(
                A_primary
            )
        )
    )
)


# ============================================================================
# NOTEBOOK INDEX 120: Cell 115. Save corrected primary GRN
# ============================================================================

# ============================================
# Cell 115. Save corrected primary GRN
# checkpoint
#
# This is the canonical primary network
# after explicit self-predictor exclusion.
# ============================================

import numpy as np
import pandas as pd
import os


primary_npz_path = (
    "/home/featurize/work/project1/"
    "replogle_primary_exact_integral_grn.npz"
)

primary_edge_csv_path = (
    "/home/featurize/work/project1/"
    "replogle_primary_edge_table.csv.gz"
)


# ============================================
# Save numerical checkpoint
# ============================================

np.savez_compressed(
    primary_npz_path,

    # Network
    A_primary=A_primary,
    c_primary=c_primary,

    # Stability
    selection_freq_primary=
        selection_freq_primary,

    positive_freq_primary=
        positive_freq_primary,

    negative_freq_primary=
        negative_freq_primary,

    sign_consistency_primary=
        sign_consistency_primary,

    # Labels
    response_genes=np.asarray(
        response_genes,
        dtype=str
    ),

    regulator_genes=np.asarray(
        regulator_genes_vc,
        dtype=str
    ),

    conditions=np.asarray(
        final_conditions_vc,
        dtype=str
    ),

    # Estimator metadata
    global_alpha_integral=np.asarray(
        global_alpha_integral
    ),

    omega_nt=np.asarray(
        omega_nt
    ),

    time_scale=np.asarray(
        time_scale
    ),

    # Identifiability summary
    full_design_rank=np.asarray(
        152
    ),

    full_design_lambda_min=np.asarray(
        0.00267429148
    ),

    full_design_condition_number=np.asarray(
        52269.9004
    ),

    # Validation summary
    validation_incremental_r2_mean=
        np.asarray(
            0.111698
        ),

    validation_incremental_r2_median=
        np.asarray(
            0.092643
        ),

    validation_fraction_positive=
        np.asarray(
            0.8936708861
        ),
)


# ============================================
# Save canonical edge table
# ============================================

edge_table_primary.to_csv(
    primary_edge_csv_path,
    index=False,
    compression="gzip"
)


# ============================================
# Reload checks
# ============================================

chk = np.load(
    primary_npz_path,
    allow_pickle=False
)


print(
    "NPZ saved:",
    primary_npz_path
)

print(
    "NPZ size MB:",
    os.path.getsize(
        primary_npz_path
    ) / 1024**2
)

print(
    "\nCSV saved:",
    primary_edge_csv_path
)

print(
    "CSV size MB:",
    os.path.getsize(
        primary_edge_csv_path
    ) / 1024**2
)


print(
    "\nA exact after reload:",
    np.array_equal(
        chk["A_primary"],
        A_primary
    )
)

print(
    "Selection frequency exact:",
    np.array_equal(
        chk[
            "selection_freq_primary"
        ],
        selection_freq_primary
    )
)


edge_reload = pd.read_csv(
    primary_edge_csv_path
)


print(
    "\nEdge-table reload shape:",
    edge_reload.shape
)

print(
    "Full selected after reload:",
    int(
        edge_reload[
            "full_selected"
        ].sum()
    )
)

print(
    "Stable >=0.8 after reload:",
    int(
        edge_reload[
            "stable_80"
        ].sum()
    )
)

print(
    "Selected self edges after reload:",
    int(
        edge_reload.loc[
            edge_reload[
                "self_edge"
            ],
            "full_selected"
        ].sum()
    )
)


# ============================================================================
# NOTEBOOK INDEX 121: Cell 116. Prepare condition-held-out CV
# ============================================================================

# ============================================
# Cell 116. Prepare condition-held-out CV
# and phase-only baseline design
#
# No model fitting yet.
#
# CV rule:
# - perturbation conditions are held out
# - non-targeting is ALWAYS training
# - every perturbation appears in test once
#
# Phase-only baseline:
# - 9 adjacent phase intervals
# - one shared rate parameter per interval
# - prediction on integral scale = dt * rate
# ============================================

import numpy as np
import pandas as pd


# ============================================
# Recover row metadata explicitly
# ============================================

row_conditions_cv = np.asarray(
    [
        row["condition"]
        for row in final_interval_rows
    ],
    dtype=object
)

row_bin_a_cv = np.asarray(
    [
        row["bin_a"]
        for row in final_interval_rows
    ],
    dtype=int
)

row_bin_b_cv = np.asarray(
    [
        row["bin_b"]
        for row in final_interval_rows
    ],
    dtype=int
)

row_dt_cv = np.asarray(
    X_final[:, 0],
    dtype=np.float64
)


# ============================================
# Basic interval checks
# ============================================

print(
    "Rows:",
    len(row_conditions_cv)
)

print(
    "Unique conditions:",
    len(
        np.unique(
            row_conditions_cv
        )
    )
)

print(
    "Unique bin_a:",
    np.unique(
        row_bin_a_cv
    )
)

print(
    "Unique bin_b:",
    np.unique(
        row_bin_b_cv
    )
)

print(
    "All intervals adjacent:",
    bool(
        np.all(
            row_bin_b_cv
            == row_bin_a_cv + 1
        )
    )
)


# ============================================
# Create fixed 5-fold perturbation CV
# ============================================

perturbations_cv = np.asarray(
    final_perturbations_vc,
    dtype=object
)

rng_cv = np.random.default_rng(
    2028
)

perm_cv = rng_cv.permutation(
    perturbations_cv
)

condition_folds_cv = np.array_split(
    perm_cv,
    5
)


print(
    "\nPerturbations per fold:",
    [
        len(x)
        for x in condition_folds_cv
    ]
)


# ============================================
# Verify fold assignment
# ============================================

all_test_conditions = np.concatenate(
    condition_folds_cv
)

print(
    "Every perturbation tested exactly once:",
    bool(
        len(all_test_conditions)
        == len(perturbations_cv)
        and
        len(
            np.unique(
                all_test_conditions
            )
        )
        == len(perturbations_cv)
        and
        set(
            all_test_conditions
        )
        == set(
            perturbations_cv
        )
    )
)

print(
    "Non-targeting ever in test folds:",
    bool(
        np.any(
            all_test_conditions
            == "non-targeting"
        )
    )
)


# ============================================
# Phase-only baseline design
#
# Column k:
#   dt if interval starts at phase bin k
#   0 otherwise
#
# Thus:
#
#   y_hat = dt * phase_rate[k]
#
# This is condition-agnostic but allows an
# arbitrary shared phase profile across the
# 9 adjacent intervals.
# ============================================

n_phase_intervals = 9

X_phase_cv = np.zeros(
    (
        len(final_interval_rows),
        n_phase_intervals
    ),
    dtype=np.float64
)

for k in range(
    n_phase_intervals
):

    X_phase_cv[
        row_bin_a_cv == k,
        k
    ] = row_dt_cv[
        row_bin_a_cv == k
    ]


print(
    "\nPhase baseline design shape:",
    X_phase_cv.shape
)

print(
    "Finite:",
    bool(
        np.all(
            np.isfinite(
                X_phase_cv
            )
        )
    )
)

print(
    "Exactly one active phase column per row:",
    bool(
        np.all(
            np.sum(
                X_phase_cv != 0,
                axis=1
            )
            == 1
        )
    )
)


# ============================================
# Test-row counts by fold
# ============================================

fold_summary_records = []

for fold_idx, test_conditions in enumerate(
    condition_folds_cv
):

    test_mask = np.isin(
        row_conditions_cv,
        test_conditions
    )

    train_mask = ~test_mask

    fold_summary_records.append({
        "fold":
            fold_idx + 1,

        "n_test_conditions":
            len(
                test_conditions
            ),

        "n_train_rows":
            int(
                train_mask.sum()
            ),

        "n_test_rows":
            int(
                test_mask.sum()
            ),

        "nt_rows_in_train":
            int(
                np.sum(
                    train_mask
                    &
                    (
                        row_conditions_cv
                        == "non-targeting"
                    )
                )
            ),

        "nt_rows_in_test":
            int(
                np.sum(
                    test_mask
                    &
                    (
                        row_conditions_cv
                        == "non-targeting"
                    )
                )
            ),
    })


cv_fold_summary = pd.DataFrame(
    fold_summary_records
)

print(
    "\nFold summary:"
)

print(
    cv_fold_summary.to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 122: Cell 117. Exact-integral condition-held-out
# ============================================================================

# ============================================
# Cell 117. Exact-integral condition-held-out
# OOF prediction
#
# Compare:
#   1. exact-integral GRN estimator
#   2. phase-only 9-interval baseline
#
# IMPORTANT:
# - perturbation conditions held out
# - NT always training
# - q = response gene rows excluded
# - self predictor excluded for 12 overlap genes
# - fold-specific FWL + RMS scaling
# - frozen global_alpha_integral
#
# Outputs:
#   oof_y
#   oof_pred_grn
#   oof_pred_phase
#   oof_fold_id
# ============================================

import numpy as np
from sklearn.linear_model import Lasso


n_rows = X_final.shape[0]
n_genes = len(response_genes)
n_regs = len(regulator_genes_vc)


# ============================================
# OOF storage
# ============================================

oof_y = np.full(
    (n_rows, n_genes),
    np.nan,
    dtype=np.float64
)

oof_pred_grn = np.full(
    (n_rows, n_genes),
    np.nan,
    dtype=np.float64
)

oof_pred_phase = np.full(
    (n_rows, n_genes),
    np.nan,
    dtype=np.float64
)

oof_fold_id = np.full(
    n_rows,
    -1,
    dtype=int
)


response_arr = np.asarray(
    response_genes,
    dtype=object
)

regulator_arr = np.asarray(
    regulator_genes_vc,
    dtype=object
)


# ============================================
# Loop over held-out condition folds
# ============================================

for fold_idx, test_conditions in enumerate(
    condition_folds_cv
):

    base_test_mask = np.isin(
        row_conditions_cv,
        test_conditions
    )

    base_train_mask = ~base_test_mask


    oof_fold_id[
        base_test_mask
    ] = fold_idx


    for gi, g in enumerate(
        response_arr
    ):

        # ------------------------------------
        # Direct intervention handling:
        # remove q = g rows from BOTH train
        # and test for response gene g
        # ------------------------------------

        if g in set(
            final_perturbations_vc
        ):

            direct_mask = (
                row_conditions_cv
                == g
            )

        else:

            direct_mask = np.zeros(
                n_rows,
                dtype=bool
            )


        train_mask = (
            base_train_mask
            & ~direct_mask
        )

        test_mask = (
            base_test_mask
            & ~direct_mask
        )


        # If direct condition g belongs to this
        # test fold, those rows intentionally
        # receive no OOF prediction.
        if not np.any(
            test_mask
        ):
            continue


        # ====================================
        # Training data
        # ====================================

        d_train = np.asarray(
            X_final[
                train_mask, 0
            ],
            dtype=np.float64
        )

        Xr_train_full = np.asarray(
            X_final[
                train_mask, 1:
            ],
            dtype=np.float64
        )

        y_train = np.asarray(
            Y_final[
                train_mask, gi
            ],
            dtype=np.float64
        )


        d_test = np.asarray(
            X_final[
                test_mask, 0
            ],
            dtype=np.float64
        )

        Xr_test_full = np.asarray(
            X_final[
                test_mask, 1:
            ],
            dtype=np.float64
        )

        y_test = np.asarray(
            Y_final[
                test_mask, gi
            ],
            dtype=np.float64
        )


        # ====================================
        # Predictor set
        #
        # For overlap genes, remove the
        # same-gene predictor structurally.
        # ====================================

        predictor_keep = np.ones(
            n_regs,
            dtype=bool
        )

        if g in set(
            overlap_genes
        ):

            self_rj = np.where(
                regulator_arr == g
            )[0][0]

            predictor_keep[
                self_rj
            ] = False


        Xr_train = Xr_train_full[
            :,
            predictor_keep
        ]

        Xr_test = Xr_test_full[
            :,
            predictor_keep
        ]


        # ====================================
        # Exact FWL on TRAINING ONLY
        # ====================================

        dd_train = (
            d_train @ d_train
        )

        coef_d_X = (
            d_train @ Xr_train
        ) / dd_train

        coef_d_y = (
            d_train @ y_train
        ) / dd_train


        Xp_train = (
            Xr_train
            - d_train[:, None]
            * coef_d_X[None, :]
        )

        yp_train = (
            y_train
            - d_train
            * coef_d_y
        )


        # ====================================
        # TRAINING-ONLY RMS scaling
        # ====================================

        x_scale = np.sqrt(
            np.mean(
                Xp_train ** 2,
                axis=0
            )
        )

        y_scale = np.sqrt(
            np.mean(
                yp_train ** 2
            )
        )


        assert np.all(
            np.isfinite(
                x_scale
            )
        )

        assert np.all(
            x_scale > 0
        )

        assert (
            np.isfinite(
                y_scale
            )
            and y_scale > 0
        )


        Z_train = (
            Xp_train
            / x_scale[None, :]
        )

        yz_train = (
            yp_train
            / y_scale
        )


        # ====================================
        # Sparse exact-integral fit
        # ====================================

        model = Lasso(
            alpha=global_alpha_integral,
            fit_intercept=False,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z_train,
            yz_train
        )


        A_reduced = (
            y_scale
            * model.coef_
            / x_scale
        )


        # Recover unpenalized basal coefficient
        c_fold = (
            d_train
            @ (
                y_train
                - Xr_train
                @ A_reduced
            )
        ) / dd_train


        pred_grn = (
            d_test * c_fold
            + Xr_test @ A_reduced
        )


        # ====================================
        # Phase-only baseline
        #
        # Fit 9 phase-interval rates on
        # TRAINING ONLY via least squares.
        # ====================================

        P_train = X_phase_cv[
            train_mask
        ]

        P_test = X_phase_cv[
            test_mask
        ]


        phase_coef = np.linalg.lstsq(
            P_train,
            y_train,
            rcond=None
        )[0]


        pred_phase = (
            P_test @ phase_coef
        )


        # ====================================
        # Store OOF predictions
        # ====================================

        oof_y[
            test_mask,
            gi
        ] = y_test

        oof_pred_grn[
            test_mask,
            gi
        ] = pred_grn

        oof_pred_phase[
            test_mask,
            gi
        ] = pred_phase


    print(
        f"Fold {fold_idx + 1}/5 done"
    )


# ============================================
# Basic diagnostics
# ============================================

print(
    "\nAll rows assigned a fold:",
    bool(
        np.all(
            oof_fold_id >= 0
        )
    )
)


valid_oof = np.isfinite(
    oof_y
)

print(
    "Total finite OOF outcomes:",
    int(
        valid_oof.sum()
    )
)

print(
    "Finite GRN predictions:",
    int(
        np.isfinite(
            oof_pred_grn
        ).sum()
    )
)

print(
    "Finite phase predictions:",
    int(
        np.isfinite(
            oof_pred_phase
        ).sum()
    )
)


print(
    "\nPrediction masks identical:",
    bool(
        np.array_equal(
            np.isfinite(
                oof_y
            ),
            np.isfinite(
                oof_pred_grn
            )
        )
        and
        np.array_equal(
            np.isfinite(
                oof_y
            ),
            np.isfinite(
                oof_pred_phase
            )
        )
    )
)


# ============================================
# Expected missing entries:
# only q = g direct-intervention rows for
# the 12 response genes that are themselves
# perturbation targets.
# ============================================

missing_per_gene = np.sum(
    ~np.isfinite(
        oof_y
    ),
    axis=0
)

print(
    "\nGenes with missing OOF rows:",
    int(
        np.sum(
            missing_per_gene > 0
        )
    )
)

print(
    "Total missing gene-row entries:",
    int(
        missing_per_gene.sum()
    )
)


print(
    "\nMissing rows for overlap genes:"
)

for g in overlap_genes:

    gi = np.where(
        response_arr == g
    )[0][0]

    print(
        g,
        int(
            missing_per_gene[
                gi
            ]
        )
    )


# ============================================================================
# NOTEBOOK INDEX 123: Cell 118. Verify OOF missing-entry accounting
# ============================================================================

# ============================================
# Cell 118. Verify OOF missing-entry accounting
#
# Cell 117 intentionally predicts ONLY
# held-out perturbation conditions.
#
# Therefore missing entries should be:
#   1. all NT rows, for all 426 genes
#   2. q = g direct-perturbation rows
#      for the 12 overlap response genes
#
# No model refitting.
# ============================================

import numpy as np
import pandas as pd


# ============================================
# NT rows
# ============================================

nt_row_mask = (
    row_conditions_cv
    == "non-targeting"
)

n_nt_rows = int(
    nt_row_mask.sum()
)

print(
    "NT interval rows:",
    n_nt_rows
)

print(
    "Expected NT missing entries:",
    n_nt_rows * len(response_genes)
)


# ============================================
# Direct q = g rows
# ============================================

direct_missing_records = []

expected_direct_missing = 0


for g in overlap_genes:

    gi = np.where(
        response_arr == g
    )[0][0]

    direct_rows = (
        row_conditions_cv == g
    )

    n_direct_rows = int(
        direct_rows.sum()
    )

    expected_direct_missing += (
        n_direct_rows
    )


    actual_missing_for_gene = (
        ~np.isfinite(
            oof_y[:, gi]
        )
    )


    # Missing rows beyond NT
    actual_direct_only = (
        actual_missing_for_gene
        & ~nt_row_mask
    )


    direct_missing_records.append({
        "gene":
            str(g),

        "n_nt_rows":
            n_nt_rows,

        "n_direct_rows":
            n_direct_rows,

        "expected_total_missing":
            n_nt_rows
            + n_direct_rows,

        "actual_total_missing":
            int(
                actual_missing_for_gene.sum()
            ),

        "actual_nonNT_missing":
            int(
                actual_direct_only.sum()
            ),

        "nonNT_missing_matches_qeqg":
            bool(
                np.array_equal(
                    actual_direct_only,
                    direct_rows
                )
            ),
    })


direct_missing_df = pd.DataFrame(
    direct_missing_records
)


print(
    "\nDirect-intervention missing rows:"
)

print(
    direct_missing_df.to_string(
        index=False
    )
)


# ============================================
# Global expected missing count
# ============================================

expected_total_missing = (
    n_nt_rows
    * len(response_genes)
    + expected_direct_missing
)

actual_total_missing = int(
    np.sum(
        ~np.isfinite(
            oof_y
        )
    )
)


print(
    "\nExpected direct-intervention "
    "missing entries:",
    expected_direct_missing
)

print(
    "Expected TOTAL missing entries:",
    expected_total_missing
)

print(
    "Actual TOTAL missing entries:",
    actual_total_missing
)

print(
    "Exact total match:",
    expected_total_missing
    == actual_total_missing
)


# ============================================
# Check non-overlap genes:
# they should miss ONLY NT rows
# ============================================

non_overlap_genes = [
    g
    for g in response_arr
    if g not in set(overlap_genes)
]


non_overlap_ok = True

for g in non_overlap_genes:

    gi = np.where(
        response_arr == g
    )[0][0]

    missing = (
        ~np.isfinite(
            oof_y[:, gi]
        )
    )

    if not np.array_equal(
        missing,
        nt_row_mask
    ):
        non_overlap_ok = False
        break


print(
    "\nAll 414 non-overlap genes "
    "miss ONLY NT rows:",
    non_overlap_ok
)


# ============================================
# Fold-ID interpretation
# ============================================

print(
    "\nRows with fold_id = -1:",
    int(
        np.sum(
            oof_fold_id == -1
        )
    )
)

print(
    "Those rows are exactly NT rows:",
    bool(
        np.array_equal(
            oof_fold_id == -1,
            nt_row_mask
        )
    )
)


# ============================================================================
# NOTEBOOK INDEX 124: Cell 119. Primary held-out validation
# ============================================================================

# ============================================
# Cell 119. Primary held-out validation
#
# Exact-integral GRN
#       vs
# strong phase-only baseline
#
# Metric:
#
# incremental R^2 =
#   1 - SSE_GRN / SSE_PHASE
#
# > 0  : GRN improves over phase-only
# = 0  : equal
# < 0  : GRN worse than phase-only
#
# IMPORTANT:
# evaluate non-calibration genes separately.
# ============================================

import numpy as np
import pandas as pd


validation_records = []


for gi, g in enumerate(
    response_arr
):

    valid = (
        np.isfinite(
            oof_y[:, gi]
        )
        &
        np.isfinite(
            oof_pred_grn[:, gi]
        )
        &
        np.isfinite(
            oof_pred_phase[:, gi]
        )
    )


    y = oof_y[
        valid, gi
    ]

    pred_grn = oof_pred_grn[
        valid, gi
    ]

    pred_phase = oof_pred_phase[
        valid, gi
    ]


    sse_grn = float(
        np.sum(
            (
                y - pred_grn
            ) ** 2
        )
    )

    sse_phase = float(
        np.sum(
            (
                y - pred_phase
            ) ** 2
        )
    )


    incremental_r2 = (
        1.0
        - sse_grn / sse_phase
        if sse_phase > 0
        else np.nan
    )


    # Conventional OOF R2 against global
    # test-outcome mean, for context only.
    sst = float(
        np.sum(
            (
                y - np.mean(y)
            ) ** 2
        )
    )


    r2_grn = (
        1.0
        - sse_grn / sst
        if sst > 0
        else np.nan
    )

    r2_phase = (
        1.0
        - sse_phase / sst
        if sst > 0
        else np.nan
    )


    validation_records.append({
        "gene":
            str(g),

        "n_oof":
            int(
                valid.sum()
            ),

        "sse_grn":
            sse_grn,

        "sse_phase":
            sse_phase,

        "incremental_r2_vs_phase":
            float(
                incremental_r2
            ),

        "r2_grn":
            float(
                r2_grn
            ),

        "r2_phase":
            float(
                r2_phase
            ),

        "grn_better_than_phase":
            bool(
                sse_grn < sse_phase
            ),
    })


exact_oof_validation = pd.DataFrame(
    validation_records
)


# ============================================
# Calibration-panel membership
# ============================================

calibration_gene_set = set(
    calibration_genes
)

exact_oof_validation[
    "is_calibration_gene"
] = exact_oof_validation[
    "gene"
].isin(
    calibration_gene_set
)


primary_validation = (
    exact_oof_validation[
        ~exact_oof_validation[
            "is_calibration_gene"
        ]
    ]
    .copy()
)

calibration_validation = (
    exact_oof_validation[
        exact_oof_validation[
            "is_calibration_gene"
        ]
    ]
    .copy()
)


# ============================================
# Summary helper
# ============================================

def print_validation_summary(
    df,
    label
):

    x = df[
        "incremental_r2_vs_phase"
    ].to_numpy(
        dtype=float
    )

    print(
        f"\n{label}"
    )

    print(
        "n genes:",
        len(df)
    )

    print(
        "incremental R2 mean:",
        float(
            np.nanmean(x)
        )
    )

    print(
        "incremental R2 median:",
        float(
            np.nanmedian(x)
        )
    )

    print(
        "fraction > 0:",
        float(
            np.nanmean(
                x > 0
            )
        )
    )

    print(
        "fraction >= 0.05:",
        float(
            np.nanmean(
                x >= 0.05
            )
        )
    )

    print(
        "fraction >= 0.10:",
        float(
            np.nanmean(
                x >= 0.10
            )
        )
    )

    print(
        "\nIncremental R2 distribution:"
    )

    print(
        pd.Series(
            x
        ).describe(
            percentiles=[
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ]
        )
    )

    print(
        "\nMean conventional OOF R2:"
    )

    print(
        "GRN:",
        float(
            df[
                "r2_grn"
            ].mean()
        )
    )

    print(
        "Phase:",
        float(
            df[
                "r2_phase"
            ].mean()
        )
    )


# ============================================
# PRIMARY:
# genes not used for alpha calibration
# ============================================

print_validation_summary(
    primary_validation,
    "PRIMARY NON-CALIBRATION GENES"
)


# ============================================
# Calibration panel
# ============================================

print_validation_summary(
    calibration_validation,
    "CALIBRATION PANEL"
)


# ============================================
# All genes, descriptive only
# ============================================

print_validation_summary(
    exact_oof_validation,
    "ALL GENES"
)


# ============================================
# Best / worst non-calibration genes
# ============================================

print(
    "\nTop 15 non-calibration genes:"
)

print(
    primary_validation
    .sort_values(
        "incremental_r2_vs_phase",
        ascending=False
    )
    .head(15)
    [
        [
            "gene",
            "n_oof",
            "incremental_r2_vs_phase",
            "r2_grn",
            "r2_phase",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nBottom 15 non-calibration genes:"
)

print(
    primary_validation
    .sort_values(
        "incremental_r2_vs_phase",
        ascending=True
    )
    .head(15)
    [
        [
            "gene",
            "n_oof",
            "incremental_r2_vs_phase",
            "r2_grn",
            "r2_phase",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================================
# NOTEBOOK INDEX 125: Cell 120. Diagnose the negative held-out result
# ============================================================================

# ============================================
# Cell 120. Diagnose the negative held-out result
#
# Questions:
# 1. Does the negative mean incremental R2 come
#    mainly from genes where phase baseline SSE
#    is already extremely small?
#
# 2. What is the pooled held-out performance?
#
# 3. How does GRN improvement depend on
#    phase-only predictive strength?
#
# No refitting. No tuning.
# ============================================

import numpy as np
import pandas as pd


df = primary_validation.copy()


# ============================================
# 1. Pooled SSE comparison
# ============================================

pooled_sse_grn = float(
    df["sse_grn"].sum()
)

pooled_sse_phase = float(
    df["sse_phase"].sum()
)

pooled_incremental_r2 = (
    1.0
    - pooled_sse_grn
    / pooled_sse_phase
)


print(
    "PRIMARY NON-CALIBRATION GENES"
)

print(
    "\nPooled SSE GRN:",
    pooled_sse_grn
)

print(
    "Pooled SSE phase:",
    pooled_sse_phase
)

print(
    "Pooled incremental R2 vs phase:",
    pooled_incremental_r2
)


# ============================================
# 2. Relationship with phase-baseline strength
# ============================================

print(
    "\nSpearman correlations:"
)

corr_cols = [
    "incremental_r2_vs_phase",
    "r2_phase",
    "r2_grn",
    "sse_phase",
]

print(
    df[
        corr_cols
    ]
    .corr(
        method="spearman"
    )
)


# ============================================
# 3. Phase-R2 quintile stratification
# ============================================

df["phase_r2_quintile"] = pd.qcut(
    df["r2_phase"],
    q=5,
    duplicates="drop"
)


phase_strata = (
    df
    .groupby(
        "phase_r2_quintile",
        observed=True
    )
    .agg(
        n_genes=(
            "gene",
            "size"
        ),

        phase_r2_mean=(
            "r2_phase",
            "mean"
        ),

        phase_r2_median=(
            "r2_phase",
            "median"
        ),

        grn_r2_mean=(
            "r2_grn",
            "mean"
        ),

        incremental_mean=(
            "incremental_r2_vs_phase",
            "mean"
        ),

        incremental_median=(
            "incremental_r2_vs_phase",
            "median"
        ),

        fraction_grn_better=(
            "grn_better_than_phase",
            "mean"
        ),

        sse_grn_sum=(
            "sse_grn",
            "sum"
        ),

        sse_phase_sum=(
            "sse_phase",
            "sum"
        ),
    )
    .reset_index()
)


phase_strata[
    "pooled_incremental_r2"
] = (
    1.0
    - phase_strata[
        "sse_grn_sum"
    ]
    / phase_strata[
        "sse_phase_sum"
    ]
)


print(
    "\nPerformance stratified by "
    "phase-only R2 quintile:"
)

print(
    phase_strata.to_string(
        index=False
    )
)


# ============================================
# 4. Distribution after excluding only the
#    most phase-predictable quintile
#
# DIAGNOSTIC ONLY.
# This is NOT a new primary result.
# ============================================

phase_cut = df[
    "r2_phase"
].quantile(
    0.80
)

lower_phase_80 = df[
    df[
        "r2_phase"
    ] <= phase_cut
]


print(
    "\nDiagnostic: genes below top 20% "
    "phase predictability"
)

print(
    "n genes:",
    len(
        lower_phase_80
    )
)

print(
    "phase R2 cutoff:",
    float(
        phase_cut
    )
)

print(
    "incremental R2 mean:",
    float(
        lower_phase_80[
            "incremental_r2_vs_phase"
        ].mean()
    )
)

print(
    "incremental R2 median:",
    float(
        lower_phase_80[
            "incremental_r2_vs_phase"
        ].median()
    )
)

print(
    "fraction > 0:",
    float(
        lower_phase_80[
            "grn_better_than_phase"
        ].mean()
    )
)


# ============================================
# 5. Genes with phase baseline already
#    explaining >= 95% of OOF variance
# ============================================

very_phase = df[
    df[
        "r2_phase"
    ] >= 0.95
]


print(
    "\nGenes with phase-only OOF R2 >= 0.95:"
)

print(
    "n genes:",
    len(
        very_phase
    )
)

if len(
    very_phase
) > 0:

    print(
        "incremental R2 mean:",
        float(
            very_phase[
                "incremental_r2_vs_phase"
            ].mean()
        )
    )

    print(
        "incremental R2 median:",
        float(
            very_phase[
                "incremental_r2_vs_phase"
            ].median()
        )
    )

    print(
        "fraction GRN better:",
        float(
            very_phase[
                "grn_better_than_phase"
            ].mean()
        )
    )


# ============================================================================
# NOTEBOOK INDEX 126: Cell 121. Diagnose pooled SSE improvement
# ============================================================================

# ============================================
# Cell 121. Diagnose pooled SSE improvement
#
# Question:
# Is pooled +16.4% improvement broad,
# or dominated by a small number of genes
# with very large phase-baseline SSE?
#
# No refitting.
# ============================================

import numpy as np
import pandas as pd


df = primary_validation.copy()


# ============================================
# Per-gene absolute SSE improvement
#
# Positive = GRN better
# Negative = phase baseline better
# ============================================

df[
    "absolute_sse_improvement"
] = (
    df["sse_phase"]
    - df["sse_grn"]
)

df[
    "phase_sse_weight"
] = (
    df["sse_phase"]
    / df["sse_phase"].sum()
)


# ============================================
# Overall accounting
# ============================================

positive_improvement = df[
    df[
        "absolute_sse_improvement"
    ] > 0
]

negative_improvement = df[
    df[
        "absolute_sse_improvement"
    ] < 0
]


total_gain = float(
    positive_improvement[
        "absolute_sse_improvement"
    ].sum()
)

total_loss = float(
    -negative_improvement[
        "absolute_sse_improvement"
    ].sum()
)

net_gain = float(
    df[
        "absolute_sse_improvement"
    ].sum()
)


print(
    "Genes with absolute SSE improvement:",
    len(
        positive_improvement
    )
)

print(
    "Genes with absolute SSE worsening:",
    len(
        negative_improvement
    )
)

print(
    "\nTotal SSE gain from improved genes:",
    total_gain
)

print(
    "Total SSE loss from worsened genes:",
    total_loss
)

print(
    "Net SSE gain:",
    net_gain
)

print(
    "Gain / loss ratio:",
    total_gain / total_loss
)


# ============================================
# How concentrated is phase SSE?
# ============================================

df_by_weight = df.sort_values(
    "sse_phase",
    ascending=False
).copy()

df_by_weight[
    "cumulative_phase_sse_fraction"
] = (
    df_by_weight[
        "sse_phase"
    ].cumsum()
    / df_by_weight[
        "sse_phase"
    ].sum()
)


for n in [
    5,
    10,
    20,
    50,
    100,
]:

    top = df_by_weight.head(
        n
    )

    print(
        f"\nTop {n} genes by phase SSE:"
    )

    print(
        "fraction of total phase SSE:",
        float(
            top[
                "sse_phase"
            ].sum()
            / df[
                "sse_phase"
            ].sum()
        )
    )

    print(
        "fraction of net SSE gain:",
        float(
            top[
                "absolute_sse_improvement"
            ].sum()
            / net_gain
        )
    )


# ============================================
# Top contributors to pooled improvement
# ============================================

print(
    "\nTop 20 contributors to pooled "
    "GRN improvement:"
)

print(
    df
    .sort_values(
        "absolute_sse_improvement",
        ascending=False
    )
    .head(20)
    [
        [
            "gene",
            "sse_phase",
            "sse_grn",
            "absolute_sse_improvement",
            "incremental_r2_vs_phase",
            "r2_phase",
            "r2_grn",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================
# Top contributors against GRN
# ============================================

print(
    "\nTop 20 contributors favoring "
    "phase-only:"
)

print(
    df
    .sort_values(
        "absolute_sse_improvement",
        ascending=True
    )
    .head(20)
    [
        [
            "gene",
            "sse_phase",
            "sse_grn",
            "absolute_sse_improvement",
            "incremental_r2_vs_phase",
            "r2_phase",
            "r2_grn",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================
# Leave-top-K-out pooled sensitivity
#
# Remove genes with largest phase SSE and
# recompute pooled incremental R2.
#
# Diagnostic only.
# ============================================

print(
    "\nLeave-high-SSE-genes-out "
    "pooled incremental R2:"
)


for k in [
    0,
    1,
    5,
    10,
    20,
    50,
    100,
]:

    remaining = (
        df_by_weight.iloc[
            k:
        ]
    )

    pooled = (
        1.0
        - remaining[
            "sse_grn"
        ].sum()
        / remaining[
            "sse_phase"
        ].sum()
    )

    print(
        f"remove top {k:3d}: "
        f"{pooled:.6f}"
    )


# ============================================================================
# NOTEBOOK INDEX 127: Cell 122. Held-out regulator feature
# ============================================================================

# ============================================
# Cell 122. Held-out regulator feature
# extrapolation diagnostic
#
# Question:
# Are test perturbation intervals outside the
# regulator-feature distribution represented
# in the training conditions?
#
# No model fitting.
# ============================================

import numpy as np
import pandas as pd


Xr_all = np.asarray(
    X_final[:, 1:],
    dtype=np.float64
)


fold_shift_records = []


for fold_idx, test_conditions in enumerate(
    condition_folds_cv
):

    test_mask = np.isin(
        row_conditions_cv,
        test_conditions
    )

    train_mask = ~test_mask


    X_train = Xr_all[
        train_mask
    ]

    X_test = Xr_all[
        test_mask
    ]


    # ========================================
    # Training-only standardization
    # ========================================

    train_mean = np.mean(
        X_train,
        axis=0
    )

    train_std = np.std(
        X_train,
        axis=0,
        ddof=0
    )

    assert np.all(
        train_std > 0
    )


    Z_train = (
        X_train
        - train_mean[None, :]
    ) / train_std[None, :]

    Z_test = (
        X_test
        - train_mean[None, :]
    ) / train_std[None, :]


    # ========================================
    # Per-row standardized distance from
    # training centroid
    #
    # RMS z-score across 151 regulators.
    # ========================================

    train_rms_distance = np.sqrt(
        np.mean(
            Z_train ** 2,
            axis=1
        )
    )

    test_rms_distance = np.sqrt(
        np.mean(
            Z_test ** 2,
            axis=1
        )
    )


    # ========================================
    # Coordinate-wise range extrapolation
    #
    # Fraction of regulator coordinates in
    # each test row lying outside the
    # training min/max.
    # ========================================

    train_min = np.min(
        X_train,
        axis=0
    )

    train_max = np.max(
        X_train,
        axis=0
    )


    outside = (
        (X_test < train_min[None, :])
        |
        (X_test > train_max[None, :])
    )

    outside_fraction_per_row = np.mean(
        outside,
        axis=1
    )


    # ========================================
    # Extreme standardized coordinates
    # ========================================

    abs_Z_test = np.abs(
        Z_test
    )

    frac_abs_z_gt_2 = np.mean(
        abs_Z_test > 2,
        axis=1
    )

    frac_abs_z_gt_3 = np.mean(
        abs_Z_test > 3,
        axis=1
    )


    fold_shift_records.append({
        "fold":
            fold_idx + 1,

        "n_train_rows":
            int(
                train_mask.sum()
            ),

        "n_test_rows":
            int(
                test_mask.sum()
            ),

        "train_rms_distance_median":
            float(
                np.median(
                    train_rms_distance
                )
            ),

        "test_rms_distance_median":
            float(
                np.median(
                    test_rms_distance
                )
            ),

        "test_rms_distance_p95":
            float(
                np.quantile(
                    test_rms_distance,
                    0.95
                )
            ),

        "test_outside_range_mean":
            float(
                np.mean(
                    outside_fraction_per_row
                )
            ),

        "test_outside_range_max":
            float(
                np.max(
                    outside_fraction_per_row
                )
            ),

        "test_frac_abs_z_gt_2_mean":
            float(
                np.mean(
                    frac_abs_z_gt_2
                )
            ),

        "test_frac_abs_z_gt_3_mean":
            float(
                np.mean(
                    frac_abs_z_gt_3
                )
            ),
    })


feature_shift_summary = pd.DataFrame(
    fold_shift_records
)


print(
    "Held-out regulator-feature shift:"
)

print(
    feature_shift_summary.to_string(
        index=False
    )
)


# ============================================
# Aggregate summaries
# ============================================

print(
    "\nMedian test/train RMS-distance ratio:"
)

print(
    float(
        np.median(
            feature_shift_summary[
                "test_rms_distance_median"
            ]
            /
            feature_shift_summary[
                "train_rms_distance_median"
            ]
        )
    )
)


print(
    "\nMean coordinate-wise outside-range "
    "fraction:"
)

print(
    float(
        feature_shift_summary[
            "test_outside_range_mean"
        ].mean()
    )
)


print(
    "\nMean fraction |z| > 2:"
)

print(
    float(
        feature_shift_summary[
            "test_frac_abs_z_gt_2_mean"
        ].mean()
    )
)


print(
    "Mean fraction |z| > 3:"
)

print(
    float(
        feature_shift_summary[
            "test_frac_abs_z_gt_3_mean"
        ].mean()
    )
)


# ============================================================================
# NOTEBOOK INDEX 128: Cell 123. Decompose regulator trajectories
# ============================================================================

# ============================================
# Cell 123. Decompose regulator trajectories
# into:
#
#   shared NT phase component
#   +
#   condition-specific deviation
#
# Goal:
# determine what information the current
# GRN predictor is actually using.
#
# No fitting.
# ============================================

import numpy as np
import pandas as pd


# ============================================
# Locate NT condition
# ============================================

condition_arr = np.asarray(
    final_conditions_vc,
    dtype=object
)

nt_ci = np.where(
    condition_arr
    == "non-targeting"
)[0][0]


# ============================================
# NT regulator phase trajectory
#
# shape:
#   10 bins x 151 regulators
# ============================================

S_reg_nt = np.asarray(
    S_regulator_hat[
        nt_ci
    ],
    dtype=np.float64
)


print(
    "NT regulator trajectory shape:",
    S_reg_nt.shape
)

print(
    "Finite:",
    bool(
        np.all(
            np.isfinite(
                S_reg_nt
            )
        )
    )
)


# ============================================
# Build interval-level shared-phase features
#
# Use exactly the same trapezoidal integral
# and condition-specific dt as X_final.
#
# X_shared:
#   what X would be if every condition had
#   the NT regulator phase trajectory.
#
# X_deviation:
#   actual X - shared component.
# ============================================

n_intervals = len(
    final_interval_rows
)

n_regs = len(
    regulator_genes_vc
)


X_shared = np.zeros(
    (
        n_intervals,
        n_regs
    ),
    dtype=np.float64
)


for i, row in enumerate(
    final_interval_rows
):

    a = int(
        row["bin_a"]
    )

    b = int(
        row["bin_b"]
    )

    dt = float(
        X_final[
            i, 0
        ]
    )


    X_shared[
        i
    ] = (
        0.5
        * dt
        * (
            S_reg_nt[a]
            + S_reg_nt[b]
        )
    )


X_actual = np.asarray(
    X_final[:, 1:],
    dtype=np.float64
)

X_deviation = (
    X_actual
    - X_shared
)


# ============================================
# Sanity checks
# ============================================

print(
    "\nShapes:"
)

print(
    "actual:",
    X_actual.shape
)

print(
    "shared:",
    X_shared.shape
)

print(
    "deviation:",
    X_deviation.shape
)


print(
    "\nExact decomposition:",
    bool(
        np.allclose(
            X_actual,
            X_shared
            + X_deviation,
            rtol=1e-12,
            atol=1e-12
        )
    )
)


# ============================================
# Relative magnitude
# ============================================

actual_rms = np.sqrt(
    np.mean(
        X_actual ** 2
    )
)

shared_rms = np.sqrt(
    np.mean(
        X_shared ** 2
    )
)

deviation_rms = np.sqrt(
    np.mean(
        X_deviation ** 2
    )
)


print(
    "\nGlobal RMS:"
)

print(
    "actual:",
    float(
        actual_rms
    )
)

print(
    "shared phase:",
    float(
        shared_rms
    )
)

print(
    "condition deviation:",
    float(
        deviation_rms
    )
)

print(
    "deviation / actual:",
    float(
        deviation_rms
        / actual_rms
    )
)


# ============================================
# Per-regulator decomposition
# ============================================

per_reg_actual_rms = np.sqrt(
    np.mean(
        X_actual ** 2,
        axis=0
    )
)

per_reg_deviation_rms = np.sqrt(
    np.mean(
        X_deviation ** 2,
        axis=0
    )
)

per_reg_deviation_fraction = (
    per_reg_deviation_rms
    / per_reg_actual_rms
)


regulator_decomposition = pd.DataFrame({
    "regulator_gene":
        np.asarray(
            regulator_genes_vc,
            dtype=str
        ),

    "actual_rms":
        per_reg_actual_rms,

    "deviation_rms":
        per_reg_deviation_rms,

    "deviation_fraction":
        per_reg_deviation_fraction,
})


print(
    "\nPer-regulator deviation fraction:"
)

print(
    regulator_decomposition[
        "deviation_fraction"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


print(
    "\nLargest condition-specific "
    "deviation fractions:"
)

print(
    regulator_decomposition
    .sort_values(
        "deviation_fraction",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================
# Condition-level deviation magnitude
# ============================================

condition_deviation_records = []


for q in final_conditions_vc:

    mask = (
        row_conditions_cv == q
    )

    if not np.any(
        mask
    ):
        continue


    q_actual_rms = np.sqrt(
        np.mean(
            X_actual[
                mask
            ] ** 2
        )
    )

    q_dev_rms = np.sqrt(
        np.mean(
            X_deviation[
                mask
            ] ** 2
        )
    )


    condition_deviation_records.append({
        "condition":
            str(q),

        "n_intervals":
            int(
                mask.sum()
            ),

        "actual_rms":
            float(
                q_actual_rms
            ),

        "deviation_rms":
            float(
                q_dev_rms
            ),

        "deviation_fraction":
            float(
                q_dev_rms
                / q_actual_rms
            ),
    })


condition_decomposition = pd.DataFrame(
    condition_deviation_records
)


print(
    "\nCondition-level deviation fraction:"
)

print(
    condition_decomposition[
        "deviation_fraction"
    ].describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


# ============================================================================
# NOTEBOOK INDEX 129: Cell 124. Freeze final numerical-experiment
# ============================================================================

# ============================================
# Cell 124. Freeze final numerical-experiment
# summary for manuscript
#
# NO new analysis.
# NO model fitting.
#
# This cell only collects the final results
# already established above.
# ============================================

import numpy as np
import pandas as pd
import os


# ============================================
# Final manuscript-level summary
# ============================================

summary_records = [
    # ----------------------------------------
    # Dataset / design
    # ----------------------------------------
    {
        "section": "dataset",
        "metric": "response_genes",
        "value": len(response_genes),
    },
    {
        "section": "dataset",
        "metric": "regulator_genes",
        "value": len(regulator_genes_vc),
    },
    {
        "section": "dataset",
        "metric": "perturbations",
        "value": len(final_perturbations_vc),
    },
    {
        "section": "dataset",
        "metric": "conditions_including_NT",
        "value": len(final_conditions_vc),
    },
    {
        "section": "dataset",
        "metric": "integral_intervals",
        "value": X_final.shape[0],
    },

    # ----------------------------------------
    # Empirical identifiability
    # ----------------------------------------
    {
        "section": "identifiability",
        "metric": "full_design_rank",
        "value": 152,
    },
    {
        "section": "identifiability",
        "metric": "full_design_columns",
        "value": 152,
    },
    {
        "section": "identifiability",
        "metric": "scaled_information_lambda_min",
        "value": 0.00267429148,
    },
    {
        "section": "identifiability",
        "metric": "scaled_information_condition_number",
        "value": 52269.9004,
    },
    {
        "section": "identifiability",
        "metric": "median_perturbations_to_full_rank",
        "value": 42,
    },

    # ----------------------------------------
    # Primary exact-integral GRN
    # ----------------------------------------
    {
        "section": "network",
        "metric": "candidate_edges",
        "value": A_primary.size,
    },
    {
        "section": "network",
        "metric": "selected_edges",
        "value": int(
            np.count_nonzero(
                A_primary
            )
        ),
    },
    {
        "section": "network",
        "metric": "selected_edge_density",
        "value": float(
            np.count_nonzero(
                A_primary
            )
            / A_primary.size
        ),
    },

    # ----------------------------------------
    # Stability
    # ----------------------------------------
    {
        "section": "stability",
        "metric": "stable_edges_freq_ge_0.80",
        "value": int(
            np.sum(
                selection_freq_primary
                >= 0.80
            )
        ),
    },
    {
        "section": "stability",
        "metric": "stable_edges_freq_ge_0.90",
        "value": int(
            np.sum(
                selection_freq_primary
                >= 0.90
            )
        ),
    },
    {
        "section": "stability",
        "metric": "stable_edges_freq_ge_0.95",
        "value": int(
            np.sum(
                selection_freq_primary
                >= 0.95
            )
        ),
    },
    {
        "section": "stability",
        "metric": "stable_edges_freq_eq_1",
        "value": int(
            np.sum(
                selection_freq_primary
                == 1.0
            )
        ),
    },
    {
        "section": "stability",
        "metric": "fraction_selected_edges_stable_ge_0.80",
        "value": float(
            np.mean(
                selection_freq_primary[
                    A_primary != 0
                ]
                >= 0.80
            )
        ),
    },
    {
        "section": "stability",
        "metric": "stable_edge_sign_agreement_mean",
        "value": float(
            stable_primary[
                "full_sign_frequency"
            ].mean()
        ),
    },
    {
        "section": "stability",
        "metric": "stable_edge_sign_agreement_median",
        "value": float(
            stable_primary[
                "full_sign_frequency"
            ].median()
        ),
    },

    # ----------------------------------------
    # Held-out perturbation validation
    # Primary = 395 non-calibration genes
    # ----------------------------------------
    {
        "section": "validation",
        "metric": "noncalibration_genes",
        "value": len(
            primary_validation
        ),
    },
    {
        "section": "validation",
        "metric": "gene_wise_incremental_R2_mean",
        "value": float(
            primary_validation[
                "incremental_r2_vs_phase"
            ].mean()
        ),
    },
    {
        "section": "validation",
        "metric": "gene_wise_incremental_R2_median",
        "value": float(
            primary_validation[
                "incremental_r2_vs_phase"
            ].median()
        ),
    },
    {
        "section": "validation",
        "metric": "fraction_genes_GRN_better_than_phase",
        "value": float(
            primary_validation[
                "grn_better_than_phase"
            ].mean()
        ),
    },
    {
        "section": "validation",
        "metric": "mean_OOF_R2_GRN",
        "value": float(
            primary_validation[
                "r2_grn"
            ].mean()
        ),
    },
    {
        "section": "validation",
        "metric": "mean_OOF_R2_phase",
        "value": float(
            primary_validation[
                "r2_phase"
            ].mean()
        ),
    },
    {
        "section": "validation",
        "metric": "pooled_incremental_R2",
        "value": float(
            1.0
            - primary_validation[
                "sse_grn"
            ].sum()
            / primary_validation[
                "sse_phase"
            ].sum()
        ),
    },

    # ----------------------------------------
    # Phase-dependence diagnostic
    # ----------------------------------------
    {
        "section": "phase_diagnostic",
        "metric": "spearman_incrementalR2_vs_phaseR2",
        "value": float(
            primary_validation[
                [
                    "incremental_r2_vs_phase",
                    "r2_phase",
                ]
            ]
            .corr(
                method="spearman"
            )
            .iloc[0, 1]
        ),
    },
    {
        "section": "phase_diagnostic",
        "metric": "genes_phase_R2_ge_0.95",
        "value": int(
            np.sum(
                primary_validation[
                    "r2_phase"
                ]
                >= 0.95
            )
        ),
    },
    {
        "section": "phase_diagnostic",
        "metric": "fraction_GRN_better_when_phase_R2_ge_0.95",
        "value": float(
            primary_validation.loc[
                primary_validation[
                    "r2_phase"
                ] >= 0.95,
                "grn_better_than_phase"
            ].mean()
        ),
    },

    # ----------------------------------------
    # Regulator-design decomposition
    # ----------------------------------------
    {
        "section": "design_decomposition",
        "metric": "actual_regulator_design_RMS",
        "value": float(
            np.sqrt(
                np.mean(
                    X_actual ** 2
                )
            )
        ),
    },
    {
        "section": "design_decomposition",
        "metric": "shared_phase_design_RMS",
        "value": float(
            np.sqrt(
                np.mean(
                    X_shared ** 2
                )
            )
        ),
    },
    {
        "section": "design_decomposition",
        "metric": "condition_deviation_RMS",
        "value": float(
            np.sqrt(
                np.mean(
                    X_deviation ** 2
                )
            )
        ),
    },
    {
        "section": "design_decomposition",
        "metric": "condition_deviation_over_actual_RMS",
        "value": float(
            np.sqrt(
                np.mean(
                    X_deviation ** 2
                )
            )
            /
            np.sqrt(
                np.mean(
                    X_actual ** 2
                )
            )
        ),
    },
    {
        "section": "design_decomposition",
        "metric": "median_condition_deviation_fraction",
        "value": float(
            condition_decomposition[
                "deviation_fraction"
            ].median()
        ),
    },
]


numerical_experiment_summary = pd.DataFrame(
    summary_records
)


# ============================================
# Save
# ============================================

summary_path = (
    "/home/featurize/work/project1/"
    "replogle_numerical_experiment_summary.csv"
)

numerical_experiment_summary.to_csv(
    summary_path,
    index=False
)


# ============================================
# Display
# ============================================

print(
    numerical_experiment_summary.to_string(
        index=False
    )
)


print(
    "\nSaved:",
    summary_path
)

print(
    "File exists:",
    os.path.exists(
        summary_path
    )
)

print(
    "Rows:",
    len(
        numerical_experiment_summary
    )
)


# ============================================
# Final headline numbers
# ============================================

print(
    "\n"
    "========================================\n"
    "MANUSCRIPT HEADLINE NUMBERS\n"
    "========================================"
)

print(
    "Design rank:",
    "152 / 152"
)

print(
    "Median perturbations to full rank:",
    42
)

print(
    "Selected edges:",
    int(
        np.count_nonzero(
            A_primary
        )
    )
)

print(
    "Stable edges (freq >= 0.80):",
    int(
        np.sum(
            selection_freq_primary
            >= 0.80
        )
    )
)

print(
    "Fraction selected edges stable:",
    float(
        np.mean(
            selection_freq_primary[
                A_primary != 0
            ]
            >= 0.80
        )
    )
)

print(
    "Held-out gene-wise median incremental R2:",
    float(
        primary_validation[
            "incremental_r2_vs_phase"
        ].median()
    )
)

print(
    "Held-out fraction genes GRN > phase:",
    float(
        primary_validation[
            "grn_better_than_phase"
        ].mean()
    )
)

print(
    "Held-out pooled incremental R2:",
    float(
        1
        - primary_validation[
            "sse_grn"
        ].sum()
        / primary_validation[
            "sse_phase"
        ].sum()
    )
)

print(
    "Spearman incremental R2 vs phase R2:",
    float(
        primary_validation[
            [
                "incremental_r2_vs_phase",
                "r2_phase",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[0, 1]
    )
)

print(
    "Condition-specific regulator "
    "deviation / actual RMS:",
    float(
        np.sqrt(
            np.mean(
                X_deviation ** 2
            )
        )
        /
        np.sqrt(
            np.mean(
                X_actual ** 2
            )
        )
    )
)


# ============================================================================
# NOTEBOOK INDEX 130: Cell 125. FINAL kernel-shutdown checkpoint
# ============================================================================

# ============================================
# Cell 125. FINAL kernel-shutdown checkpoint
#
# Save all manuscript-relevant final objects.
# NO fitting. NO new analysis.
#
# After this cell succeeds, the kernel can
# be safely shut down.
# ============================================

import numpy as np
import pandas as pd
import os


out_dir = "/home/featurize/work/project1"

npz_path = os.path.join(
    out_dir,
    "replogle_manuscript_final_checkpoint.npz"
)

validation_path = os.path.join(
    out_dir,
    "replogle_manuscript_validation.csv"
)

condition_decomp_path = os.path.join(
    out_dir,
    "replogle_condition_decomposition.csv"
)

regulator_decomp_path = os.path.join(
    out_dir,
    "replogle_regulator_decomposition.csv"
)


# --------------------------------------------
# Main numerical objects
# --------------------------------------------

np.savez_compressed(
    npz_path,

    # gene / condition labels
    response_genes=np.asarray(
        response_genes,
        dtype=str
    ),
    regulator_genes=np.asarray(
        regulator_genes_vc,
        dtype=str
    ),
    conditions=np.asarray(
        final_conditions_vc,
        dtype=str
    ),
    perturbations=np.asarray(
        final_perturbations_vc,
        dtype=str
    ),

    # exact integral design / response
    X_final=np.asarray(
        X_final,
        dtype=np.float64
    ),
    Y_final=np.asarray(
        Y_final,
        dtype=np.float64
    ),

    # corrected primary GRN
    A_primary=np.asarray(
        A_primary,
        dtype=np.float64
    ),
    c_primary=np.asarray(
        c_primary,
        dtype=np.float64
    ),

    # stability
    selection_freq_primary=np.asarray(
        selection_freq_primary,
        dtype=np.float64
    ),
    positive_freq_primary=np.asarray(
        positive_freq_primary,
        dtype=np.float64
    ),
    negative_freq_primary=np.asarray(
        negative_freq_primary,
        dtype=np.float64
    ),
    sign_consistency_primary=np.asarray(
        sign_consistency_primary,
        dtype=np.float64
    ),

    # held-out predictions
    oof_y=np.asarray(
        oof_y,
        dtype=np.float64
    ),
    oof_pred_grn=np.asarray(
        oof_pred_grn,
        dtype=np.float64
    ),
    oof_pred_phase=np.asarray(
        oof_pred_phase,
        dtype=np.float64
    ),
    oof_fold_id=np.asarray(
        oof_fold_id,
        dtype=np.int64
    ),

    # phase / regulator decomposition
    X_shared=np.asarray(
        X_shared,
        dtype=np.float64
    ),
    X_deviation=np.asarray(
        X_deviation,
        dtype=np.float64
    ),

    # important scalar parameters
    global_alpha_integral=np.asarray(
        global_alpha_integral
    ),
    omega_nt=np.asarray(
        omega_nt
    ),
    time_scale=np.asarray(
        time_scale
    ),
    beta_tilde=np.asarray(
        beta_tilde,
        dtype=np.float64
    ),
)


# --------------------------------------------
# Tables useful for manuscript figures
# --------------------------------------------

primary_validation.to_csv(
    validation_path,
    index=False
)

condition_decomposition.to_csv(
    condition_decomp_path,
    index=False
)

regulator_decomposition.to_csv(
    regulator_decomp_path,
    index=False
)


# Cell 124 summary should already exist.
summary_path = os.path.join(
    out_dir,
    "replogle_numerical_experiment_summary.csv"
)


# --------------------------------------------
# Verify files
# --------------------------------------------

paths = [
    npz_path,
    validation_path,
    condition_decomp_path,
    regulator_decomp_path,
    summary_path,
    os.path.join(
        out_dir,
        "replogle_primary_exact_integral_grn.npz"
    ),
    os.path.join(
        out_dir,
        "replogle_primary_edge_table.csv.gz"
    ),
]

print("FINAL FILE CHECK")
print("=" * 60)

for path in paths:

    exists = os.path.exists(path)

    size_mb = (
        os.path.getsize(path)
        / 1024**2
        if exists
        else np.nan
    )

    print(
        f"{os.path.basename(path):50s}",
        f"exists={exists}",
        f"size={size_mb:.3f} MB"
    )


# --------------------------------------------
# Reload the master checkpoint
# --------------------------------------------

check = np.load(
    npz_path,
    allow_pickle=False
)

print("\nMASTER CHECKPOINT")

print(
    "keys:",
    check.files
)

print(
    "X:",
    check["X_final"].shape
)

print(
    "Y:",
    check["Y_final"].shape
)

print(
    "A:",
    check["A_primary"].shape
)

print(
    "selection frequency:",
    check[
        "selection_freq_primary"
    ].shape
)

print(
    "OOF predictions:",
    check[
        "oof_pred_grn"
    ].shape
)

print(
    "shared/deviation:",
    check["X_shared"].shape,
    check["X_deviation"].shape
)


# --------------------------------------------
# Critical integrity checks
# --------------------------------------------

assert np.array_equal(
    check["A_primary"],
    A_primary
)

assert np.array_equal(
    check["selection_freq_primary"],
    selection_freq_primary
)

assert np.allclose(
    check["X_final"],
    X_final,
    equal_nan=True
)

assert np.allclose(
    check["Y_final"],
    Y_final,
    equal_nan=True
)

assert np.allclose(
    check["oof_pred_grn"],
    oof_pred_grn,
    equal_nan=True
)

assert np.allclose(
    check["oof_pred_phase"],
    oof_pred_phase,
    equal_nan=True
)

assert np.allclose(
    check["X_shared"] + check["X_deviation"],
    X_final[:, 1:],
    rtol=1e-12,
    atol=1e-12
)


print(
    "\nAll integrity checks passed."
)

print(
    "\nSAFE TO SHUT DOWN KERNEL."
)

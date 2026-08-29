"""
RPE1 stage 1: data inspection, QC, VeloCycle alignment, and final analysis set.

This stage preserves the notebook's data/QC code. The historical analysis
loaded replogle_grn_checkpoint.npz to restore a 154-condition intermediate panel
before the final VeloCycle phase-coverage filter. That checkpoint is not bundled
because the notebook itself does not contain its bytes. See data/README.md and
archive/CELL_MAP.md before attempting a clean rerun.

This file was mechanically extracted from archive/Replogle_cloud.ipynb.
Notebook cell numbers are preserved below to make provenance auditable.
"""


# ============================================================================
# SOURCE NOTEBOOK INDEX 0: Cell 1. Imports and project paths
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
# SOURCE NOTEBOOK INDEX 1: Cell 2. Load full RPE1 AnnData
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
# SOURCE NOTEBOOK INDEX 2: Cell 3. Check spliced/unspliced and conditions
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
# SOURCE NOTEBOOK INDEX 3: Cell 4. First perturbation QC: cell count
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
# SOURCE NOTEBOOK INDEX 4: Cell 5. Second perturbation QC:
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
# SOURCE NOTEBOOK INDEX 5: Cell 6. Verify target suppression
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
# SOURCE NOTEBOOK INDEX 6: Cell 7. Define conditions and perturbation operators M_q
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
# SOURCE NOTEBOOK INDEX 7: Cell 8. Load official VeloCycle phase metadata
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
# SOURCE NOTEBOOK INDEX 8: Cell 9. Match official phase metadata to full AnnData
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
# SOURCE NOTEBOOK INDEX 9: Cell 10. Inspect VeloCycle MedGeneSet
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
# SOURCE NOTEBOOK INDEX 32: Cell 33. Inspect VeloCycle speed estimates
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
# SOURCE NOTEBOOK INDEX 33: Cell 33b. Recover final perturbation list
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
# SOURCE NOTEBOOK INDEX 34: Cell 34. Restore final perturbations
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
# SOURCE NOTEBOOK INDEX 35: Cell 35. Check kinetics coverage for the
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
# SOURCE NOTEBOOK INDEX 36: Cell 36. Check the 426 VeloCycle kinetic genes
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
# SOURCE NOTEBOOK INDEX 37: Cell 37. NT expression QC for the 426
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
# SOURCE NOTEBOOK INDEX 41: Cell 41. Attach official VeloCycle phase
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
# SOURCE NOTEBOOK INDEX 42: Cell 42. Freeze final condition set under
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
# SOURCE NOTEBOOK INDEX 43: Cell 43. Put VeloCycle speed and kinetics
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
# SOURCE NOTEBOOK INDEX 44: Cell 44. Build VeloCycle-normalized sparse
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
# SOURCE NOTEBOOK INDEX 48: Cell 48. Build spliced trajectories for the
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
# SOURCE NOTEBOOK INDEX 54: Cell 52c. Build RAW-count trajectories
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

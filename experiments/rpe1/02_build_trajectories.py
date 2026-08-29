"""
RPE1 stage 2: final response and regulator trajectory construction.

This stage contains the final trajectory construction used downstream:
10-bin VeloCycle phase grid; NT-relative raw-spliced log-fold-change response
amplitudes; VeloCycle-consistent unspliced trajectories; and weighted
one-harmonic circular smoothing for regulator trajectories.

This file was mechanically extracted from archive/Replogle_cloud.ipynb.
Notebook cell numbers are preserved below to make provenance auditable.
"""


# ============================================================================
# SOURCE NOTEBOOK INDEX 60: Cell 58. Extract compact latent VeloCycle
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
# SOURCE NOTEBOOK INDEX 65: Cell 62b. Restore fixed 10-bin phase grid
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
# SOURCE NOTEBOOK INDEX 66: Cell 63. Build VeloCycle latent trajectories
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
# SOURCE NOTEBOOK INDEX 67: Cell 64. Reconstruct the exact VeloCycle
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
# SOURCE NOTEBOOK INDEX 68: Cell 65. Estimate condition-specific
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
# SOURCE NOTEBOOK INDEX 69: Cell 66. Estimate perturbation-specific
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
# SOURCE NOTEBOOK INDEX 70: Cell 67. Build perturbation-specific
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
# SOURCE NOTEBOOK INDEX 71: Cell 68. Build perturbation-specific
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
# SOURCE NOTEBOOK INDEX 72: Cell 69. Build raw spliced trajectories
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
# SOURCE NOTEBOOK INDEX 73: Cell 70. Circular Fourier smoothing of
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

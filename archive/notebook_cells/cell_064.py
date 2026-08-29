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
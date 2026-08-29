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
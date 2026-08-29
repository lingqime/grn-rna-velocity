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
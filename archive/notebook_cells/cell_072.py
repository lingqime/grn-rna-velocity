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
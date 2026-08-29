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
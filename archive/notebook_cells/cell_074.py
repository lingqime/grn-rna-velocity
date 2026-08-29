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
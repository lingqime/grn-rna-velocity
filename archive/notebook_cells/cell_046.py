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
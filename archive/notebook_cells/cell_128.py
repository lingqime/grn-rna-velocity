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
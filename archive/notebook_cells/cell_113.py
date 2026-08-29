# ============================================
# Cell 108. Regulator hub confounding diagnostic
#
# Question:
# Is stable outgoing degree mainly explained by
# regulator trajectory scale / variability?
#
# Uses final S_regulator_hat only.
# No model refitting.
# ============================================

import numpy as np
import pandas as pd


regulator_diagnostic_records = []


for rj, regulator in enumerate(
    regulator_genes_vc
):

    values = np.asarray(
        S_regulator_hat[
            :, :, rj
        ],
        dtype=np.float64
    ).ravel()


    # ----------------------------------------
    # Basic trajectory scale
    # ----------------------------------------

    mean_level = float(
        np.mean(values)
    )

    median_level = float(
        np.median(values)
    )

    rms_level = float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )

    sd_level = float(
        np.std(values)
    )


    # ----------------------------------------
    # Condition-level mean trajectory
    # ----------------------------------------

    condition_means = np.mean(
        S_regulator_hat[
            :, :, rj
        ],
        axis=1
    )

    condition_sd = float(
        np.std(
            condition_means
        )
    )

    condition_range = float(
        np.max(
            condition_means
        )
        -
        np.min(
            condition_means
        )
    )


    # ----------------------------------------
    # Phase variability within conditions
    # ----------------------------------------

    within_condition_sd = np.std(
        S_regulator_hat[
            :, :, rj
        ],
        axis=1
    )

    mean_phase_sd = float(
        np.mean(
            within_condition_sd
        )
    )


    regulator_diagnostic_records.append({
        "regulator_gene":
            str(regulator),

        "trajectory_mean":
            mean_level,

        "trajectory_median":
            median_level,

        "trajectory_rms":
            rms_level,

        "trajectory_sd":
            sd_level,

        "condition_mean_sd":
            condition_sd,

        "condition_mean_range":
            condition_range,

        "mean_phase_sd":
            mean_phase_sd,
    })


regulator_diagnostics = pd.DataFrame(
    regulator_diagnostic_records
)


# --------------------------------------------
# Merge with network summary
# --------------------------------------------

regulator_hub_diagnostic = (
    regulator_summary
    .merge(
        regulator_diagnostics,
        on="regulator_gene",
        how="left",
        validate="one_to_one"
    )
)


print(
    "Diagnostic table shape:",
    regulator_hub_diagnostic.shape
)


# ============================================
# Spearman correlations
# ============================================

diagnostic_cols = [
    "trajectory_mean",
    "trajectory_rms",
    "trajectory_sd",
    "condition_mean_sd",
    "condition_mean_range",
    "mean_phase_sd",
]


print(
    "\nSpearman correlation with "
    "stable outgoing degree:"
)

corr_stable = (
    regulator_hub_diagnostic[
        [
            "n_stable80_outgoing"
        ]
        + diagnostic_cols
    ]
    .corr(
        method="spearman"
    )
    .loc[
        diagnostic_cols,
        "n_stable80_outgoing"
    ]
    .sort_values(
        ascending=False
    )
)

print(
    corr_stable
)


print(
    "\nSpearman correlation with "
    "full selected outgoing degree:"
)

corr_selected = (
    regulator_hub_diagnostic[
        [
            "n_selected_outgoing"
        ]
        + diagnostic_cols
    ]
    .corr(
        method="spearman"
    )
    .loc[
        diagnostic_cols,
        "n_selected_outgoing"
    ]
    .sort_values(
        ascending=False
    )
)

print(
    corr_selected
)


# ============================================
# Inspect top stable hubs with diagnostics
# ============================================

print(
    "\nTop 25 stable hubs + "
    "trajectory diagnostics:"
)

print(
    regulator_hub_diagnostic
    .sort_values(
        "n_stable80_outgoing",
        ascending=False
    )
    .head(25)
    [
        [
            "regulator_gene",
            "n_stable80_outgoing",
            "stable_fraction_selected",
            "trajectory_mean",
            "trajectory_rms",
            "trajectory_sd",
            "condition_mean_sd",
            "condition_mean_range",
            "mean_phase_sd",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================
# Rank correlations among network metrics
# ============================================

print(
    "\nNetwork-metric Spearman correlations:"
)

print(
    regulator_hub_diagnostic[
        [
            "n_selected_outgoing",
            "n_stable80_outgoing",
            "stable_fraction_selected",
            "mean_selection_freq_selected",
        ]
    ]
    .corr(
        method="spearman"
    )
)
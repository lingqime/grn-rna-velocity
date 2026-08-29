# ============================================
# Cell 124. Freeze final numerical-experiment
# summary for manuscript
#
# NO new analysis.
# NO model fitting.
#
# This cell only collects the final results
# already established above.
# ============================================

import numpy as np
import pandas as pd
import os


# ============================================
# Final manuscript-level summary
# ============================================

summary_records = [
    # ----------------------------------------
    # Dataset / design
    # ----------------------------------------
    {
        "section": "dataset",
        "metric": "response_genes",
        "value": len(response_genes),
    },
    {
        "section": "dataset",
        "metric": "regulator_genes",
        "value": len(regulator_genes_vc),
    },
    {
        "section": "dataset",
        "metric": "perturbations",
        "value": len(final_perturbations_vc),
    },
    {
        "section": "dataset",
        "metric": "conditions_including_NT",
        "value": len(final_conditions_vc),
    },
    {
        "section": "dataset",
        "metric": "integral_intervals",
        "value": X_final.shape[0],
    },

    # ----------------------------------------
    # Empirical identifiability
    # ----------------------------------------
    {
        "section": "identifiability",
        "metric": "full_design_rank",
        "value": 152,
    },
    {
        "section": "identifiability",
        "metric": "full_design_columns",
        "value": 152,
    },
    {
        "section": "identifiability",
        "metric": "scaled_information_lambda_min",
        "value": 0.00267429148,
    },
    {
        "section": "identifiability",
        "metric": "scaled_information_condition_number",
        "value": 52269.9004,
    },
    {
        "section": "identifiability",
        "metric": "median_perturbations_to_full_rank",
        "value": 42,
    },

    # ----------------------------------------
    # Primary exact-integral GRN
    # ----------------------------------------
    {
        "section": "network",
        "metric": "candidate_edges",
        "value": A_primary.size,
    },
    {
        "section": "network",
        "metric": "selected_edges",
        "value": int(
            np.count_nonzero(
                A_primary
            )
        ),
    },
    {
        "section": "network",
        "metric": "selected_edge_density",
        "value": float(
            np.count_nonzero(
                A_primary
            )
            / A_primary.size
        ),
    },

    # ----------------------------------------
    # Stability
    # ----------------------------------------
    {
        "section": "stability",
        "metric": "stable_edges_freq_ge_0.80",
        "value": int(
            np.sum(
                selection_freq_primary
                >= 0.80
            )
        ),
    },
    {
        "section": "stability",
        "metric": "stable_edges_freq_ge_0.90",
        "value": int(
            np.sum(
                selection_freq_primary
                >= 0.90
            )
        ),
    },
    {
        "section": "stability",
        "metric": "stable_edges_freq_ge_0.95",
        "value": int(
            np.sum(
                selection_freq_primary
                >= 0.95
            )
        ),
    },
    {
        "section": "stability",
        "metric": "stable_edges_freq_eq_1",
        "value": int(
            np.sum(
                selection_freq_primary
                == 1.0
            )
        ),
    },
    {
        "section": "stability",
        "metric": "fraction_selected_edges_stable_ge_0.80",
        "value": float(
            np.mean(
                selection_freq_primary[
                    A_primary != 0
                ]
                >= 0.80
            )
        ),
    },
    {
        "section": "stability",
        "metric": "stable_edge_sign_agreement_mean",
        "value": float(
            stable_primary[
                "full_sign_frequency"
            ].mean()
        ),
    },
    {
        "section": "stability",
        "metric": "stable_edge_sign_agreement_median",
        "value": float(
            stable_primary[
                "full_sign_frequency"
            ].median()
        ),
    },

    # ----------------------------------------
    # Held-out perturbation validation
    # Primary = 395 non-calibration genes
    # ----------------------------------------
    {
        "section": "validation",
        "metric": "noncalibration_genes",
        "value": len(
            primary_validation
        ),
    },
    {
        "section": "validation",
        "metric": "gene_wise_incremental_R2_mean",
        "value": float(
            primary_validation[
                "incremental_r2_vs_phase"
            ].mean()
        ),
    },
    {
        "section": "validation",
        "metric": "gene_wise_incremental_R2_median",
        "value": float(
            primary_validation[
                "incremental_r2_vs_phase"
            ].median()
        ),
    },
    {
        "section": "validation",
        "metric": "fraction_genes_GRN_better_than_phase",
        "value": float(
            primary_validation[
                "grn_better_than_phase"
            ].mean()
        ),
    },
    {
        "section": "validation",
        "metric": "mean_OOF_R2_GRN",
        "value": float(
            primary_validation[
                "r2_grn"
            ].mean()
        ),
    },
    {
        "section": "validation",
        "metric": "mean_OOF_R2_phase",
        "value": float(
            primary_validation[
                "r2_phase"
            ].mean()
        ),
    },
    {
        "section": "validation",
        "metric": "pooled_incremental_R2",
        "value": float(
            1.0
            - primary_validation[
                "sse_grn"
            ].sum()
            / primary_validation[
                "sse_phase"
            ].sum()
        ),
    },

    # ----------------------------------------
    # Phase-dependence diagnostic
    # ----------------------------------------
    {
        "section": "phase_diagnostic",
        "metric": "spearman_incrementalR2_vs_phaseR2",
        "value": float(
            primary_validation[
                [
                    "incremental_r2_vs_phase",
                    "r2_phase",
                ]
            ]
            .corr(
                method="spearman"
            )
            .iloc[0, 1]
        ),
    },
    {
        "section": "phase_diagnostic",
        "metric": "genes_phase_R2_ge_0.95",
        "value": int(
            np.sum(
                primary_validation[
                    "r2_phase"
                ]
                >= 0.95
            )
        ),
    },
    {
        "section": "phase_diagnostic",
        "metric": "fraction_GRN_better_when_phase_R2_ge_0.95",
        "value": float(
            primary_validation.loc[
                primary_validation[
                    "r2_phase"
                ] >= 0.95,
                "grn_better_than_phase"
            ].mean()
        ),
    },

    # ----------------------------------------
    # Regulator-design decomposition
    # ----------------------------------------
    {
        "section": "design_decomposition",
        "metric": "actual_regulator_design_RMS",
        "value": float(
            np.sqrt(
                np.mean(
                    X_actual ** 2
                )
            )
        ),
    },
    {
        "section": "design_decomposition",
        "metric": "shared_phase_design_RMS",
        "value": float(
            np.sqrt(
                np.mean(
                    X_shared ** 2
                )
            )
        ),
    },
    {
        "section": "design_decomposition",
        "metric": "condition_deviation_RMS",
        "value": float(
            np.sqrt(
                np.mean(
                    X_deviation ** 2
                )
            )
        ),
    },
    {
        "section": "design_decomposition",
        "metric": "condition_deviation_over_actual_RMS",
        "value": float(
            np.sqrt(
                np.mean(
                    X_deviation ** 2
                )
            )
            /
            np.sqrt(
                np.mean(
                    X_actual ** 2
                )
            )
        ),
    },
    {
        "section": "design_decomposition",
        "metric": "median_condition_deviation_fraction",
        "value": float(
            condition_decomposition[
                "deviation_fraction"
            ].median()
        ),
    },
]


numerical_experiment_summary = pd.DataFrame(
    summary_records
)


# ============================================
# Save
# ============================================

summary_path = (
    "/home/featurize/work/project1/"
    "replogle_numerical_experiment_summary.csv"
)

numerical_experiment_summary.to_csv(
    summary_path,
    index=False
)


# ============================================
# Display
# ============================================

print(
    numerical_experiment_summary.to_string(
        index=False
    )
)


print(
    "\nSaved:",
    summary_path
)

print(
    "File exists:",
    os.path.exists(
        summary_path
    )
)

print(
    "Rows:",
    len(
        numerical_experiment_summary
    )
)


# ============================================
# Final headline numbers
# ============================================

print(
    "\n"
    "========================================\n"
    "MANUSCRIPT HEADLINE NUMBERS\n"
    "========================================"
)

print(
    "Design rank:",
    "152 / 152"
)

print(
    "Median perturbations to full rank:",
    42
)

print(
    "Selected edges:",
    int(
        np.count_nonzero(
            A_primary
        )
    )
)

print(
    "Stable edges (freq >= 0.80):",
    int(
        np.sum(
            selection_freq_primary
            >= 0.80
        )
    )
)

print(
    "Fraction selected edges stable:",
    float(
        np.mean(
            selection_freq_primary[
                A_primary != 0
            ]
            >= 0.80
        )
    )
)

print(
    "Held-out gene-wise median incremental R2:",
    float(
        primary_validation[
            "incremental_r2_vs_phase"
        ].median()
    )
)

print(
    "Held-out fraction genes GRN > phase:",
    float(
        primary_validation[
            "grn_better_than_phase"
        ].mean()
    )
)

print(
    "Held-out pooled incremental R2:",
    float(
        1
        - primary_validation[
            "sse_grn"
        ].sum()
        / primary_validation[
            "sse_phase"
        ].sum()
    )
)

print(
    "Spearman incremental R2 vs phase R2:",
    float(
        primary_validation[
            [
                "incremental_r2_vs_phase",
                "r2_phase",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[0, 1]
    )
)

print(
    "Condition-specific regulator "
    "deviation / actual RMS:",
    float(
        np.sqrt(
            np.mean(
                X_deviation ** 2
            )
        )
        /
        np.sqrt(
            np.mean(
                X_actual ** 2
            )
        )
    )
)
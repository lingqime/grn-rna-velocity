# ============================================
# Cell 111. Diagnose self-signal coupling
#
# For the 12 genes that are both:
# - response genes
# - regulator genes
#
# Compare condition-level response trajectory
# against the same-gene regulator trajectory.
#
# IMPORTANT:
# - exclude q = g condition
# - diagnostic only
# - no model refitting
# ============================================

import numpy as np
import pandas as pd


self_coupling_records = []


condition_names_arr = np.asarray(
    final_conditions_vc,
    dtype=object
)


overlap_genes = np.intersect1d(
    np.asarray(
        response_genes,
        dtype=object
    ),
    np.asarray(
        regulator_genes_vc,
        dtype=object
    )
)


print(
    "Overlap genes:",
    len(overlap_genes)
)

print(
    overlap_genes
)


for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]

    rj = np.where(
        np.asarray(
            regulator_genes_vc,
            dtype=object
        ) == g
    )[0][0]


    # ----------------------------------------
    # Condition-level means
    # ----------------------------------------

    response_mean = np.mean(
        S_response_hat[
            :, :, gi
        ],
        axis=1
    )

    regulator_mean = np.mean(
        S_regulator_hat[
            :, :, rj
        ],
        axis=1
    )


    # Exclude the direct perturbation q = g
    keep = (
        condition_names_arr
        != g
    )

    x = regulator_mean[
        keep
    ]

    y = response_mean[
        keep
    ]


    pearson = float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )

    spearman = float(
        pd.Series(x).corr(
            pd.Series(y),
            method="spearman"
        )
    )


    # ----------------------------------------
    # Compare NT-relative condition effects
    # instead of absolute expression levels
    # ----------------------------------------

    nt_idx = np.where(
        condition_names_arr
        == "non-targeting"
    )[0][0]

    eps = 1e-12

    response_log_ratio = np.log(
        (
            response_mean
            + eps
        )
        /
        (
            response_mean[
                nt_idx
            ]
            + eps
        )
    )

    regulator_log_ratio = np.log(
        (
            regulator_mean
            + eps
        )
        /
        (
            regulator_mean[
                nt_idx
            ]
            + eps
        )
    )


    x_lr = regulator_log_ratio[
        keep
    ]

    y_lr = response_log_ratio[
        keep
    ]


    pearson_logratio = float(
        np.corrcoef(
            x_lr,
            y_lr
        )[0, 1]
    )

    spearman_logratio = float(
        pd.Series(
            x_lr
        ).corr(
            pd.Series(
                y_lr
            ),
            method="spearman"
        )
    )


    self_coupling_records.append({
        "gene":
            str(g),

        "pearson_level":
            pearson,

        "spearman_level":
            spearman,

        "pearson_nt_logratio":
            pearson_logratio,

        "spearman_nt_logratio":
            spearman_logratio,

        "self_coefficient":
            float(
                A_hat_integral[
                    gi, rj
                ]
            ),

        "self_selection_frequency":
            float(
                selection_freq[
                    gi, rj
                ]
            ),
    })


self_coupling_df = pd.DataFrame(
    self_coupling_records
)


print(
    "\nSelf-signal coupling:"
)

print(
    self_coupling_df
    .sort_values(
        "spearman_nt_logratio",
        ascending=False
    )
    .to_string(
        index=False
    )
)


print(
    "\nSummary:"
)

print(
    self_coupling_df[
        [
            "pearson_level",
            "spearman_level",
            "pearson_nt_logratio",
            "spearman_nt_logratio",
        ]
    ].describe()
)
"""
RPE1 stage 4: exact-integral L1 calibration and corrected primary GRN fit.

This stage preserves the final exact-integral estimator. The fixed 31-gene
calibration panel is restored explicitly; alpha is calibrated for the exact
integral estimator; the full 426x151 network is fitted; and the 12 overlap
response/regulator rows are refitted with their self predictor excluded.

This file was mechanically extracted from archive/Replogle_cloud.ipynb.
Notebook cell numbers are preserved below to make provenance auditable.
"""


# ============================================================================
# SOURCE NOTEBOOK INDEX 98: Cell 94b. Restore the fixed 31-gene
# ============================================================================

# ============================================
# Cell 94b. Restore the fixed 31-gene
# calibration panel from Cell 82
# ============================================

import numpy as np

calibration_genes = np.asarray([
    "CDKN1B",
    "MLH1",
    "THOC1",
    "BCL7C",
    "EXOC6",
    "MAP3K20",
    "UCHL5",
    "CAMK2G",
    "TXLNG",
    "CSNK2A2",
    "KIF22",
    "DSN1",
    "FANCI",
    "DNA2",
    "NCAPH",
    "PPP6C",
    "NCAPD2",
    "RMI1",
    "KMT2E",
    "VPS4B",
    "TACC3",
    "CACUL1",
    "TFDP2",
    "CDKN3",
    "CSNK1A1",
    "ERCC6",
    "UBE2B",
    "EXT1",
    "HSP90AB1",
    "PLK2",
    "RPS3",
], dtype=object)


print(
    "Calibration genes:",
    len(calibration_genes)
)

print(
    calibration_genes
)


# Verify all are still in the 426 response set
missing = [
    g
    for g in calibration_genes
    if g not in set(response_genes)
]

print(
    "\nMissing from response_genes:",
    missing
)

assert len(calibration_genes) == 31
assert len(missing) == 0

print(
    "\nFixed calibration panel restored."
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 102: Cell 97. Alpha calibration for the EXACT
# ============================================================================

# ============================================
# Cell 97. Alpha calibration for the EXACT
# integral estimator (Algorithm 5.1)
#
# Model:
#   Y = c * dt + X_reg A
#
# c is unpenalized via fold-specific FWL.
# L1 penalty applies only to A.
#
# Held-out perturbation conditions;
# NT always remains in training.
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso


alpha_grid_integral = np.logspace(
    -4,
    0,
    40
)

integral_alpha_records = []


for pi, g in enumerate(
    calibration_genes
):

    gi = np.where(
        response_genes == g
    )[0][0]

    valid_mask = valid_row_masks[g]

    d_g = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_g = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y_g = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )

    groups_g = row_conditions[
        valid_mask
    ]

    pert_conditions = np.asarray([
        q
        for q in np.unique(
            groups_g
        )
        if q != "non-targeting"
    ], dtype=object)

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=2026
    )

    fold_data = []


    # ----------------------------------------
    # Prepare folds once per gene
    # ----------------------------------------

    for _, val_cond_idx in kf.split(
        pert_conditions
    ):

        val_conditions = pert_conditions[
            val_cond_idx
        ]

        val_fold = np.isin(
            groups_g,
            val_conditions
        )

        train_fold = ~val_fold

        assert np.any(
            groups_g[
                train_fold
            ] == "non-targeting"
        )

        assert not np.any(
            groups_g[
                val_fold
            ] == "non-targeting"
        )


        d_train = d_g[
            train_fold
        ]

        X_train = Xr_g[
            train_fold
        ]

        y_train = y_g[
            train_fold
        ]

        d_val = d_g[
            val_fold
        ]

        X_val = Xr_g[
            val_fold
        ]

        y_val = y_g[
            val_fold
        ]


        # ------------------------------------
        # Training-only FWL projection
        # ------------------------------------

        dd = np.dot(
            d_train,
            d_train
        )

        coef_d_X = (
            d_train @ X_train
        ) / dd

        coef_d_y = (
            d_train @ y_train
        ) / dd


        Xp_train = (
            X_train
            - d_train[:, None]
            * coef_d_X[None, :]
        )

        yp_train = (
            y_train
            - d_train
            * coef_d_y
        )


        # ------------------------------------
        # Scale projected predictors
        # WITHOUT centering
        # ------------------------------------

        x_scale = np.sqrt(
            np.mean(
                Xp_train ** 2,
                axis=0
            )
        )

        assert np.all(
            np.isfinite(
                x_scale
            )
        )

        assert np.all(
            x_scale > 0
        )


        y_scale = np.sqrt(
            np.mean(
                yp_train ** 2
            )
        )

        assert (
            np.isfinite(y_scale)
            and y_scale > 0
        )


        Z_train = (
            Xp_train
            / x_scale[None, :]
        )

        yz_train = (
            yp_train
            / y_scale
        )


        fold_data.append({
            "d_train":
                d_train,
            "X_train":
                X_train,
            "y_train":
                y_train,

            "d_val":
                d_val,
            "X_val":
                X_val,
            "y_val":
                y_val,

            "dd":
                dd,
            "x_scale":
                x_scale,
            "y_scale":
                y_scale,
            "Z_train":
                Z_train,
            "yz_train":
                yz_train,
        })


    # ----------------------------------------
    # Evaluate alpha grid
    # ----------------------------------------

    for alpha in alpha_grid_integral:

        sse_model = 0.0
        sse_baseline = 0.0

        fold_nonzero = []


        for fd in fold_data:

            model = Lasso(
                alpha=float(alpha),
                fit_intercept=False,
                max_iter=20000,
                tol=1e-6,
                selection="cyclic",
            )

            model.fit(
                fd["Z_train"],
                fd["yz_train"]
            )


            # --------------------------------
            # Back-transform A
            # --------------------------------

            A_fold = (
                fd["y_scale"]
                * model.coef_
                / fd["x_scale"]
            )


            # --------------------------------
            # Exact unpenalized c on training
            # --------------------------------

            c_fold = (
                fd["d_train"]
                @ (
                    fd["y_train"]
                    - fd["X_train"]
                    @ A_fold
                )
            ) / fd["dd"]


            # --------------------------------
            # Held-out integral prediction
            # --------------------------------

            pred_val = (
                fd["d_val"]
                * c_fold
                + fd["X_val"]
                @ A_fold
            )


            # --------------------------------
            # Baseline:
            # intercept-only integral model
            # fitted on training rows
            # --------------------------------

            c0_fold = (
                fd["d_train"]
                @ fd["y_train"]
            ) / fd["dd"]

            pred0_val = (
                fd["d_val"]
                * c0_fold
            )


            sse_model += np.sum(
                (
                    fd["y_val"]
                    - pred_val
                ) ** 2
            )

            sse_baseline += np.sum(
                (
                    fd["y_val"]
                    - pred0_val
                ) ** 2
            )

            fold_nonzero.append(
                np.count_nonzero(
                    A_fold
                )
            )


        relative_mse = (
            sse_model
            / sse_baseline
        )

        integral_alpha_records.append({
            "gene":
                g,
            "alpha":
                float(alpha),
            "relative_mse":
                relative_mse,
            "skill_vs_intercept":
                1.0
                - relative_mse,
            "mean_nonzero":
                np.mean(
                    fold_nonzero
                ),
        })


    print(
        f"[{pi + 1:02d}/"
        f"{len(calibration_genes)}] "
        f"{g} done"
    )


integral_alpha_cv = pd.DataFrame(
    integral_alpha_records
)


# --------------------------------------------
# Equal weighting across calibration genes
# --------------------------------------------

integral_alpha_summary = (
    integral_alpha_cv
    .groupby(
        "alpha",
        as_index=False
    )
    .agg(
        mean_relative_mse=(
            "relative_mse",
            "mean"
        ),
        median_relative_mse=(
            "relative_mse",
            "median"
        ),
        mean_skill=(
            "skill_vs_intercept",
            "mean"
        ),
        mean_nonzero=(
            "mean_nonzero",
            "mean"
        ),
    )
    .sort_values(
        "mean_relative_mse"
    )
    .reset_index(
        drop=True
    )
)


best_integral_row = (
    integral_alpha_summary.iloc[0]
)

global_alpha_integral = float(
    best_integral_row[
        "alpha"
    ]
)


print(
    "\nBest exact-integral alpha:",
    global_alpha_integral
)

print(
    "\nTop 10 alpha values:"
)

print(
    integral_alpha_summary
    .head(10)
    .to_string(
        index=False
    )
)

print(
    "\nBest-alpha mean skill:",
    float(
        best_integral_row[
            "mean_skill"
        ]
    )
)

print(
    "Best-alpha median relative MSE:",
    float(
        best_integral_row[
            "median_relative_mse"
        ]
    )
)

print(
    "Best-alpha mean nonzero:",
    float(
        best_integral_row[
            "mean_nonzero"
        ]
    )
)

print(
    "\nRate-form alpha:",
    global_alpha
)

print(
    "Residual-validation alpha:",
    global_alpha_resid
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 103: Cell 98. FINAL exact-integral GRN fit
# ============================================================================

# ============================================
# Cell 98. FINAL exact-integral GRN fit
#
# Algorithm 5.1 objective:
#
#   Y_g = c_g * dt + X_reg A_g
#
# c_g unpenalized via exact FWL projection.
# L1 penalty only on A_g.
#
# Direct q=g rows remain excluded through
# valid_row_masks.
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


n_response = len(
    response_genes
)

n_regulator = len(
    regulator_genes_vc
)


A_hat_integral = np.zeros(
    (
        n_response,
        n_regulator
    ),
    dtype=np.float64
)

c_hat_integral = np.zeros(
    n_response,
    dtype=np.float64
)


integral_fit_records = []


for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]


    # ----------------------------------------
    # Original integral system
    # ----------------------------------------

    d = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )


    # ----------------------------------------
    # Exact FWL projection of unpenalized dt
    # ----------------------------------------

    dd = np.dot(
        d,
        d
    )

    coef_d_X = (
        d @ Xr
    ) / dd

    coef_d_y = (
        d @ y
    ) / dd


    Xp = (
        Xr
        - d[:, None]
        * coef_d_X[None, :]
    )

    yp = (
        y
        - d * coef_d_y
    )


    # ----------------------------------------
    # RMS scaling only — NO centering
    # ----------------------------------------

    x_scale = np.sqrt(
        np.mean(
            Xp ** 2,
            axis=0
        )
    )

    assert np.all(
        np.isfinite(
            x_scale
        )
    )

    assert np.all(
        x_scale > 0
    )


    y_scale = np.sqrt(
        np.mean(
            yp ** 2
        )
    )

    assert (
        np.isfinite(y_scale)
        and y_scale > 0
    )


    Z = (
        Xp
        / x_scale[None, :]
    )

    yz = (
        yp
        / y_scale
    )


    # ----------------------------------------
    # Frozen exact-integral alpha
    # ----------------------------------------

    model = Lasso(
        alpha=global_alpha_integral,
        fit_intercept=False,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z,
        yz
    )


    # ----------------------------------------
    # Back-transform regulatory coefficients
    # ----------------------------------------

    A_g = (
        y_scale
        * model.coef_
        / x_scale
    )


    # ----------------------------------------
    # Exact unpenalized basal coefficient
    # conditional on A_g
    # ----------------------------------------

    c_g = (
        d
        @ (
            y
            - Xr @ A_g
        )
    ) / dd


    A_hat_integral[
        gi, :
    ] = A_g

    c_hat_integral[
        gi
    ] = c_g


    # ----------------------------------------
    # Diagnostics on original integral scale
    # ----------------------------------------

    pred = (
        d * c_g
        + Xr @ A_g
    )

    residual = (
        y
        - pred
    )


    # Intercept-only baseline
    c0 = (
        d @ y
    ) / dd

    pred0 = (
        d * c0
    )


    sse = np.sum(
        residual ** 2
    )

    sse0 = np.sum(
        (
            y
            - pred0
        ) ** 2
    )


    integral_fit_records.append({
        "gene":
            g,

        "n_rows":
            len(y),

        "n_nonzero":
            int(
                np.count_nonzero(
                    A_g
                )
            ),

        "integral_mse":
            np.mean(
                residual ** 2
            ),

        "skill_vs_intercept":
            1.0
            - sse / sse0,

        "fwl_intercept_score":
            abs(
                d @ residual
            ),
    })


    if (
        (gi + 1) % 25 == 0
        or gi == 0
        or gi + 1
        == n_response
    ):
        print(
            f"[{gi + 1:03d}/"
            f"{n_response}] "
            f"{g} done"
        )


integral_fit_summary = pd.DataFrame(
    integral_fit_records
)


# ============================================
# Global validation checks
# ============================================

print(
    "\nA_hat_integral shape:",
    A_hat_integral.shape
)

print(
    "c_hat_integral shape:",
    c_hat_integral.shape
)

print(
    "NaN/inf in A:",
    (
        ~np.isfinite(
            A_hat_integral
        )
    ).any()
)

print(
    "NaN/inf in c:",
    (
        ~np.isfinite(
            c_hat_integral
        )
    ).any()
)


print(
    "\nNonzero edges / response gene:"
)

print(
    integral_fit_summary[
        "n_nonzero"
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


total_nonzero = np.count_nonzero(
    A_hat_integral
)

total_possible = (
    A_hat_integral.size
)

print(
    "\nTotal nonzero edges:",
    total_nonzero,
    "/",
    total_possible
)

print(
    "Network density:",
    total_nonzero
    / total_possible
)


print(
    "\nTraining skill vs intercept-only:"
)

print(
    integral_fit_summary[
        "skill_vs_intercept"
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
    "\nFWL first-order-condition check:"
)

print(
    "max |dt^T residual|:",
    integral_fit_summary[
        "fwl_intercept_score"
    ].max()
)


# --------------------------------------------
# Compare exact-integral and old rate network
# --------------------------------------------

old_nz = np.count_nonzero(
    A_hat,
    axis=1
)

new_nz = np.count_nonzero(
    A_hat_integral,
    axis=1
)

print(
    "\nOld rate-form mean nonzero:",
    old_nz.mean()
)

print(
    "Exact-integral mean nonzero:",
    new_nz.mean()
)

print(
    "Median change in edges/gene:",
    np.median(
        new_nz
        - old_nz
    )
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 115: Cell 110. Diagnose the 12 possible self-edges
# ============================================================================

# ============================================
# Cell 110. Diagnose the 12 possible self-edges
#
# Goal:
# inspect whether stable self-edges are
# unusually strong / unusually stable and
# whether they are associated with regulator
# redundancy.
#
# No model refitting.
# ============================================

import numpy as np
import pandas as pd


self_edge_diag = (
    edge_table[
        edge_table[
            "self_edge"
        ]
    ]
    .merge(
        redundancy_df,
        on="regulator_gene",
        how="left",
        validate="one_to_one"
    )
    .copy()
)


# --------------------------------------------
# Rank each self-edge among all outgoing edges
# of the same regulator.
#
# Rank 1 = largest |coefficient|
# Rank 1 = highest selection frequency
# --------------------------------------------

coef_ranks = []
freq_ranks = []

for _, row in self_edge_diag.iterrows():

    regulator = row[
        "regulator_gene"
    ]

    response = row[
        "response_gene"
    ]

    outgoing = edge_table[
        edge_table[
            "regulator_gene"
        ] == regulator
    ].copy()


    coef_rank = (
        outgoing[
            "abs_coefficient"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .loc[
            outgoing[
                "response_gene"
            ] == response
        ]
        .iloc[0]
    )


    freq_rank = (
        outgoing[
            "selection_frequency"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .loc[
            outgoing[
                "response_gene"
            ] == response
        ]
        .iloc[0]
    )


    coef_ranks.append(
        int(coef_rank)
    )

    freq_ranks.append(
        int(freq_rank)
    )


self_edge_diag[
    "abs_coef_outgoing_rank"
] = coef_ranks

self_edge_diag[
    "selection_freq_outgoing_rank"
] = freq_ranks


# --------------------------------------------
# Compare self-edge stability against
# non-self selected edges
# --------------------------------------------

selected_self = edge_table[
    edge_table[
        "self_edge"
    ]
    &
    edge_table[
        "full_selected"
    ]
]

selected_nonself = edge_table[
    (~edge_table[
        "self_edge"
    ])
    &
    edge_table[
        "full_selected"
    ]
]


print(
    "Possible self edges:",
    len(
        self_edge_diag
    )
)

print(
    "Selected self edges:",
    len(
        selected_self
    )
)

print(
    "Stable >=0.8 self edges:",
    int(
        selected_self[
            "stable_80"
        ].sum()
    )
)


print(
    "\nSelf-edge selection frequencies:"
)

print(
    self_edge_diag[
        "selection_frequency"
    ].describe()
)


print(
    "\nSelected non-self edge "
    "selection frequencies:"
)

print(
    selected_nonself[
        "selection_frequency"
    ].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nDetailed self-edge diagnostics:"
)

print(
    self_edge_diag[
        [
            "response_gene",
            "regulator_gene",
            "coefficient",
            "selection_frequency",
            "positive_frequency",
            "negative_frequency",
            "sign_consistency",
            "abs_coef_outgoing_rank",
            "selection_freq_outgoing_rank",
            "max_abs_corr",
            "n_abs_corr_ge_080",
        ]
    ]
    .sort_values(
        "selection_frequency",
        ascending=False
    )
    .to_string(
        index=False
    )
)


# --------------------------------------------
# Sign composition
# --------------------------------------------

print(
    "\nSelected self-edge signs:"
)

print(
    selected_self[
        "sign"
    ].value_counts()
    .sort_index()
)


# --------------------------------------------
# Compare stability proportions
# --------------------------------------------

self_stable_fraction = float(
    selected_self[
        "stable_80"
    ].mean()
)

nonself_stable_fraction = float(
    selected_nonself[
        "stable_80"
    ].mean()
)

print(
    "\nStable fraction among "
    "selected self edges:",
    self_stable_fraction
)

print(
    "Stable fraction among "
    "selected non-self edges:",
    nonself_stable_fraction
)

print(
    "Ratio:",
    self_stable_fraction
    / nonself_stable_fraction
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 116: Cell 111. Diagnose self-signal coupling
# ============================================================================

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


# ============================================================================
# SOURCE NOTEBOOK INDEX 117: Cell 112. Refit the 12 overlap response genes
# ============================================================================

# ============================================
# Cell 112. Refit the 12 overlap response genes
# with self predictor explicitly excluded
#
# Primary exact-integral estimator:
# - same global_alpha_integral
# - same valid rows
# - same exact FWL treatment of dt
# - same RMS scaling
# - ONLY difference:
#     predictor g is removed when response = g
#
# Other 414 response genes remain unchanged.
# ============================================

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso


A_hat_integral_noself = (
    A_hat_integral.copy()
)

c_hat_integral_noself = (
    c_hat_integral.copy()
)


noself_refit_records = []


for g in overlap_genes:

    gi = np.where(
        np.asarray(
            response_genes,
            dtype=object
        ) == g
    )[0][0]

    self_rj = np.where(
        np.asarray(
            regulator_genes_vc,
            dtype=object
        ) == g
    )[0][0]


    valid_mask = valid_row_masks[g]


    d = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Xr_full = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    )

    y = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    )


    # ----------------------------------------
    # Explicitly remove same-gene predictor
    # ----------------------------------------

    predictor_keep = np.ones(
        len(regulator_genes_vc),
        dtype=bool
    )

    predictor_keep[
        self_rj
    ] = False


    Xr = Xr_full[
        :,
        predictor_keep
    ]


    # ----------------------------------------
    # Exact FWL projection for unpenalized
    # dt / basal term
    # ----------------------------------------

    dd = d @ d

    coef_d_X = (
        d @ Xr
    ) / dd

    coef_d_y = (
        d @ y
    ) / dd


    Xp = (
        Xr
        - d[:, None]
        * coef_d_X[None, :]
    )

    yp = (
        y
        - d * coef_d_y
    )


    # ----------------------------------------
    # RMS scaling
    # ----------------------------------------

    x_scale = np.sqrt(
        np.mean(
            Xp ** 2,
            axis=0
        )
    )

    y_scale = np.sqrt(
        np.mean(
            yp ** 2
        )
    )


    assert np.all(
        np.isfinite(
            x_scale
        )
    )

    assert np.all(
        x_scale > 0
    )

    assert (
        np.isfinite(y_scale)
        and y_scale > 0
    )


    Z = (
        Xp
        / x_scale[None, :]
    )

    yz = (
        yp
        / y_scale
    )


    model = Lasso(
        alpha=global_alpha_integral,
        fit_intercept=False,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        Z,
        yz
    )


    A_reduced = (
        y_scale
        * model.coef_
        / x_scale
    )


    # ----------------------------------------
    # Reconstruct full 151-vector,
    # forcing self coefficient exactly zero
    # ----------------------------------------

    A_new = np.zeros(
        len(regulator_genes_vc),
        dtype=np.float64
    )

    A_new[
        predictor_keep
    ] = A_reduced

    A_new[
        self_rj
    ] = 0.0


    # ----------------------------------------
    # Recover unpenalized basal coefficient
    # ----------------------------------------

    c_new = (
        d
        @ (
            y
            - Xr_full @ A_new
        )
    ) / dd


    # ----------------------------------------
    # Diagnostics
    # ----------------------------------------

    pred_new = (
        d * c_new
        + Xr_full @ A_new
    )

    residual_new = (
        y - pred_new
    )

    baseline_c = (
        d @ y
    ) / dd

    baseline_pred = (
        d * baseline_c
    )

    mse_new = float(
        np.mean(
            residual_new ** 2
        )
    )

    mse_baseline = float(
        np.mean(
            (
                y
                - baseline_pred
            ) ** 2
        )
    )

    skill_new = (
        1.0
        - mse_new
        / mse_baseline
    )


    A_old = (
        A_hat_integral[
            gi
        ]
    )

    old_selected = (
        A_old != 0
    )

    new_selected = (
        A_new != 0
    )


    changed_selection = np.sum(
        old_selected
        != new_selected
    )


    # Save corrected row
    A_hat_integral_noself[
        gi
    ] = A_new

    c_hat_integral_noself[
        gi
    ] = c_new


    noself_refit_records.append({
        "gene":
            str(g),

        "old_self_coef":
            float(
                A_old[
                    self_rj
                ]
            ),

        "new_self_coef":
            float(
                A_new[
                    self_rj
                ]
            ),

        "old_n_nonzero":
            int(
                np.count_nonzero(
                    A_old
                )
            ),

        "new_n_nonzero":
            int(
                np.count_nonzero(
                    A_new
                )
            ),

        "changed_selection":
            int(
                changed_selection
            ),

        "old_skill":
            float(
                integral_fit_summary.loc[
                    integral_fit_summary[
                        "gene"
                    ] == g,
                    "skill_vs_intercept"
                ].iloc[0]
            ),

        "new_skill":
            float(
                skill_new
            ),

        "skill_change":
            float(
                skill_new
                -
                integral_fit_summary.loc[
                    integral_fit_summary[
                        "gene"
                    ] == g,
                    "skill_vs_intercept"
                ].iloc[0]
            ),

        "dt_residual_orthogonality":
            float(
                abs(
                    d @ residual_new
                )
            ),
    })


noself_refit_summary = pd.DataFrame(
    noself_refit_records
)


print(
    "Corrected overlap genes:",
    len(
        noself_refit_summary
    )
)


print(
    "\nRefit summary:"
)

print(
    noself_refit_summary
    .sort_values(
        "skill_change"
    )
    .to_string(
        index=False
    )
)


# ============================================
# Global network changes
# ============================================

old_selected_all = (
    A_hat_integral != 0
)

new_selected_all = (
    A_hat_integral_noself != 0
)


print(
    "\nOld total edges:",
    int(
        old_selected_all.sum()
    )
)

print(
    "New total edges:",
    int(
        new_selected_all.sum()
    )
)

print(
    "Net edge-count change:",
    int(
        new_selected_all.sum()
        - old_selected_all.sum()
    )
)


print(
    "\nTotal selection-status changes:",
    int(
        np.sum(
            old_selected_all
            != new_selected_all
        )
    )
)


# ============================================
# Verify only the 12 overlap response rows
# changed
# ============================================

overlap_mask_response = np.isin(
    np.asarray(
        response_genes,
        dtype=object
    ),
    overlap_genes
)

print(
    "\nNon-overlap rows unchanged exactly:",
    np.array_equal(
        A_hat_integral_noself[
            ~overlap_mask_response
        ],
        A_hat_integral[
            ~overlap_mask_response
        ]
    )
)


# ============================================
# Verify all possible self coefficients = 0
# ============================================

self_values_after = []

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

    self_values_after.append(
        A_hat_integral_noself[
            gi, rj
        ]
    )


self_values_after = np.asarray(
    self_values_after
)


print(
    "\nAll 12 self coefficients exactly zero:",
    bool(
        np.all(
            self_values_after == 0
        )
    )
)

print(
    "Max |self coefficient|:",
    float(
        np.max(
            np.abs(
                self_values_after
            )
        )
    )
)


# ============================================================================
# SOURCE NOTEBOOK INDEX 119: Cell 114. Freeze corrected primary network
# ============================================================================

# ============================================
# Cell 114. Freeze corrected primary network
# and rebuild the canonical edge table
#
# From this cell onward:
#
# PRIMARY network:
#   A_primary = A_hat_integral_noself
#
# PRIMARY stability:
#   selection_freq_primary
#     = selection_freq_noself
#
# Self predictors are structurally excluded
# for the 12 response/regulator overlap genes.
# ============================================

import numpy as np
import pandas as pd


# ============================================
# Canonical primary objects
# ============================================

A_primary = (
    A_hat_integral_noself.copy()
)

c_primary = (
    c_hat_integral_noself.copy()
)

selection_freq_primary = (
    selection_freq_noself.copy()
)

positive_freq_primary = (
    positive_freq_noself.copy()
)

negative_freq_primary = (
    negative_freq_noself.copy()
)

sign_consistency_primary = (
    sign_consistency_noself.copy()
)


# ============================================
# Rebuild edge table
# ============================================

edge_records_primary = []


response_arr = np.asarray(
    response_genes,
    dtype=object
)

regulator_arr = np.asarray(
    regulator_genes_vc,
    dtype=object
)


for gi, response in enumerate(
    response_arr
):

    for rj, regulator in enumerate(
        regulator_arr
    ):

        coef = float(
            A_primary[
                gi, rj
            ]
        )

        sel_freq = float(
            selection_freq_primary[
                gi, rj
            ]
        )

        pos_freq = float(
            positive_freq_primary[
                gi, rj
            ]
        )

        neg_freq = float(
            negative_freq_primary[
                gi, rj
            ]
        )

        sign_cons = float(
            sign_consistency_primary[
                gi, rj
            ]
        )

        is_self = (
            str(response)
            == str(regulator)
        )

        edge_records_primary.append({
            "response_gene":
                str(response),

            "regulator_gene":
                str(regulator),

            "coefficient":
                coef,

            "abs_coefficient":
                abs(coef),

            "sign":
                int(
                    np.sign(coef)
                ),

            "full_selected":
                bool(
                    coef != 0
                ),

            "selection_frequency":
                sel_freq,

            "positive_frequency":
                pos_freq,

            "negative_frequency":
                neg_freq,

            "sign_consistency":
                sign_cons,

            "stable_80":
                bool(
                    sel_freq >= 0.80
                ),

            "stable_90":
                bool(
                    sel_freq >= 0.90
                ),

            "stable_95":
                bool(
                    sel_freq >= 0.95
                ),

            "stable_100":
                bool(
                    sel_freq == 1.0
                ),

            "self_edge":
                bool(
                    is_self
                ),
        })


edge_table_primary = pd.DataFrame(
    edge_records_primary
)


# ============================================
# Core checks
# ============================================

print(
    "Primary edge table shape:",
    edge_table_primary.shape
)

print(
    "Expected:",
    (
        len(response_genes)
        * len(regulator_genes_vc),
        15
    )
)


print(
    "\nFull-data selected:",
    int(
        edge_table_primary[
            "full_selected"
        ].sum()
    )
)


print(
    "Stable >=0.8:",
    int(
        edge_table_primary[
            "stable_80"
        ].sum()
    )
)

print(
    "Stable >=0.9:",
    int(
        edge_table_primary[
            "stable_90"
        ].sum()
    )
)

print(
    "Stable >=0.95:",
    int(
        edge_table_primary[
            "stable_95"
        ].sum()
    )
)

print(
    "Stable ==1:",
    int(
        edge_table_primary[
            "stable_100"
        ].sum()
    )
)


# ============================================
# Stable edges should be selected in the
# full-data primary fit
# ============================================

stable_not_selected = edge_table_primary[
    edge_table_primary[
        "stable_80"
    ]
    &
    ~edge_table_primary[
        "full_selected"
    ]
]


print(
    "\nStable >=0.8 but not selected "
    "in full-data fit:",
    len(
        stable_not_selected
    )
)


# ============================================
# Verify no self edge survives
# ============================================

self_primary = edge_table_primary[
    edge_table_primary[
        "self_edge"
    ]
]


print(
    "\nPossible self edges:",
    len(
        self_primary
    )
)

print(
    "Selected self edges:",
    int(
        self_primary[
            "full_selected"
        ].sum()
    )
)

print(
    "Stable self edges:",
    int(
        self_primary[
            "stable_80"
        ].sum()
    )
)


# ============================================
# Sign agreement for stable primary edges
# ============================================

stable_primary = edge_table_primary[
    edge_table_primary[
        "stable_80"
    ]
    &
    edge_table_primary[
        "full_selected"
    ]
].copy()


stable_primary[
    "full_sign_frequency"
] = np.where(
    stable_primary[
        "coefficient"
    ].to_numpy()
    > 0,

    stable_primary[
        "positive_frequency"
    ].to_numpy(),

    stable_primary[
        "negative_frequency"
    ].to_numpy()
)


print(
    "\nStable primary sign agreement:"
)

print(
    stable_primary[
        "full_sign_frequency"
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


# ============================================
# Final explicit consistency checks
# ============================================

print(
    "\nA_primary equals corrected matrix:",
    np.array_equal(
        A_primary,
        A_hat_integral_noself
    )
)

print(
    "Selection-frequency matrix equals "
    "corrected stability matrix:",
    np.array_equal(
        selection_freq_primary,
        selection_freq_noself
    )
)

print(
    "All primary coefficients finite:",
    bool(
        np.all(
            np.isfinite(
                A_primary
            )
        )
    )
)

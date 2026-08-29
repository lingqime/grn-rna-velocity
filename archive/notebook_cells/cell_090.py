# ============================================
# Cell 87. Full 426-gene perturbation-held-out
# OOF evaluation with frozen global alpha
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

rng_seed = 2026
n_folds = 5

oof_gene_records = []

for gi, g in enumerate(
    response_genes
):

    valid_mask = valid_row_masks[g]

    dt_g = np.asarray(
        X_final[
            valid_mask, 0
        ],
        dtype=np.float64
    )

    Z_g = np.asarray(
        X_final[
            valid_mask, 1:
        ],
        dtype=np.float64
    ) / dt_g[:, None]

    r_g = np.asarray(
        Y_final[
            valid_mask, gi
        ],
        dtype=np.float64
    ) / dt_g

    groups_g = row_conditions[
        valid_mask
    ]

    # ----------------------------------------
    # Only perturbations are held out.
    # NT remains training reference.
    # ----------------------------------------

    pert_conditions = np.asarray([
        q
        for q in np.unique(groups_g)
        if q != "non-targeting"
    ], dtype=object)

    kf = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=rng_seed
    )

    oof_pred = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    oof_baseline = np.full(
        len(r_g),
        np.nan,
        dtype=np.float64
    )

    fold_nonzero = []

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

        # NT must always be training data
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

        Z_train = Z_g[
            train_fold
        ]

        Z_val = Z_g[
            val_fold
        ]

        r_train = r_g[
            train_fold
        ]

        # ------------------------------------
        # Training-only X scaling
        # ------------------------------------

        x_scaler = StandardScaler()

        Z_train_z = (
            x_scaler.fit_transform(
                Z_train
            )
        )

        Z_val_z = (
            x_scaler.transform(
                Z_val
            )
        )

        # ------------------------------------
        # Training-only y scaling
        # ------------------------------------

        r_mean = np.mean(
            r_train
        )

        r_std = np.std(
            r_train,
            ddof=0
        )

        assert (
            np.isfinite(r_std)
            and r_std > 0
        )

        r_train_z = (
            r_train
            - r_mean
        ) / r_std

        # ------------------------------------
        # Frozen global alpha
        # ------------------------------------

        model = Lasso(
            alpha=global_alpha,
            fit_intercept=True,
            max_iter=20000,
            tol=1e-6,
            selection="cyclic",
        )

        model.fit(
            Z_train_z,
            r_train_z
        )

        pred_z = model.predict(
            Z_val_z
        )

        pred = (
            r_mean
            + r_std * pred_z
        )

        oof_pred[
            val_fold
        ] = pred

        oof_baseline[
            val_fold
        ] = r_mean

        fold_nonzero.append(
            np.count_nonzero(
                model.coef_
            )
        )

    # ----------------------------------------
    # Evaluate perturbation rows only
    # ----------------------------------------

    eval_mask = (
        groups_g
        != "non-targeting"
    )

    assert np.all(
        np.isfinite(
            oof_pred[
                eval_mask
            ]
        )
    )

    assert not np.any(
        np.isfinite(
            oof_pred[
                ~eval_mask
            ]
        )
    )

    y_eval = r_g[
        eval_mask
    ]

    p_eval = oof_pred[
        eval_mask
    ]

    b_eval = oof_baseline[
        eval_mask
    ]

    sse_model = np.sum(
        (
            y_eval
            - p_eval
        ) ** 2
    )

    sse_baseline = np.sum(
        (
            y_eval
            - b_eval
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y_eval
            - np.mean(
                y_eval
            )
        ) ** 2
    )

    oof_r2 = (
        1.0
        - sse_model / ss_tot
    )

    relative_mse = (
        sse_model
        / sse_baseline
    )

    oof_gene_records.append({
        "gene": g,
        "n_eval_rows":
            int(
                eval_mask.sum()
            ),
        "n_eval_conditions":
            len(
                pert_conditions
            ),
        "oof_r2":
            oof_r2,
        "relative_mse":
            relative_mse,
        "mean_fold_nonzero":
            np.mean(
                fold_nonzero
            ),
    })

    if (
        (gi + 1) % 25 == 0
        or gi == 0
        or gi + 1 == len(
            response_genes
        )
    ):
        print(
            f"[{gi + 1:03d}/"
            f"{len(response_genes)}] "
            f"{g} done"
        )


oof_gene_summary = pd.DataFrame(
    oof_gene_records
)


print(
    "\nOOF R^2 summary:"
)

print(
    oof_gene_summary[
        "oof_r2"
    ].describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nRelative MSE summary:"
)

print(
    oof_gene_summary[
        "relative_mse"
    ].describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)


print(
    "\nFraction with OOF R^2 > 0:"
)

print(
    np.mean(
        oof_gene_summary[
            "oof_r2"
        ] > 0
    )
)


print(
    "\nFraction beating fold-specific "
    "mean baseline:"
)

print(
    np.mean(
        oof_gene_summary[
            "relative_mse"
        ] < 1
    )
)


print(
    "\nWorst 10 genes by OOF R^2:"
)

print(
    oof_gene_summary
    .sort_values(
        "oof_r2"
    )
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nBest 10 genes by OOF R^2:"
)

print(
    oof_gene_summary
    .sort_values(
        "oof_r2",
        ascending=False
    )
    .head(10)
    .to_string(
        index=False
    )
)
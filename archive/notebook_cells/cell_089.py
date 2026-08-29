# ============================================
# Cell 86. Strict held-out perturbation
# prediction for TACC3
#
# - frozen global alpha
# - NT always stays in training
# - validation contains perturbations only
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

test_gene = "TACC3"

gi = np.where(
    response_genes == test_gene
)[0][0]

valid_mask = valid_row_masks[
    test_gene
]

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


# --------------------------------------------
# Split perturbation CONDITIONS, not rows.
# NT is never held out.
# --------------------------------------------

pert_conditions = np.asarray([
    q for q in np.unique(groups_g)
    if q != "non-targeting"
], dtype=object)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=2026
)

oof_pred = np.full(
    len(r_g),
    np.nan,
    dtype=np.float64
)

fold_records = []

for fold, (_, val_cond_idx) in enumerate(
    kf.split(pert_conditions),
    start=1
):

    val_conditions = pert_conditions[
        val_cond_idx
    ]

    val_mask_fold = np.isin(
        groups_g,
        val_conditions
    )

    train_mask_fold = (
        ~val_mask_fold
    )

    # NT must always remain in training
    assert np.any(
        groups_g[
            train_mask_fold
        ] == "non-targeting"
    )

    assert not np.any(
        groups_g[
            val_mask_fold
        ] == "non-targeting"
    )

    Z_train = Z_g[
        train_mask_fold
    ]

    Z_val = Z_g[
        val_mask_fold
    ]

    r_train = r_g[
        train_mask_fold
    ]

    r_val = r_g[
        val_mask_fold
    ]

    # ----------------------------------------
    # Training-only predictor scaling
    # ----------------------------------------

    x_scaler = StandardScaler()

    Z_train_z = x_scaler.fit_transform(
        Z_train
    )

    Z_val_z = x_scaler.transform(
        Z_val
    )

    # ----------------------------------------
    # Training-only response scaling
    # ----------------------------------------

    r_mean = np.mean(
        r_train
    )

    r_std = np.std(
        r_train,
        ddof=0
    )

    r_train_z = (
        r_train - r_mean
    ) / r_std

    # ----------------------------------------
    # Frozen global alpha
    # ----------------------------------------

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
        val_mask_fold
    ] = pred

    # ----------------------------------------
    # Baseline:
    # predict training mean response rate
    # ----------------------------------------

    baseline = np.full(
        len(r_val),
        r_mean
    )

    sse_model = np.sum(
        (r_val - pred) ** 2
    )

    sse_baseline = np.sum(
        (r_val - baseline) ** 2
    )

    fold_records.append({
        "fold": fold,
        "n_val_conditions":
            len(val_conditions),
        "n_val_rows":
            int(val_mask_fold.sum()),
        "mse":
            np.mean(
                (r_val - pred) ** 2
            ),
        "baseline_mse":
            np.mean(
                (r_val - baseline) ** 2
            ),
        "relative_mse":
            sse_model
            / sse_baseline,
        "n_nonzero":
            np.count_nonzero(
                model.coef_
            ),
    })


fold_oof_tacc3 = pd.DataFrame(
    fold_records
)

# Only perturbation rows receive OOF predictions
eval_mask = np.isfinite(
    oof_pred
)

r_eval = r_g[
    eval_mask
]

pred_eval = oof_pred[
    eval_mask
]

# Global OOF R^2 relative to the observed
# held-out perturbation response distribution
ss_res = np.sum(
    (r_eval - pred_eval) ** 2
)

ss_tot = np.sum(
    (
        r_eval
        - np.mean(r_eval)
    ) ** 2
)

oof_r2 = (
    1.0
    - ss_res / ss_tot
)

print(
    "Test gene:",
    test_gene
)

print(
    "Held-out perturbation conditions:",
    len(pert_conditions)
)

print(
    "OOF rows:",
    eval_mask.sum()
)

print(
    "NT rows predicted:",
    np.sum(
        eval_mask
        & (
            groups_g
            == "non-targeting"
        )
    )
)

print(
    "\nFold results:"
)

print(
    fold_oof_tacc3.to_string(
        index=False
    )
)

print(
    "\nOverall perturbation-held-out OOF R^2:",
    oof_r2
)

print(
    "Mean fold relative MSE:",
    fold_oof_tacc3[
        "relative_mse"
    ].mean()
)

print(
    "Median fold relative MSE:",
    fold_oof_tacc3[
        "relative_mse"
    ].median()
)

print(
    "\nAll perturbation rows predicted once:",
    np.all(
        np.isfinite(
            oof_pred[
                groups_g
                != "non-targeting"
            ]
        )
    )
)
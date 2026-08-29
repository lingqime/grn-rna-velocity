# ============================================
# Cell 77. Prepare condition-grouped CV
# for one test response gene: TACC3
# ============================================

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

test_gene = "TACC3"

g_idx = np.where(
    response_genes == test_gene
)[0][0]

valid_mask = valid_row_masks[
    test_gene
]

X_test = np.asarray(
    X_final[
        valid_mask, :
    ],
    dtype=np.float64
)

y_test = np.asarray(
    Y_final[
        valid_mask,
        g_idx
    ],
    dtype=np.float64
)

groups_test = row_conditions[
    valid_mask
]

print(
    "Test gene:",
    test_gene
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)

print(
    "Number of condition groups:",
    len(
        np.unique(
            groups_test
        )
    )
)

print(
    "TACC3 condition still present:",
    np.any(
        groups_test == "TACC3"
    )
)


# --------------------------------------------
# 5-fold GroupKFold:
# entire perturbation conditions stay together
# --------------------------------------------

n_splits = 5

gkf = GroupKFold(
    n_splits=n_splits
)

fold_rows = []

fold_indices = []

for fold, (
    train_idx,
    val_idx
) in enumerate(
    gkf.split(
        X_test,
        y_test,
        groups=groups_test
    ),
    start=1
):

    train_conditions = np.unique(
        groups_test[
            train_idx
        ]
    )

    val_conditions = np.unique(
        groups_test[
            val_idx
        ]
    )

    overlap = np.intersect1d(
        train_conditions,
        val_conditions
    )

    fold_indices.append(
        (
            train_idx,
            val_idx
        )
    )

    fold_rows.append({
        "fold": fold,
        "train_rows":
            len(train_idx),
        "val_rows":
            len(val_idx),
        "train_conditions":
            len(train_conditions),
        "val_conditions":
            len(val_conditions),
        "condition_overlap":
            len(overlap),
    })


fold_summary = pd.DataFrame(
    fold_rows
)

print(
    "\nGrouped CV folds:"
)

print(
    fold_summary.to_string(
        index=False
    )
)

print(
    "\nAll folds have zero "
    "condition leakage:",
    (
        fold_summary[
            "condition_overlap"
        ] == 0
    ).all()
)


# --------------------------------------------
# Check every valid row appears exactly once
# as validation data
# --------------------------------------------

validation_counts = np.zeros(
    len(y_test),
    dtype=int
)

for _, val_idx in fold_indices:
    validation_counts[
        val_idx
    ] += 1

print(
    "Every row validated exactly once:",
    np.all(
        validation_counts == 1
    )
)
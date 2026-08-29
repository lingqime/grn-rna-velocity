# ============================================
# Cell 118. Verify OOF missing-entry accounting
#
# Cell 117 intentionally predicts ONLY
# held-out perturbation conditions.
#
# Therefore missing entries should be:
#   1. all NT rows, for all 426 genes
#   2. q = g direct-perturbation rows
#      for the 12 overlap response genes
#
# No model refitting.
# ============================================

import numpy as np
import pandas as pd


# ============================================
# NT rows
# ============================================

nt_row_mask = (
    row_conditions_cv
    == "non-targeting"
)

n_nt_rows = int(
    nt_row_mask.sum()
)

print(
    "NT interval rows:",
    n_nt_rows
)

print(
    "Expected NT missing entries:",
    n_nt_rows * len(response_genes)
)


# ============================================
# Direct q = g rows
# ============================================

direct_missing_records = []

expected_direct_missing = 0


for g in overlap_genes:

    gi = np.where(
        response_arr == g
    )[0][0]

    direct_rows = (
        row_conditions_cv == g
    )

    n_direct_rows = int(
        direct_rows.sum()
    )

    expected_direct_missing += (
        n_direct_rows
    )


    actual_missing_for_gene = (
        ~np.isfinite(
            oof_y[:, gi]
        )
    )


    # Missing rows beyond NT
    actual_direct_only = (
        actual_missing_for_gene
        & ~nt_row_mask
    )


    direct_missing_records.append({
        "gene":
            str(g),

        "n_nt_rows":
            n_nt_rows,

        "n_direct_rows":
            n_direct_rows,

        "expected_total_missing":
            n_nt_rows
            + n_direct_rows,

        "actual_total_missing":
            int(
                actual_missing_for_gene.sum()
            ),

        "actual_nonNT_missing":
            int(
                actual_direct_only.sum()
            ),

        "nonNT_missing_matches_qeqg":
            bool(
                np.array_equal(
                    actual_direct_only,
                    direct_rows
                )
            ),
    })


direct_missing_df = pd.DataFrame(
    direct_missing_records
)


print(
    "\nDirect-intervention missing rows:"
)

print(
    direct_missing_df.to_string(
        index=False
    )
)


# ============================================
# Global expected missing count
# ============================================

expected_total_missing = (
    n_nt_rows
    * len(response_genes)
    + expected_direct_missing
)

actual_total_missing = int(
    np.sum(
        ~np.isfinite(
            oof_y
        )
    )
)


print(
    "\nExpected direct-intervention "
    "missing entries:",
    expected_direct_missing
)

print(
    "Expected TOTAL missing entries:",
    expected_total_missing
)

print(
    "Actual TOTAL missing entries:",
    actual_total_missing
)

print(
    "Exact total match:",
    expected_total_missing
    == actual_total_missing
)


# ============================================
# Check non-overlap genes:
# they should miss ONLY NT rows
# ============================================

non_overlap_genes = [
    g
    for g in response_arr
    if g not in set(overlap_genes)
]


non_overlap_ok = True

for g in non_overlap_genes:

    gi = np.where(
        response_arr == g
    )[0][0]

    missing = (
        ~np.isfinite(
            oof_y[:, gi]
        )
    )

    if not np.array_equal(
        missing,
        nt_row_mask
    ):
        non_overlap_ok = False
        break


print(
    "\nAll 414 non-overlap genes "
    "miss ONLY NT rows:",
    non_overlap_ok
)


# ============================================
# Fold-ID interpretation
# ============================================

print(
    "\nRows with fold_id = -1:",
    int(
        np.sum(
            oof_fold_id == -1
        )
    )
)

print(
    "Those rows are exactly NT rows:",
    bool(
        np.array_equal(
            oof_fold_id == -1,
            nt_row_mask
        )
    )
)
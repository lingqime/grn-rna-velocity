# ============================================
# Cell 115. Save corrected primary GRN
# checkpoint
#
# This is the canonical primary network
# after explicit self-predictor exclusion.
# ============================================

import numpy as np
import pandas as pd
import os


primary_npz_path = (
    "/home/featurize/work/project1/"
    "replogle_primary_exact_integral_grn.npz"
)

primary_edge_csv_path = (
    "/home/featurize/work/project1/"
    "replogle_primary_edge_table.csv.gz"
)


# ============================================
# Save numerical checkpoint
# ============================================

np.savez_compressed(
    primary_npz_path,

    # Network
    A_primary=A_primary,
    c_primary=c_primary,

    # Stability
    selection_freq_primary=
        selection_freq_primary,

    positive_freq_primary=
        positive_freq_primary,

    negative_freq_primary=
        negative_freq_primary,

    sign_consistency_primary=
        sign_consistency_primary,

    # Labels
    response_genes=np.asarray(
        response_genes,
        dtype=str
    ),

    regulator_genes=np.asarray(
        regulator_genes_vc,
        dtype=str
    ),

    conditions=np.asarray(
        final_conditions_vc,
        dtype=str
    ),

    # Estimator metadata
    global_alpha_integral=np.asarray(
        global_alpha_integral
    ),

    omega_nt=np.asarray(
        omega_nt
    ),

    time_scale=np.asarray(
        time_scale
    ),

    # Identifiability summary
    full_design_rank=np.asarray(
        152
    ),

    full_design_lambda_min=np.asarray(
        0.00267429148
    ),

    full_design_condition_number=np.asarray(
        52269.9004
    ),

    # Validation summary
    validation_incremental_r2_mean=
        np.asarray(
            0.111698
        ),

    validation_incremental_r2_median=
        np.asarray(
            0.092643
        ),

    validation_fraction_positive=
        np.asarray(
            0.8936708861
        ),
)


# ============================================
# Save canonical edge table
# ============================================

edge_table_primary.to_csv(
    primary_edge_csv_path,
    index=False,
    compression="gzip"
)


# ============================================
# Reload checks
# ============================================

chk = np.load(
    primary_npz_path,
    allow_pickle=False
)


print(
    "NPZ saved:",
    primary_npz_path
)

print(
    "NPZ size MB:",
    os.path.getsize(
        primary_npz_path
    ) / 1024**2
)

print(
    "\nCSV saved:",
    primary_edge_csv_path
)

print(
    "CSV size MB:",
    os.path.getsize(
        primary_edge_csv_path
    ) / 1024**2
)


print(
    "\nA exact after reload:",
    np.array_equal(
        chk["A_primary"],
        A_primary
    )
)

print(
    "Selection frequency exact:",
    np.array_equal(
        chk[
            "selection_freq_primary"
        ],
        selection_freq_primary
    )
)


edge_reload = pd.read_csv(
    primary_edge_csv_path
)


print(
    "\nEdge-table reload shape:",
    edge_reload.shape
)

print(
    "Full selected after reload:",
    int(
        edge_reload[
            "full_selected"
        ].sum()
    )
)

print(
    "Stable >=0.8 after reload:",
    int(
        edge_reload[
            "stable_80"
        ].sum()
    )
)

print(
    "Selected self edges after reload:",
    int(
        edge_reload.loc[
            edge_reload[
                "self_edge"
            ],
            "full_selected"
        ].sum()
    )
)
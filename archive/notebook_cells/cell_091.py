# ============================================
# Cell 88. Save reconstruction + OOF checkpoint
# ============================================

import numpy as np
import pandas as pd
import os

checkpoint_path = (
    "/home/featurize/work/project1/"
    "replogle_grn_reconstruction_checkpoint.npz"
)

np.savez_compressed(
    checkpoint_path,

    # Final reconstructed GRN
    A_hat=A_hat,
    c_hat=c_hat,

    response_genes=np.asarray(
        response_genes,
        dtype=str
    ),

    regulator_genes=np.asarray(
        regulator_genes_vc,
        dtype=str
    ),

    global_alpha=np.asarray(
        global_alpha
    ),

    # Final integral system
    X_final=X_final,
    Y_final=Y_final,

    # OOF summary
    oof_gene_names=np.asarray(
        oof_gene_summary["gene"],
        dtype=str
    ),

    oof_r2=np.asarray(
        oof_gene_summary["oof_r2"],
        dtype=float
    ),

    relative_mse=np.asarray(
        oof_gene_summary["relative_mse"],
        dtype=float
    ),

    mean_fold_nonzero=np.asarray(
        oof_gene_summary[
            "mean_fold_nonzero"
        ],
        dtype=float
    ),
)

print(
    "Saved:",
    checkpoint_path
)

print(
    "Exists:",
    os.path.exists(
        checkpoint_path
    )
)

print(
    "Size MB:",
    os.path.getsize(
        checkpoint_path
    ) / 1024**2
)


# --------------------------------------------
# Reload immediately and verify critical arrays
# --------------------------------------------

check = np.load(
    checkpoint_path,
    allow_pickle=False
)

print(
    "\nSaved keys:"
)

print(
    check.files
)

print(
    "\nA_hat exact:",
    np.array_equal(
        check["A_hat"],
        A_hat
    )
)

print(
    "c_hat exact:",
    np.array_equal(
        check["c_hat"],
        c_hat
    )
)

print(
    "OOF R2 exact:",
    np.array_equal(
        check["oof_r2"],
        oof_gene_summary[
            "oof_r2"
        ].to_numpy()
    )
)

print(
    "Global alpha exact:",
    float(
        check["global_alpha"]
    ) == global_alpha
)
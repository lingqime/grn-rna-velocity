# ============================================
# Cell 99. Save FINAL exact-integral GRN
# + phase-residual validation checkpoint
# ============================================

import numpy as np
import os

final_checkpoint_path = (
    "/home/featurize/work/project1/"
    "replogle_final_exact_integral_grn.npz"
)

np.savez_compressed(
    final_checkpoint_path,

    # ----------------------------------------
    # Final Algorithm 5.1 reconstruction
    # ----------------------------------------

    A_hat_integral=A_hat_integral,
    c_hat_integral=c_hat_integral,

    response_genes=np.asarray(
        response_genes,
        dtype=str
    ),

    regulator_genes=np.asarray(
        regulator_genes_vc,
        dtype=str
    ),

    global_alpha_integral=np.asarray(
        global_alpha_integral
    ),

    # ----------------------------------------
    # Exact integral design / response
    # ----------------------------------------

    X_final=X_final,
    Y_final=Y_final,

    # ----------------------------------------
    # Final fit diagnostics
    # ----------------------------------------

    integral_n_nonzero=np.asarray(
        integral_fit_summary[
            "n_nonzero"
        ],
        dtype=int
    ),

    integral_skill=np.asarray(
        integral_fit_summary[
            "skill_vs_intercept"
        ],
        dtype=float
    ),

    integral_mse=np.asarray(
        integral_fit_summary[
            "integral_mse"
        ],
        dtype=float
    ),

    # ----------------------------------------
    # Strict phase-residualized validation
    # ----------------------------------------

    global_alpha_resid=np.asarray(
        global_alpha_resid
    ),

    validation_genes=np.asarray(
        final_resid_oof_summary[
            "gene"
        ],
        dtype=str
    ),

    validation_incremental_r2=np.asarray(
        final_resid_oof_summary[
            "incremental_r2"
        ],
        dtype=float
    ),

    validation_relative_mse=np.asarray(
        final_resid_oof_summary[
            "relative_mse_vs_phase"
        ],
        dtype=float
    ),

    validation_mean_nonzero=np.asarray(
        final_resid_oof_summary[
            "mean_fold_nonzero"
        ],
        dtype=float
    ),

    validation_calibration_flag=np.asarray(
        final_resid_oof_summary[
            "used_for_alpha_calibration"
        ],
        dtype=bool
    ),
)


print(
    "Saved:",
    final_checkpoint_path
)

print(
    "Exists:",
    os.path.exists(
        final_checkpoint_path
    )
)

print(
    "Size MB:",
    os.path.getsize(
        final_checkpoint_path
    ) / 1024**2
)


# --------------------------------------------
# Immediate reload verification
# --------------------------------------------

check_final = np.load(
    final_checkpoint_path,
    allow_pickle=False
)

print(
    "\nSaved keys:"
)

print(
    check_final.files
)

print(
    "\nA exact:",
    np.array_equal(
        check_final[
            "A_hat_integral"
        ],
        A_hat_integral
    )
)

print(
    "c exact:",
    np.array_equal(
        check_final[
            "c_hat_integral"
        ],
        c_hat_integral
    )
)

print(
    "alpha exact:",
    float(
        check_final[
            "global_alpha_integral"
        ]
    )
    == global_alpha_integral
)

print(
    "validation exact:",
    np.array_equal(
        check_final[
            "validation_incremental_r2"
        ],
        final_resid_oof_summary[
            "incremental_r2"
        ].to_numpy()
    )
)
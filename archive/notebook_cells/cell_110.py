# ============================================
# Cell 105. Save full edge-stability results
# into final checkpoint v3
# ============================================

import numpy as np
import os


final_checkpoint_v3_path = (
    "/home/featurize/work/project1/"
    "replogle_final_exact_integral_grn_v3.npz"
)


np.savez_compressed(
    final_checkpoint_v3_path,

    # ========================================
    # Final exact-integral GRN
    # ========================================

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

    X_final=X_final,
    Y_final=Y_final,

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


    # ========================================
    # Strict phase-residualized validation
    # ========================================

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


    # ========================================
    # Corrected identifiability analysis
    # ========================================

    ident_col_scale=ident_col_scale,

    ident_full_rank=np.asarray(
        rank_scaled
    ),

    ident_full_lambda_min=np.asarray(
        lambda_min_ident
    ),

    ident_full_lambda_max=np.asarray(
        lambda_max_ident
    ),

    ident_full_condition_number=np.asarray(
        lambda_max_ident
        / lambda_min_ident
    ),

    ident_first_full_rank=first_full_rank,

    ident_k_grid=k_grid_ident,

    ident_path_k=np.asarray(
        ident_path_df[
            "n_perturbations"
        ],
        dtype=int
    ),

    ident_path_rep=np.asarray(
        ident_path_df[
            "rep"
        ],
        dtype=int
    ),

    ident_path_rank=np.asarray(
        ident_path_df[
            "rank"
        ],
        dtype=int
    ),

    ident_path_full_rank=np.asarray(
        ident_path_df[
            "full_rank"
        ],
        dtype=bool
    ),

    ident_path_lambda_min=np.asarray(
        ident_path_df[
            "lambda_min"
        ],
        dtype=float
    ),

    ident_path_condition_number=np.asarray(
        ident_path_df[
            "condition_number"
        ],
        dtype=float
    ),


    # ========================================
    # Full stability analysis
    # ========================================

    stability_n_reps=np.asarray(
        n_stability_reps
    ),

    stability_fraction=np.asarray(
        stability_fraction
    ),

    stability_selection_count=selection_count,

    stability_positive_count=positive_count,

    stability_negative_count=negative_count,

    stability_selection_freq=selection_freq,

    stability_positive_freq=positive_freq,

    stability_negative_freq=negative_freq,

    stability_sign_consistency=sign_consistency,

    stability_edge_count_by_rep=edge_count_by_rep,
)


print(
    "Saved:",
    final_checkpoint_v3_path
)

print(
    "Exists:",
    os.path.exists(
        final_checkpoint_v3_path
    )
)

print(
    "Size MB:",
    os.path.getsize(
        final_checkpoint_v3_path
    ) / 1024**2
)


# --------------------------------------------
# Reload checks
# --------------------------------------------

check_v3 = np.load(
    final_checkpoint_v3_path,
    allow_pickle=False
)

print(
    "\nA exact:",
    np.array_equal(
        check_v3[
            "A_hat_integral"
        ],
        A_hat_integral
    )
)

print(
    "Selection frequency exact:",
    np.array_equal(
        check_v3[
            "stability_selection_freq"
        ],
        selection_freq
    )
)

print(
    "Positive frequency exact:",
    np.array_equal(
        check_v3[
            "stability_positive_freq"
        ],
        positive_freq
    )
)

print(
    "Negative frequency exact:",
    np.array_equal(
        check_v3[
            "stability_negative_freq"
        ],
        negative_freq
    )
)

print(
    "Identifiability exact:",
    np.array_equal(
        check_v3[
            "ident_first_full_rank"
        ],
        first_full_rank
    )
)


# --------------------------------------------
# Key stability invariants
# --------------------------------------------

stable80_check = (
    check_v3[
        "stability_selection_freq"
    ] >= 0.8
)

full_selected_check = (
    check_v3[
        "A_hat_integral"
    ] != 0
)

print(
    "\nStable >=0.8 edges:",
    int(
        stable80_check.sum()
    )
)

print(
    "All stable edges in full network:",
    bool(
        np.all(
            full_selected_check[
                stable80_check
            ]
        )
    )
)

print(
    "Fraction full-network edges stable >=0.8:",
    float(
        stable80_check[
            full_selected_check
        ].mean()
    )
)
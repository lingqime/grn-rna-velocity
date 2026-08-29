# ============================================
# Cell 125. FINAL kernel-shutdown checkpoint
#
# Save all manuscript-relevant final objects.
# NO fitting. NO new analysis.
#
# After this cell succeeds, the kernel can
# be safely shut down.
# ============================================

import numpy as np
import pandas as pd
import os


out_dir = "/home/featurize/work/project1"

npz_path = os.path.join(
    out_dir,
    "replogle_manuscript_final_checkpoint.npz"
)

validation_path = os.path.join(
    out_dir,
    "replogle_manuscript_validation.csv"
)

condition_decomp_path = os.path.join(
    out_dir,
    "replogle_condition_decomposition.csv"
)

regulator_decomp_path = os.path.join(
    out_dir,
    "replogle_regulator_decomposition.csv"
)


# --------------------------------------------
# Main numerical objects
# --------------------------------------------

np.savez_compressed(
    npz_path,

    # gene / condition labels
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
    perturbations=np.asarray(
        final_perturbations_vc,
        dtype=str
    ),

    # exact integral design / response
    X_final=np.asarray(
        X_final,
        dtype=np.float64
    ),
    Y_final=np.asarray(
        Y_final,
        dtype=np.float64
    ),

    # corrected primary GRN
    A_primary=np.asarray(
        A_primary,
        dtype=np.float64
    ),
    c_primary=np.asarray(
        c_primary,
        dtype=np.float64
    ),

    # stability
    selection_freq_primary=np.asarray(
        selection_freq_primary,
        dtype=np.float64
    ),
    positive_freq_primary=np.asarray(
        positive_freq_primary,
        dtype=np.float64
    ),
    negative_freq_primary=np.asarray(
        negative_freq_primary,
        dtype=np.float64
    ),
    sign_consistency_primary=np.asarray(
        sign_consistency_primary,
        dtype=np.float64
    ),

    # held-out predictions
    oof_y=np.asarray(
        oof_y,
        dtype=np.float64
    ),
    oof_pred_grn=np.asarray(
        oof_pred_grn,
        dtype=np.float64
    ),
    oof_pred_phase=np.asarray(
        oof_pred_phase,
        dtype=np.float64
    ),
    oof_fold_id=np.asarray(
        oof_fold_id,
        dtype=np.int64
    ),

    # phase / regulator decomposition
    X_shared=np.asarray(
        X_shared,
        dtype=np.float64
    ),
    X_deviation=np.asarray(
        X_deviation,
        dtype=np.float64
    ),

    # important scalar parameters
    global_alpha_integral=np.asarray(
        global_alpha_integral
    ),
    omega_nt=np.asarray(
        omega_nt
    ),
    time_scale=np.asarray(
        time_scale
    ),
    beta_tilde=np.asarray(
        beta_tilde,
        dtype=np.float64
    ),
)


# --------------------------------------------
# Tables useful for manuscript figures
# --------------------------------------------

primary_validation.to_csv(
    validation_path,
    index=False
)

condition_decomposition.to_csv(
    condition_decomp_path,
    index=False
)

regulator_decomposition.to_csv(
    regulator_decomp_path,
    index=False
)


# Cell 124 summary should already exist.
summary_path = os.path.join(
    out_dir,
    "replogle_numerical_experiment_summary.csv"
)


# --------------------------------------------
# Verify files
# --------------------------------------------

paths = [
    npz_path,
    validation_path,
    condition_decomp_path,
    regulator_decomp_path,
    summary_path,
    os.path.join(
        out_dir,
        "replogle_primary_exact_integral_grn.npz"
    ),
    os.path.join(
        out_dir,
        "replogle_primary_edge_table.csv.gz"
    ),
]

print("FINAL FILE CHECK")
print("=" * 60)

for path in paths:

    exists = os.path.exists(path)

    size_mb = (
        os.path.getsize(path)
        / 1024**2
        if exists
        else np.nan
    )

    print(
        f"{os.path.basename(path):50s}",
        f"exists={exists}",
        f"size={size_mb:.3f} MB"
    )


# --------------------------------------------
# Reload the master checkpoint
# --------------------------------------------

check = np.load(
    npz_path,
    allow_pickle=False
)

print("\nMASTER CHECKPOINT")

print(
    "keys:",
    check.files
)

print(
    "X:",
    check["X_final"].shape
)

print(
    "Y:",
    check["Y_final"].shape
)

print(
    "A:",
    check["A_primary"].shape
)

print(
    "selection frequency:",
    check[
        "selection_freq_primary"
    ].shape
)

print(
    "OOF predictions:",
    check[
        "oof_pred_grn"
    ].shape
)

print(
    "shared/deviation:",
    check["X_shared"].shape,
    check["X_deviation"].shape
)


# --------------------------------------------
# Critical integrity checks
# --------------------------------------------

assert np.array_equal(
    check["A_primary"],
    A_primary
)

assert np.array_equal(
    check["selection_freq_primary"],
    selection_freq_primary
)

assert np.allclose(
    check["X_final"],
    X_final,
    equal_nan=True
)

assert np.allclose(
    check["Y_final"],
    Y_final,
    equal_nan=True
)

assert np.allclose(
    check["oof_pred_grn"],
    oof_pred_grn,
    equal_nan=True
)

assert np.allclose(
    check["oof_pred_phase"],
    oof_pred_phase,
    equal_nan=True
)

assert np.allclose(
    check["X_shared"] + check["X_deviation"],
    X_final[:, 1:],
    rtol=1e-12,
    atol=1e-12
)


print(
    "\nAll integrity checks passed."
)

print(
    "\nSAFE TO SHUT DOWN KERNEL."
)
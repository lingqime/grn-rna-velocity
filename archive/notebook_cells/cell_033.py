# ============================================
# Cell 33b. Recover final perturbation list
# from saved GRN checkpoint
# ============================================

import numpy as np

checkpoint_path = base_dir / "replogle_grn_checkpoint.npz"

ckpt = np.load(
    checkpoint_path,
    allow_pickle=True
)

print("Checkpoint keys:")
print(ckpt.files)

for key in ckpt.files:
    x = ckpt[key]

    try:
        shape = x.shape
    except Exception:
        shape = None

    print(f"{key:30s} shape={shape}")
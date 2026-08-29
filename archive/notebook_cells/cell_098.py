# ============================================
# Cell 94b. Restore the fixed 31-gene
# calibration panel from Cell 82
# ============================================

import numpy as np

calibration_genes = np.asarray([
    "CDKN1B",
    "MLH1",
    "THOC1",
    "BCL7C",
    "EXOC6",
    "MAP3K20",
    "UCHL5",
    "CAMK2G",
    "TXLNG",
    "CSNK2A2",
    "KIF22",
    "DSN1",
    "FANCI",
    "DNA2",
    "NCAPH",
    "PPP6C",
    "NCAPD2",
    "RMI1",
    "KMT2E",
    "VPS4B",
    "TACC3",
    "CACUL1",
    "TFDP2",
    "CDKN3",
    "CSNK1A1",
    "ERCC6",
    "UBE2B",
    "EXT1",
    "HSP90AB1",
    "PLK2",
    "RPS3",
], dtype=object)


print(
    "Calibration genes:",
    len(calibration_genes)
)

print(
    calibration_genes
)


# Verify all are still in the 426 response set
missing = [
    g
    for g in calibration_genes
    if g not in set(response_genes)
]

print(
    "\nMissing from response_genes:",
    missing
)

assert len(calibration_genes) == 31
assert len(missing) == 0

print(
    "\nFixed calibration panel restored."
)
"""Metrics used in the held-out RPE1 validation."""

from __future__ import annotations

import numpy as np


def incremental_r2_vs_phase(y, pred_grn, pred_phase):
    """Return 1 - SSE_GRN / SSE_phase on finite common entries."""
    y = np.asarray(y, dtype=float)
    pred_grn = np.asarray(pred_grn, dtype=float)
    pred_phase = np.asarray(pred_phase, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred_grn) & np.isfinite(pred_phase)
    if not np.any(mask):
        return np.nan
    sse_grn = np.sum((y[mask] - pred_grn[mask]) ** 2)
    sse_phase = np.sum((y[mask] - pred_phase[mask]) ** 2)
    if sse_phase <= 0:
        return np.nan
    return float(1.0 - sse_grn / sse_phase)

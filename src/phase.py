"""Phase-grid helpers matching the final RPE1 analysis."""

from __future__ import annotations

import numpy as np


def phase_grid(n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


def one_harmonic_design(phase_centers):
    """Return [1, sin(2*pi*t), cos(2*pi*t)] design."""
    phase_centers = np.asarray(phase_centers, dtype=float)
    phi = 2.0 * np.pi * phase_centers
    return np.column_stack([np.ones_like(phi), np.sin(phi), np.cos(phi)])

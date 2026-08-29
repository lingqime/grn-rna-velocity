"""Core exact-integral sparse regression utilities used in the RPE1 analysis."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso


def fwl_rms_lasso(
    d: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    max_iter: int = 20000,
    tol: float = 1e-6,
):
    """Fit y = c*d + X@A with c unpenalized and A L1-penalized.

    This reproduces the final notebook convention:
    Frisch--Waugh--Lovell projection on the basal/dt column, no ordinary
    mean-centering, RMS scaling, and sklearn Lasso(fit_intercept=False).
    """
    d = np.asarray(d, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    dd = float(d @ d)
    if not np.isfinite(dd) or dd <= 0:
        raise ValueError("Basal/dt column has non-positive squared norm.")

    coef_d_X = (d @ X) / dd
    coef_d_y = (d @ y) / dd

    Xp = X - d[:, None] * coef_d_X[None, :]
    yp = y - d * coef_d_y

    x_scale = np.sqrt(np.mean(Xp**2, axis=0))
    y_scale = float(np.sqrt(np.mean(yp**2)))

    if np.any(~np.isfinite(x_scale)) or np.any(x_scale <= 0):
        raise ValueError("Invalid predictor RMS scale.")
    if not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError("Invalid response RMS scale.")

    Z = Xp / x_scale[None, :]
    yz = yp / y_scale

    model = Lasso(
        alpha=float(alpha),
        fit_intercept=False,
        max_iter=max_iter,
        tol=tol,
        selection="cyclic",
    )
    model.fit(Z, yz)

    A = y_scale * model.coef_ / x_scale
    c = float(d @ (y - X @ A) / dd)
    return A, c

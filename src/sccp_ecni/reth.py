"""Reth Fa–D' quantification for SCCP mixtures.

Reth, Zencak and Oehme (J. Chromatogr. A 2005, 1081, 225–231) showed that the
total ECNI response factor of an SCCP mixture is linear in its chlorine
content. Both the calibration mixtures and the sample use the chlorine content
calculated from the same relative areas, so the response bias of ECNI is on
both axes of the line.

Symbols used here:

* ``Fa_i`` — relative area of formula group ``i``. With an internal standard,
  ``Fa_i = A_i / A_ISTD``; otherwise ``Fa_i = A_i``. ``Fa = sum(Fa_i)``.
* ``D_i`` — chlorine mass percent of formula group ``i``.
* ``D'`` — area-weighted chlorine mass percent,
  ``D' = sum(Fa_i * D_i) / sum(Fa_i)``.
* ``RF = Fa / c = a * D' + b`` fitted across calibration mixtures.
* Sample concentration ``c = Fa / (a * D' + b)``.

``D'`` is not the manufacturer chlorine content from elemental analysis. That
declared value is stored beside the result so the ECNI shift can be seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sccp_ecni.ions import N_IONS, chlorine_percents


@dataclass(frozen=True)
class CalibrationFit:
    """Straight line ``RF = slope * D' + intercept``."""

    slope: float
    intercept: float
    r_squared: float
    d_prime: np.ndarray
    response_factor: np.ndarray

    def predict(self, d_prime_value: float) -> float:
        return self.slope * float(d_prime_value) + self.intercept


def fractional_areas(areas: np.ndarray, istd_area: float | None = None) -> np.ndarray:
    """Build the Fa vector from raw ion areas."""
    values = np.asarray(areas, dtype=float).reshape(-1)
    if values.shape != (N_IONS,):
        raise ValueError(f"expected {N_IONS} areas, got {values.size}")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("areas must be finite and non-negative")
    if istd_area is None:
        return values.copy()
    if not np.isfinite(istd_area) or istd_area <= 0:
        raise ValueError("internal-standard area must be positive and finite")
    return values / float(istd_area)


def d_prime(fa: np.ndarray, group_chlorine_percent: np.ndarray | None = None) -> float:
    """Area-weighted chlorine mass percent, D'."""
    weights = np.asarray(fa, dtype=float).reshape(-1)
    if group_chlorine_percent is None:
        chlorine = np.asarray(chlorine_percents(), dtype=float)
    else:
        chlorine = np.asarray(group_chlorine_percent, dtype=float).reshape(-1)
    if weights.shape != chlorine.shape:
        raise ValueError("Fa and the per-group chlorine contents must have the same length")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Fa must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("total relative area Fa is zero, so D' is undefined")
    return float(weights @ chlorine / total)


def response_factor(fa: np.ndarray, concentration: float) -> float:
    """Total response factor ``RF = Fa / concentration``."""
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be positive and finite")
    weights = np.asarray(fa, dtype=float).reshape(-1)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Fa must be finite and non-negative")
    return float(weights.sum() / concentration)


def fit_reth(d_prime_values: np.ndarray, response_factors: np.ndarray) -> CalibrationFit:
    """Fit ``RF = a * D' + b`` to one point per calibration mixture.

    At least two mixtures with different D' are required. The slope may be
    positive, as in the original Reth plots, and the intercept is often negative.
    The line is only used inside the chlorine window of the standards.
    """
    x = np.asarray(d_prime_values, dtype=float).reshape(-1)
    y = np.asarray(response_factors, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("D' and RF must have the same length")
    if x.size < 2:
        raise ValueError("Reth calibration needs at least two standards with different D'")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("D' and RF must be finite")
    if np.unique(np.round(x, decimals=6)).size < 2:
        raise ValueError("Reth calibration needs standards with different calculated D' values")
    if np.any(y <= 0):
        raise ValueError("response factors must be positive")
    slope, intercept = (float(value) for value in np.polyfit(x, y, 1))
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return CalibrationFit(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        d_prime=x.copy(),
        response_factor=y.copy(),
    )


def quantify_reth(fa: np.ndarray, d_prime_value: float, fit: CalibrationFit) -> float:
    """Convert a sample Fa vector and its D' into a concentration on the Reth line."""
    rf = fit.predict(d_prime_value)
    if rf <= 0:
        raise ValueError(
            f"Reth response factor is {rf:.4g} at D' = {d_prime_value:.2f}. "
            "That chlorine content is outside the usable calibration line."
        )
    total = float(np.asarray(fa, dtype=float).sum())
    if total < 0:
        raise ValueError("Fa must be non-negative")
    return total / rf

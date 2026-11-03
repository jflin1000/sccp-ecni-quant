"""NNLS pattern deconvolution for SCCP formula-group profiles.

A measured 24-ion response is expressed as a non-negative linear combination of
calibration-mixture responses. Coefficients are not forced to sum to one: every
available standard can contribute, which is the same constraint set CPxplorer
uses for batch quantification. The underlying reconstruction is the pattern
deconvolution described by Bogdal and implemented for CP mixtures by Perkons
(CP-Crawler), solved here with non-negative least squares.

When each standard column is the response per concentration unit, the
coefficients are concentrations in the same units and their sum is the
deconvolved SCCP concentration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


@dataclass(frozen=True)
class DeconvResult:
    """NNLS solution for one sample pattern."""

    coefficients: np.ndarray
    reconstructed: np.ndarray
    residual_norm: float
    rmse: float
    r_squared: float


def pattern_r_squared(measured: np.ndarray, reconstructed: np.ndarray) -> float:
    """Coefficient of determination between a measured pattern and its reconstruction.

    This is the goodness-of-pattern-fit reported alongside CP deconvolution.
    A flat measured pattern has no variance; a perfect match then scores 1.
    """
    observed = np.asarray(measured, dtype=float).reshape(-1)
    predicted = np.asarray(reconstructed, dtype=float).reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError("measured and reconstructed patterns must have the same length")
    residual = observed - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 1e-18 else 0.0
    return 1.0 - ss_res / ss_tot


def nnls_deconv(sample: np.ndarray, standards: np.ndarray) -> DeconvResult:
    """Solve ``min ||standards @ c - sample||_2`` with ``c >= 0``.

    Args:
        sample: Response vector, shape ``(n_ions,)``.
        standards: Matrix of calibration responses, shape ``(n_ions, n_standards)``.
            Columns should be responses per concentration unit when the
            coefficients are to be read as concentrations.

    Returns:
        Coefficients, the reconstructed response, and fit statistics.
    """
    observed = np.asarray(sample, dtype=float).reshape(-1)
    matrix = np.asarray(standards, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("standards must be a 2-D matrix of shape (n_ions, n_standards)")
    if matrix.shape[0] != observed.shape[0]:
        raise ValueError("sample length must match the ion axis of standards")
    if matrix.shape[1] == 0:
        raise ValueError("at least one standard is required")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(matrix)):
        raise ValueError("sample and standards must be finite")
    if np.any(observed < 0) or np.any(matrix < 0):
        raise ValueError("sample and standards must be non-negative")
    coefficients, _rnorm = nnls(matrix, observed)
    reconstructed = matrix @ coefficients
    residual = observed - reconstructed
    rmse = float(np.sqrt(np.mean(residual**2)))
    return DeconvResult(
        coefficients=np.asarray(coefficients, dtype=float),
        reconstructed=reconstructed,
        residual_norm=float(np.linalg.norm(residual)),
        rmse=rmse,
        r_squared=pattern_r_squared(observed, reconstructed),
    )

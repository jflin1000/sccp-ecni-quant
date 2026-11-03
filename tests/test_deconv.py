"""Non-negative least squares pattern reconstruction."""

import numpy as np
import pytest

from sccp_ecni.deconv import nnls_deconv


def test_nnls_recovers_known_mixture() -> None:
    standards = np.array(
        [
            [1.0, 0.0, 0.2],
            [0.2, 1.0, 0.0],
            [0.0, 0.4, 1.0],
            [0.3, 0.1, 0.5],
            [0.0, 0.2, 0.2],
        ]
    )
    coefficients = np.array([1.5, 2.5, 0.5])
    sample = standards @ coefficients
    result = nnls_deconv(sample, standards)
    assert result.coefficients == pytest.approx(coefficients, abs=1e-8)
    assert result.reconstructed == pytest.approx(sample, abs=1e-8)
    assert result.r_squared == pytest.approx(1.0, abs=1e-10)


def test_nnls_coefficients_stay_non_negative() -> None:
    standards = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.2]])
    # A zero in the first channel cannot be fit by a positive weight on column 1.
    sample = np.array([0.0, 1.0, 0.2])
    result = nnls_deconv(sample, standards)
    assert np.all(result.coefficients >= -1e-12)
    assert result.coefficients[0] == pytest.approx(0.0, abs=1e-8)
    assert result.coefficients[1] > 0
    assert result.residual_norm < np.linalg.norm(sample)

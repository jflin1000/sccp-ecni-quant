"""Reth Fa–D' calibration."""

import numpy as np
import pytest

from sccp_ecni.ions import SCCP_GROUPS
from sccp_ecni.reth import d_prime, fit_reth, fractional_areas, quantify_reth, response_factor
from sccp_ecni.synthetic import RETH_INTERCEPT, RETH_SLOPE


def test_fa_and_d_prime() -> None:
    areas = np.zeros(24)
    areas[0] = 5.0
    areas[1] = 15.0
    fa = fractional_areas(areas, istd_area=2.0)
    assert fa[0] == pytest.approx(2.5)
    assert fa[1] == pytest.approx(7.5)
    assert float(fa.sum()) == pytest.approx(10.0)
    # D' is the area-weighted mean of the formula-group chlorine contents.
    expected = (2.5 * SCCP_GROUPS[0].chlorine_percent + 7.5 * SCCP_GROUPS[1].chlorine_percent) / 10.0
    assert d_prime(fa) == pytest.approx(expected)
    assert response_factor(fa, 2.0) == pytest.approx(5.0)


def test_line_quantifies_a_point_on_the_line() -> None:
    fit = fit_reth([58.0, 62.0, 68.0], [140.0, 220.0, 340.0])
    assert fit.slope == pytest.approx(RETH_SLOPE)
    assert fit.intercept == pytest.approx(RETH_INTERCEPT)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    fa = np.full(24, 220.0 * 4.0 / 24.0)
    # Total Fa = RF(62) * 4, so the concentration on the line is 4.
    assert float(fa.sum()) == pytest.approx(880.0)
    assert quantify_reth(fa, 62.0, fit) == pytest.approx(4.0)


def test_zero_area_and_flat_chlorine_content_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero"):
        d_prime(np.zeros(24))
    with pytest.raises(ValueError, match="different"):
        fit_reth([60.0, 60.0], [100.0, 110.0])

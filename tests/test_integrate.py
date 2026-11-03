"""Chromatogram integration and the vendor-format TODO."""

from pathlib import Path

import numpy as np
import pytest

from sccp_ecni.integrate import (
    VendorFormatNotImplemented,
    header_to_channel,
    integrate_csv,
    integrate_file,
    integrate_trace,
)
from sccp_ecni.ions import CONGENER_KEYS, SCCP_GROUPS
from sccp_ecni.synthetic import write_chromatogram_csv


def test_gaussian_area_matches_analytical() -> None:
    rt = np.linspace(-20.0, 20.0, 2001)
    sigma = 1.0
    signal = np.exp(-0.5 * (rt / sigma) ** 2)
    area = integrate_trace(rt, signal)
    expected = sigma * np.sqrt(2.0 * np.pi)
    assert area == pytest.approx(expected, rel=1e-3)


def test_csv_round_trip_recovers_areas(tmp_path: Path) -> None:
    target = np.linspace(10.0, 33.0, 24)
    path = tmp_path / "sample__a__rep-1.csv"
    write_chromatogram_csv(path, target, istd_area=12.5)
    integrated = integrate_csv(path)
    assert integrated.areas == pytest.approx(target, rel=1e-6, abs=1e-6)
    assert integrated.istd_area == pytest.approx(12.5, rel=1e-6)


def test_mz_and_formula_headers_and_semicolon(tmp_path: Path) -> None:
    rt = [0.0, 1.0, 2.0, 3.0, 4.0]
    intensities = [0.0, 0.0, 10.0, 0.0, 0.0]
    headers = ["rt_min"] + [str(group.mz) for group in SCCP_GROUPS]
    headers[1] = SCCP_GROUPS[0].formula
    lines = [";".join(headers)]
    for time, height in zip(rt, intensities, strict=True):
        lines.append(";".join([f"{time}"] + [f"{height}"] * 24))
    path = tmp_path / "wide.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    integrated = integrate_csv(path)
    assert integrated.areas == pytest.approx([10.0] * 24)
    assert integrated.istd_area is None


def test_missing_ion_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "partial.csv"
    path.write_text("rt_min,C10Cl5\n0,0\n1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing ion"):
        integrate_csv(path)


def test_header_aliases() -> None:
    assert header_to_channel("rt_min") == "rt"
    assert header_to_channel("C10H17Cl5") == "C10Cl5"
    assert header_to_channel("m/z 279") == "C10Cl5"
    assert header_to_channel("ISTD") == "istd"
    assert set(CONGENER_KEYS) >= {"C10Cl5", "C13Cl10"}


def test_vendor_formats_are_todo(tmp_path: Path) -> None:
    raw = tmp_path / "injection.mzML"
    raw.write_bytes(b"")
    with pytest.raises(VendorFormatNotImplemented, match="TODO"):
        integrate_file(raw)
    with pytest.raises(NotImplementedError):
        integrate_file(tmp_path / "tune.wiff")

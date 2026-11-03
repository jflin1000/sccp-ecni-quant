"""End-to-end integration, NNLS, and Reth on the synthetic batch."""

from pathlib import Path

import numpy as np
import pytest

from sccp_ecni.integrate import integrate_csv
from sccp_ecni.pipeline import run_folder
from sccp_ecni.synthetic import (
    RETH_INTERCEPT,
    RETH_SLOPE,
    STANDARD_SPECS,
    example_files,
    expected_parts,
    standard_area_library,
    write_chromatogram_csv,
    write_synthetic_batch,
)


def _rows(result):
    return {(row.role, row.name): row for row in result.quantified}


def test_written_batch_integrates_back_to_the_target_areas(tmp_path: Path) -> None:
    files = write_synthetic_batch(tmp_path)
    for item in files:
        integrated = integrate_csv(tmp_path / item.filename)
        assert integrated.areas == pytest.approx(item.areas, rel=1e-6, abs=1e-6)
        assert integrated.istd_area is None


def test_synthetic_batch_recovers_nnls_and_reth(tmp_path: Path) -> None:
    write_synthetic_batch(tmp_path / "batch")
    result = run_folder(tmp_path / "batch", tmp_path / "out")
    assert result.fit.slope == pytest.approx(RETH_SLOPE, rel=1e-6)
    assert result.fit.intercept == pytest.approx(RETH_INTERCEPT, rel=1e-6, abs=1e-6)
    assert result.fit.r_squared == pytest.approx(1.0, abs=1e-8)
    assert result.warnings == []
    rows = _rows(result)
    parts = expected_parts()

    for spec in STANDARD_SPECS:
        row = rows[("std", spec.name)]
        assert row.d_prime == pytest.approx(spec.target_d_prime, abs=1e-4)
        assert row.declared_chlorine == spec.declared_chlorine
        assert row.concentration_nnls == pytest.approx(spec.concentration, rel=1e-4)
        assert row.concentration_reth == pytest.approx(spec.concentration, rel=1e-4)
        assert row.recovery_nnls == pytest.approx(100.0, abs=0.05)
        _assert_parts(row, parts[spec.name])

    pure = rows[("sample", "pure55")]
    assert pure.concentration_nnls == pytest.approx(4.0, rel=1e-4)
    assert pure.concentration_reth == pytest.approx(4.0, rel=1e-4)
    assert pure.result_nnls == pytest.approx(4.0, rel=1e-4)
    _assert_parts(pure, parts["pure55"])

    mixed = rows[("sample", "mix")]
    assert mixed.concentration_nnls == pytest.approx(8.0, rel=1e-4)
    assert mixed.r_squared == pytest.approx(1.0, abs=1e-6)
    assert 58.0 < mixed.d_prime < 68.0
    _assert_parts(mixed, {"sccp51": 3.0, "sccp55": 0.0, "sccp63": 5.0})

    sediment = rows[("sample", "sediment")]
    assert sediment.concentration_nnls == pytest.approx(4.0, rel=1e-4)
    assert sediment.concentration_reth == pytest.approx(4.0, rel=1e-4)
    assert sediment.factor == 2.0
    assert sediment.result_nnls == pytest.approx(8.0, rel=1e-4)
    assert sediment.result_reth == pytest.approx(8.0, rel=1e-4)

    qc = rows[("qc", "check63")]
    assert qc.concentration_nnls == pytest.approx(6.0, rel=1e-4)
    assert qc.concentration_reth == pytest.approx(6.0, rel=1e-4)
    assert qc.recovery_reth == pytest.approx(100.0, abs=0.05)
    assert qc.recovery_nnls == pytest.approx(100.0, abs=0.05)

    # The two sccp55 replicates are scattered by ±2% and then averaged.
    injections = [item for item in result.injections if item.meta.name == "sccp55"]
    totals = sorted(float(item.raw_areas.sum()) for item in injections)
    assert totals[1] / totals[0] == pytest.approx(1.02 / 0.98, rel=1e-4)
    assert (tmp_path / "out" / "quantification.csv").is_file()
    assert (tmp_path / "out" / "calibration_fit.json").is_file()
    assert (tmp_path / "out" / "injections.csv").is_file()
    assert not (tmp_path / "out" / "warnings.txt").exists()


def test_blank_subtraction_and_internal_standard(tmp_path: Path) -> None:
    library = standard_area_library()
    blank = 0.04 * library["sccp51"]
    istd = 8.0
    specs = {spec.name: spec for spec in STANDARD_SPECS}

    def write(name: str, areas: np.ndarray) -> None:
        write_chromatogram_csv(tmp_path / name, areas, istd_area=istd)

    write(
        "std__sccp51__cl-51.5__conc-10__rep-1.csv",
        library["sccp51"] + blank,
    )
    write(
        "std__sccp63__cl-63.0__conc-10__rep-1.csv",
        library["sccp63"] + blank,
    )
    # A second level of the same mixture must collapse to the same unit response.
    write(
        "std__sccp51__cl-51.5__conc-5__rep-1.csv",
        library["sccp51"] * 0.5 + blank,
    )
    sample = library["sccp51"] * 0.3 + library["sccp63"] * 0.5 + blank
    write("sample__blend__rep-1.csv", sample)
    write("blank__lab__rep-1.csv", blank)

    result = run_folder(tmp_path, tmp_path / "out")
    rows = _rows(result)
    low = rows[("std", "sccp51")]
    # Both levels are quantified; the conc-5 row is a separate averaged injection.
    # Grouping is by concentration, so two std rows share the name. _rows keeps one.
    # Look the conc-5 row up directly.
    levels = [row for row in result.quantified if row.role == "std" and row.name == "sccp51"]
    by_conc = {row.expected_concentration: row for row in levels}
    assert by_conc[10.0].concentration_nnls == pytest.approx(10.0, rel=2e-3)
    assert by_conc[5.0].concentration_nnls == pytest.approx(5.0, rel=2e-3)
    assert by_conc[10.0].concentration_reth == pytest.approx(10.0, rel=2e-3)
    blend = rows[("sample", "blend")]
    assert blend.concentration_nnls == pytest.approx(8.0, rel=2e-3)
    names = list(blend.standard_names)
    coeffs = dict(zip(names, blend.coefficients, strict=True))
    assert coeffs["sccp51"] == pytest.approx(3.0, rel=2e-3)
    assert coeffs["sccp63"] == pytest.approx(5.0, rel=2e-3)
    assert low.d_prime == pytest.approx(specs["sccp51"].target_d_prime, abs=0.05)
    assert result.warnings == []


def test_committed_examples_match_the_generator() -> None:
    folder = Path(__file__).resolve().parents[1] / "examples" / "synthetic_batch"
    if not folder.is_dir():
        pytest.skip("examples have not been generated")
    expected = {item.filename: item.areas for item in example_files()}
    found = sorted(path.name for path in folder.glob("*.csv"))
    assert found == sorted(expected)
    for name, areas in expected.items():
        integrated = integrate_csv(folder / name)
        assert integrated.areas == pytest.approx(areas, rel=1e-6, abs=1e-6)


def _assert_parts(row, parts: dict[str, float]) -> None:
    found = dict(zip(row.standard_names, row.coefficients, strict=True))
    for name, value in parts.items():
        assert found[name] == pytest.approx(value, abs=1e-3)

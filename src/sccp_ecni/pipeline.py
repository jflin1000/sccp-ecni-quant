"""Batch path: named folder, CSV integration, NNLS deconvolution, Reth Fa–D'.

Vendor files in the folder are rejected with a TODO error. CSV files are
integrated, blank-subtracted, averaged across replicates, then quantified two
ways:

* NNLS against the standard response per concentration unit. The coefficients
  are the concentration of each calibration mixture, and their sum is the
  deconvolved concentration.
* Reth Fa–D'. D' is the area-weighted chlorine mass percent. The total response
  factor of each standard is fitted as ``RF = a * D' + b``, and the sample
  concentration is ``Fa / RF(D')``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sccp_ecni.deconv import nnls_deconv
from sccp_ecni.integrate import VENDOR_SUFFIXES, IntegratedChromatogram, integrate_file
from sccp_ecni.ions import CONGENER_KEYS, N_IONS
from sccp_ecni.naming import InjectionName, parse_injection_name
from sccp_ecni.reth import (
    CalibrationFit,
    d_prime,
    fit_reth,
    fractional_areas,
    quantify_reth,
)

_ROLE_ORDER = {"std": 0, "sample": 1, "qc": 2}


@dataclass
class LoadedInjection:
    """One file after integration and conversion to Fa."""

    meta: InjectionName
    path: Path
    raw_areas: np.ndarray
    istd_area: float | None
    fa: np.ndarray


@dataclass
class AveragedInjection:
    """Replicates of one solution, after blank subtraction."""

    role: str
    name: str
    replicates: tuple[int, ...]
    declared_chlorine: float | None
    concentration: float | None
    factor: float
    fa: np.ndarray
    paths: tuple[str, ...]


@dataclass
class StandardModel:
    """One calibration mixture, possibly averaged over several levels."""

    name: str
    declared_chlorine: float | None
    d_prime: float
    response_factor: float
    unit_response: np.ndarray
    concentrations: tuple[float, ...]
    n_replicates: int


@dataclass
class QuantifiedRow:
    """NNLS and Reth results for one averaged injection."""

    role: str
    name: str
    n_replicates: int
    declared_chlorine: float | None
    expected_concentration: float | None
    factor: float
    fa: np.ndarray
    d_prime: float
    response_factor: float
    concentration_reth: float
    concentration_nnls: float
    coefficients: np.ndarray
    reconstructed: np.ndarray
    standard_names: tuple[str, ...]
    r_squared: float
    rmse: float
    extrapolated: bool
    result_reth: float
    result_nnls: float
    recovery_reth: float | None
    recovery_nnls: float | None


@dataclass
class BatchResult:
    """Everything written for one folder."""

    folder: Path
    out_dir: Path
    injections: list[LoadedInjection]
    averaged: list[AveragedInjection]
    standards: list[StandardModel]
    fit: CalibrationFit
    quantified: list[QuantifiedRow]
    blank_fa: np.ndarray
    warnings: list[str] = field(default_factory=list)


def discover_chromatograms(folder: Path) -> list[Path]:
    """List CSV chromatograms in ``folder`` and reject known vendor files."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"batch folder does not exist: {folder}")
    csv_files: list[Path] = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            csv_files.append(path)
        elif suffix in VENDOR_SUFFIXES:
            # integrate_file raises the TODO error with the format name.
            integrate_file(path)
    if not csv_files:
        raise FileNotFoundError(f"no CSV chromatograms in {folder}")
    return csv_files


def load_injections(folder: Path) -> list[LoadedInjection]:
    """Parse names and integrate every CSV in the folder."""
    loaded: list[LoadedInjection] = []
    for path in discover_chromatograms(folder):
        meta = parse_injection_name(path.name)
        integrated: IntegratedChromatogram = integrate_file(path)
        fa = fractional_areas(integrated.areas, integrated.istd_area)
        loaded.append(
            LoadedInjection(
                meta=meta,
                path=path,
                raw_areas=integrated.areas.copy(),
                istd_area=integrated.istd_area,
                fa=fa,
            )
        )
    _require_consistent_istd(loaded)
    return loaded


def _require_consistent_istd(injections: list[LoadedInjection]) -> None:
    present = [item.istd_area is not None for item in injections]
    if any(present) and not all(present):
        missing = [item.meta.filename for item, flag in zip(injections, present, strict=True) if not flag]
        raise ValueError(
            "either every chromatogram has an ISTD column or none do. Missing ISTD: "
            + ", ".join(missing)
        )


def _blank_vector(injections: list[LoadedInjection], subtract_blank: bool) -> np.ndarray:
    blanks = [item for item in injections if item.meta.role == "blank"]
    if not subtract_blank or not blanks:
        return np.zeros(N_IONS, dtype=float)
    return np.mean([item.fa for item in blanks], axis=0)


def _average_replicates(
    injections: list[LoadedInjection],
    blank_fa: np.ndarray,
) -> list[AveragedInjection]:
    grouped: dict[tuple, list[LoadedInjection]] = {}
    for item in injections:
        if item.meta.role == "blank":
            continue
        if item.meta.role == "std":
            key = ("std", item.meta.name, item.meta.concentration)
        else:
            key = (item.meta.role, item.meta.name)
        grouped.setdefault(key, []).append(item)

    averaged: list[AveragedInjection] = []
    for items in grouped.values():
        first = items[0].meta
        replicates = [item.meta.replicate for item in items]
        if len(replicates) != len(set(replicates)):
            raise ValueError(f"duplicate replicate numbers for {first.role} {first.name}")
        if first.role != "std":
            factors = {item.meta.factor for item in items}
            concentrations = {item.meta.concentration for item in items}
            chlorines = {item.meta.chlorine for item in items}
            if len(factors) > 1 or len(concentrations) > 1 or len(chlorines) > 1:
                raise ValueError(
                    f"{first.role} {first.name}: replicates disagree on factor, conc, or cl"
                )
        corrected = [np.clip(item.fa - blank_fa, 0.0, None) for item in items]
        fa = np.mean(corrected, axis=0)
        if float(fa.sum()) <= 0:
            raise ValueError(
                f"{first.role} {first.name} has no SCCP signal after blank subtraction"
            )
        averaged.append(
            AveragedInjection(
                role=first.role,
                name=first.name,
                replicates=tuple(sorted(replicates)),
                declared_chlorine=first.chlorine,
                concentration=first.concentration,
                factor=first.factor,
                fa=fa,
                paths=tuple(item.meta.filename for item in items),
            )
        )
    averaged.sort(key=lambda item: (_ROLE_ORDER.get(item.role, 9), item.name, item.concentration or 0))
    return averaged


def _standard_models(averaged: list[AveragedInjection]) -> list[StandardModel]:
    by_name: dict[str, list[AveragedInjection]] = {}
    for item in averaged:
        if item.role == "std":
            by_name.setdefault(item.name, []).append(item)
    if len(by_name) < 2:
        raise ValueError("Reth calibration needs at least two standards with different D' values")
    models: list[StandardModel] = []
    for name in sorted(by_name):
        rows = by_name[name]
        declared = {row.declared_chlorine for row in rows}
        if len(declared) > 1:
            raise ValueError(f"standard {name} has conflicting cl- values")
        units = []
        for row in rows:
            if row.concentration is None or row.concentration <= 0:
                raise ValueError(f"standard {name} is missing conc-")
            units.append(row.fa / row.concentration)
        unit = np.mean(units, axis=0)
        if float(unit.sum()) <= 0:
            raise ValueError(f"standard {name} has a zero response factor")
        models.append(
            StandardModel(
                name=name,
                declared_chlorine=rows[0].declared_chlorine,
                d_prime=d_prime(unit),
                response_factor=float(unit.sum()),
                unit_response=unit,
                concentrations=tuple(float(row.concentration) for row in rows if row.concentration is not None),
                n_replicates=sum(len(row.replicates) for row in rows),
            )
        )
    return models


def _recovery(found: float, expected: float | None) -> float | None:
    if expected is None:
        return None
    return 100.0 * found / expected


def quantify_loaded(
    injections: list[LoadedInjection],
    *,
    subtract_blank: bool = True,
) -> tuple[list[AveragedInjection], list[StandardModel], CalibrationFit, list[QuantifiedRow], np.ndarray, list[str]]:
    """Run blank subtraction, NNLS, and Reth on injections that are already integrated."""
    blank_fa = _blank_vector(injections, subtract_blank)
    averaged = _average_replicates(injections, blank_fa)
    standards = _standard_models(averaged)
    fit = fit_reth(
        np.array([item.d_prime for item in standards]),
        np.array([item.response_factor for item in standards]),
    )
    matrix = np.column_stack([item.unit_response for item in standards])
    names = tuple(item.name for item in standards)
    d_values = np.array([item.d_prime for item in standards])
    d_min = float(d_values.min())
    d_max = float(d_values.max())
    quantified: list[QuantifiedRow] = []
    for item in averaged:
        solution = nnls_deconv(item.fa, matrix)
        chlorine = d_prime(item.fa)
        conc_reth = quantify_reth(item.fa, chlorine, fit)
        conc_nnls = float(solution.coefficients.sum())
        quantified.append(
            QuantifiedRow(
                role=item.role,
                name=item.name,
                n_replicates=len(item.replicates),
                declared_chlorine=item.declared_chlorine,
                expected_concentration=item.concentration,
                factor=item.factor,
                fa=item.fa,
                d_prime=chlorine,
                response_factor=fit.predict(chlorine),
                concentration_reth=conc_reth,
                concentration_nnls=conc_nnls,
                coefficients=solution.coefficients,
                reconstructed=solution.reconstructed,
                standard_names=names,
                r_squared=solution.r_squared,
                rmse=solution.rmse,
                extrapolated=chlorine < d_min - 0.05 or chlorine > d_max + 0.05,
                result_reth=conc_reth * item.factor,
                result_nnls=conc_nnls * item.factor,
                recovery_reth=_recovery(conc_reth, item.concentration),
                recovery_nnls=_recovery(conc_nnls, item.concentration),
            )
        )
    warnings = _warnings(fit, quantified)
    return averaged, standards, fit, quantified, blank_fa, warnings


def _warnings(fit: CalibrationFit, rows: list[QuantifiedRow]) -> list[str]:
    notes: list[str] = []
    if fit.r_squared < 0.99:
        notes.append(f"Reth calibration R² is {fit.r_squared:.4f}, below 0.99")
    for row in rows:
        label = f"{row.role} {row.name}"
        if row.r_squared < 0.8:
            notes.append(f"{label}: pattern fit R² = {row.r_squared:.3f}")
        if row.extrapolated:
            notes.append(
                f"{label}: D' = {row.d_prime:.2f} is outside the standard chlorine window"
            )
        for kind, recovery in (("NNLS", row.recovery_nnls), ("Reth", row.recovery_reth)):
            if recovery is not None and not 80.0 <= recovery <= 120.0:
                notes.append(f"{label}: {kind} recovery is {recovery:.1f}%")
    return notes


def run_folder(folder: Path, out_dir: Path, *, subtract_blank: bool = True) -> BatchResult:
    """Integrate ``folder`` and write quantification tables to ``out_dir``."""
    folder = Path(folder)
    out_dir = Path(out_dir)
    injections = load_injections(folder)
    averaged, standards, fit, quantified, blank_fa, warnings = quantify_loaded(
        injections, subtract_blank=subtract_blank
    )
    result = BatchResult(
        folder=folder,
        out_dir=out_dir,
        injections=injections,
        averaged=averaged,
        standards=standards,
        fit=fit,
        quantified=quantified,
        blank_fa=blank_fa,
        warnings=warnings,
    )
    write_result(result)
    return result


def write_result(result: BatchResult) -> None:
    """Write the CSV and JSON tables for a finished batch."""
    out_dir = result.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_injections(out_dir / "injections.csv", result.injections)
    _write_responses(out_dir / "responses.csv", result.averaged)
    _write_calibration(out_dir / "calibration.csv", result.standards)
    _write_fit(out_dir / "calibration_fit.json", result.fit, result.standards)
    _write_quantification(out_dir / "quantification.csv", result.quantified)
    _write_profiles(out_dir / "profiles.csv", result.quantified)
    warning_path = out_dir / "warnings.txt"
    if result.warnings:
        warning_path.write_text("\n".join(result.warnings) + "\n", encoding="utf-8")
    elif warning_path.exists():
        warning_path.unlink()


def format_report(result: BatchResult) -> str:
    """Text table of solution concentrations and factored results."""
    sign = "+" if result.fit.intercept >= 0 else "-"
    lines = [
        f"Reth Fa–D': RF = {result.fit.slope:.6g} × D' {sign} {abs(result.fit.intercept):.6g}"
        f"    R² = {result.fit.r_squared:.6f}    standards = {len(result.standards)}",
        "",
        f"{'role':<8} {'name':<12} {'D%':>8} {'NNLS':>10} {'Reth':>10} "
        f"{'result_NNLS':>12} {'result_Reth':>12} {'R²':>8}",
    ]
    for row in result.quantified:
        lines.append(
            f"{row.role:<8} {row.name:<12} {row.d_prime:8.2f} {row.concentration_nnls:10.4f} "
            f"{row.concentration_reth:10.4f} {row.result_nnls:12.4f} {row.result_reth:12.4f} "
            f"{row.r_squared:8.4f}"
        )
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {note}" for note in result.warnings)
    lines.append("")
    lines.append(f"Results: {result.out_dir}")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10g}"
    return str(value)


def _write_rows(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_fmt(value) for value in row])


def _write_injections(path: Path, injections: list[LoadedInjection]) -> None:
    header = [
        "file",
        "role",
        "name",
        "replicate",
        "declared_cl",
        "concentration",
        "factor",
        "istd_area",
        "total_area",
        *CONGENER_KEYS,
    ]
    rows = []
    for item in injections:
        rows.append(
            [
                item.meta.filename,
                item.meta.role,
                item.meta.name,
                item.meta.replicate,
                item.meta.chlorine,
                item.meta.concentration,
                item.meta.factor,
                item.istd_area,
                float(item.raw_areas.sum()),
                *item.raw_areas.tolist(),
            ]
        )
    _write_rows(path, header, rows)


def _write_responses(path: Path, averaged: list[AveragedInjection]) -> None:
    header = [
        "role",
        "name",
        "replicates",
        "declared_cl",
        "concentration",
        "factor",
        "fa_total",
        "d_prime",
        *CONGENER_KEYS,
    ]
    rows = []
    for item in averaged:
        rows.append(
            [
                item.role,
                item.name,
                " ".join(str(number) for number in item.replicates),
                item.declared_chlorine,
                item.concentration,
                item.factor,
                float(item.fa.sum()),
                d_prime(item.fa),
                *item.fa.tolist(),
            ]
        )
    _write_rows(path, header, rows)


def _write_calibration(path: Path, standards: list[StandardModel]) -> None:
    header = [
        "name",
        "declared_cl",
        "d_prime",
        "response_factor",
        "n_replicates",
        "concentrations",
    ]
    rows = [
        [
            item.name,
            item.declared_chlorine,
            item.d_prime,
            item.response_factor,
            item.n_replicates,
            " ".join(_fmt(value) for value in item.concentrations),
        ]
        for item in standards
    ]
    _write_rows(path, header, rows)


def _write_fit(path: Path, fit: CalibrationFit, standards: list[StandardModel]) -> None:
    payload = {
        "method": "Reth 2005 Fa-D'",
        "equation": "RF = a * D' + b",
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r_squared": fit.r_squared,
        "n_standards": len(standards),
        "d_prime_min": float(np.min(fit.d_prime)),
        "d_prime_max": float(np.max(fit.d_prime)),
        "standards": [
            {
                "name": item.name,
                "declared_chlorine_percent": item.declared_chlorine,
                "d_prime": item.d_prime,
                "response_factor": item.response_factor,
                "concentrations": list(item.concentrations),
            }
            for item in standards
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_quantification(path: Path, quantified: list[QuantifiedRow]) -> None:
    names = quantified[0].standard_names if quantified else ()
    header = [
        "role",
        "name",
        "n_replicates",
        "declared_cl",
        "expected_concentration",
        "factor",
        "fa_total",
        "d_prime",
        "rf_reth",
        "concentration_reth",
        "concentration_nnls",
        "result_reth",
        "result_nnls",
        "recovery_reth_percent",
        "recovery_nnls_percent",
        "r_squared",
        "rmse",
        "extrapolated",
        *[f"coef__{name}" for name in names],
        *[f"share__{name}" for name in names],
    ]
    rows = []
    for item in quantified:
        total = item.concentration_nnls
        shares = item.coefficients / total if total > 0 else np.zeros_like(item.coefficients)
        rows.append(
            [
                item.role,
                item.name,
                item.n_replicates,
                item.declared_chlorine,
                item.expected_concentration,
                item.factor,
                float(item.fa.sum()),
                item.d_prime,
                item.response_factor,
                item.concentration_reth,
                item.concentration_nnls,
                item.result_reth,
                item.result_nnls,
                item.recovery_reth,
                item.recovery_nnls,
                item.r_squared,
                item.rmse,
                item.extrapolated,
                *item.coefficients.tolist(),
                *shares.tolist(),
            ]
        )
    _write_rows(path, header, rows)


def _write_profiles(path: Path, quantified: list[QuantifiedRow]) -> None:
    header = ["role", "name", "kind", *CONGENER_KEYS]
    rows = []
    for item in quantified:
        rows.append([item.role, item.name, "measured", *item.fa.tolist()])
        rows.append([item.role, item.name, "reconstructed", *item.reconstructed.tolist()])
    _write_rows(path, header, rows)

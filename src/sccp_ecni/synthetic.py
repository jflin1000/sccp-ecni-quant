"""Synthetic SCCP chromatograms with a known NNLS mixture and a linear Reth line.

Three technical mixtures are given distinct C10–C13 patterns. Their observed
areas are scaled so the area-weighted chlorine content D' and the total
response factor sit on

    RF = 20 * D' - 1020

A pure sample therefore has an exact Reth concentration, and a mixture of the
unit-response vectors has an exact NNLS concentration. Replicate files of the
55.5% Cl standard are scaled by 0.98 and 1.02; their average is the ideal
pattern. The blank is baseline only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import exp
from pathlib import Path

import numpy as np

from sccp_ecni.integrate import integrate_trace
from sccp_ecni.ions import N_IONS, SCCP_GROUPS, chlorine_percents

RETH_SLOPE = 20.0
RETH_INTERCEPT = -1020.0
_RT_START = 4.0
_RT_STOP = 28.0
_N_POINTS = 241
_BASELINE = 5.0
_PEAK_SIGMA = 0.50


@dataclass(frozen=True)
class StandardSpec:
    """One synthetic technical mixture at a single calibration level."""

    name: str
    declared_chlorine: float
    concentration: float
    target_d_prime: float
    carbon_center: float
    carbon_sigma: float
    chlorine_sigma: float


@dataclass(frozen=True)
class ExampleFile:
    """One CSV in the synthetic batch and the areas it is built to produce."""

    filename: str
    role: str
    name: str
    declared_chlorine: float | None
    concentration: float | None
    replicate: int
    factor: float
    area_scale: float
    parts: dict[str, float]
    areas: np.ndarray


STANDARD_SPECS: tuple[StandardSpec, ...] = (
    StandardSpec("sccp51", 51.5, 10.0, 58.0, carbon_center=12.7, carbon_sigma=0.85, chlorine_sigma=0.75),
    StandardSpec("sccp55", 55.5, 10.0, 62.0, carbon_center=11.5, carbon_sigma=1.35, chlorine_sigma=1.15),
    StandardSpec("sccp63", 63.0, 10.0, 68.0, carbon_center=10.5, carbon_sigma=0.80, chlorine_sigma=0.70),
)


def rf_on_line(d_prime_value: float) -> float:
    """Response factor on the synthetic Reth line."""
    return RETH_SLOPE * d_prime_value + RETH_INTERCEPT


def _composition(spec: StandardSpec, chlorine_center: float) -> np.ndarray:
    weights = np.empty(N_IONS, dtype=float)
    for index, group in enumerate(SCCP_GROUPS):
        carbon_term = (group.carbon - spec.carbon_center) / spec.carbon_sigma
        chlorine_term = (group.chlorine - chlorine_center) / spec.chlorine_sigma
        weights[index] = exp(-0.5 * (carbon_term**2 + chlorine_term**2))
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError(f"empty composition for {spec.name}")
    return weights / total


def composition_for_d_prime(spec: StandardSpec) -> np.ndarray:
    """Area fractions whose D' equals the spec, found by shifting the chlorine center."""
    percent = np.asarray(chlorine_percents(), dtype=float)

    def achieved(center: float) -> float:
        return float(_composition(spec, center) @ percent)

    low = achieved(5.0)
    high = achieved(10.0)
    if not low - 1e-6 <= spec.target_d_prime <= high + 1e-6:
        raise RuntimeError(
            f"{spec.name}: target D' {spec.target_d_prime:.2f} is outside the "
            f"achievable window {low:.2f}–{high:.2f}"
        )
    lo, hi = 5.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if achieved(mid) < spec.target_d_prime:
            lo = mid
        else:
            hi = mid
    return _composition(spec, 0.5 * (lo + hi))


def standard_area_library() -> dict[str, np.ndarray]:
    """Ideal integrated areas of each standard at its labelled concentration."""
    percent = np.asarray(chlorine_percents(), dtype=float)
    library: dict[str, np.ndarray] = {}
    for spec in STANDARD_SPECS:
        fractions = composition_for_d_prime(spec)
        total_area = rf_on_line(spec.target_d_prime) * spec.concentration
        areas = fractions * total_area
        calculated = float(areas @ percent / areas.sum())
        if abs(calculated - spec.target_d_prime) > 1e-6:
            raise RuntimeError(
                f"{spec.name}: D' landed on {calculated:.6f}, not {spec.target_d_prime:.6f}"
            )
        response = float(areas.sum() / spec.concentration)
        if abs(response - rf_on_line(spec.target_d_prime)) > 1e-6:
            raise RuntimeError(f"{spec.name}: RF {response} left the synthetic line")
        library[spec.name] = areas
    matrix = np.column_stack([library[spec.name] for spec in STANDARD_SPECS])
    if np.linalg.matrix_rank(matrix, tol=1e-6) < len(STANDARD_SPECS):
        raise RuntimeError("synthetic standards are linearly dependent")
    return library


def _ideal_areas(parts: dict[str, float], library: dict[str, np.ndarray]) -> np.ndarray:
    areas = np.zeros(N_IONS, dtype=float)
    concentrations = {spec.name: spec.concentration for spec in STANDARD_SPECS}
    for name, contribution in parts.items():
        areas += library[name] * (contribution / concentrations[name])
    return areas


def example_files() -> list[ExampleFile]:
    """Describe every file in ``examples/synthetic_batch``."""
    library = standard_area_library()
    files: list[ExampleFile] = []

    def add(
        filename: str,
        role: str,
        name: str,
        *,
        declared_chlorine: float | None,
        concentration: float | None,
        replicate: int,
        factor: float,
        area_scale: float,
        parts: dict[str, float],
    ) -> None:
        areas = _ideal_areas(parts, library) * area_scale
        files.append(
            ExampleFile(
                filename=filename,
                role=role,
                name=name,
                declared_chlorine=declared_chlorine,
                concentration=concentration,
                replicate=replicate,
                factor=factor,
                area_scale=area_scale,
                parts=dict(parts),
                areas=areas,
            )
        )

    for spec in STANDARD_SPECS:
        replicates = (1, 2) if spec.name == "sccp55" else (1,)
        for replicate in replicates:
            scale = {1: 0.98, 2: 1.02}.get(replicate, 1.0) if len(replicates) > 1 else 1.0
            filename = (
                f"std__{spec.name}__cl-{spec.declared_chlorine:.1f}"
                f"__conc-{spec.concentration:.10g}__rep-{replicate}.csv"
            )
            add(
                filename,
                "std",
                spec.name,
                declared_chlorine=spec.declared_chlorine,
                concentration=spec.concentration,
                replicate=replicate,
                factor=1.0,
                area_scale=scale,
                parts={spec.name: spec.concentration},
            )
    add(
        "sample__pure55__rep-1.csv",
        "sample",
        "pure55",
        declared_chlorine=None,
        concentration=None,
        replicate=1,
        factor=1.0,
        area_scale=1.0,
        parts={"sccp55": 4.0},
    )
    add(
        "sample__mix__rep-1.csv",
        "sample",
        "mix",
        declared_chlorine=None,
        concentration=None,
        replicate=1,
        factor=1.0,
        area_scale=1.0,
        parts={"sccp51": 3.0, "sccp63": 5.0},
    )
    add(
        "sample__sediment__factor-2__rep-1.csv",
        "sample",
        "sediment",
        declared_chlorine=None,
        concentration=None,
        replicate=1,
        factor=2.0,
        area_scale=1.0,
        parts={"sccp55": 4.0},
    )
    add(
        "qc__check63__cl-63.0__conc-6__rep-1.csv",
        "qc",
        "check63",
        declared_chlorine=63.0,
        concentration=6.0,
        replicate=1,
        factor=1.0,
        area_scale=1.0,
        parts={"sccp63": 6.0},
    )
    add(
        "blank__solvent__rep-1.csv",
        "blank",
        "solvent",
        declared_chlorine=None,
        concentration=None,
        replicate=1,
        factor=1.0,
        area_scale=1.0,
        parts={},
    )
    return files


def write_chromatogram_csv(
    path: Path,
    areas: np.ndarray,
    istd_area: float | None = None,
) -> None:
    """Write a wide CSV whose integrated SCCP areas equal ``areas``.

    Peak height is scaled with the same baseline integrator the batch uses, so
    a later call to :func:`sccp_ecni.integrate.integrate_csv` returns ``areas``
    up to CSV rounding.
    """
    vector = np.asarray(areas, dtype=float).reshape(-1)
    if vector.shape != (N_IONS,):
        raise ValueError(f"expected {N_IONS} areas, got {vector.size}")
    if np.any(vector < 0):
        raise ValueError("areas must be non-negative")
    rt = np.linspace(_RT_START, _RT_STOP, _N_POINTS)
    columns: list[np.ndarray] = []
    targets = list(vector) + ([] if istd_area is None else [float(istd_area)])
    centers = [11.0 + (group.carbon - 10) * 3.0 + (group.chlorine - 7.5) * 0.35 for group in SCCP_GROUPS]
    if istd_area is not None:
        centers.append(16.0)
    for target, center in zip(targets, centers, strict=True):
        shape = np.exp(-0.5 * ((rt - center) / _PEAK_SIGMA) ** 2)
        if target == 0:
            columns.append(np.full(rt.shape, _BASELINE))
            continue
        measured = integrate_trace(rt, shape + _BASELINE)
        if measured <= 0:
            raise RuntimeError(f"synthetic peak at {center} min produced no area")
        columns.append(shape * (target / measured) + _BASELINE)
    matrix = np.column_stack(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["rt_min", *[group.key for group in SCCP_GROUPS]]
    if istd_area is not None:
        header.append("ISTD")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_index in range(rt.size):
            row = [f"{rt[row_index]:.6f}"]
            row.extend(f"{matrix[row_index, column]:.8f}" for column in range(matrix.shape[1]))
            writer.writerow(row)


def write_synthetic_batch(folder: Path) -> list[ExampleFile]:
    """Write the synthetic batch and return the file manifest."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = example_files()
    for item in files:
        write_chromatogram_csv(folder / item.filename, item.areas)
    return files


def expected_parts() -> dict[str, dict[str, float]]:
    """Ideal NNLS concentration of each standard in each averaged injection.

    Replicate scatter is averaged away, so these are the concentrations the
    deconvolution should recover from the batch.
    """
    expected: dict[str, dict[str, float]] = {}
    for item in example_files():
        if item.role == "blank":
            continue
        bucket = expected.setdefault(item.name, {spec.name: 0.0 for spec in STANDARD_SPECS})
        for name, contribution in item.parts.items():
            bucket[name] = contribution
    return expected

"""Integrate unit-resolution SCCP extracted-ion chromatograms.

CSV is fully supported. Each file is one injection: a time column plus one
intensity column for every quantifier ion (formula key or nominal m/z). An
optional internal-standard column is integrated the same way.

Vendor acquisitions (Thermo/Waters RAW, Agilent ``.d``, SCIEX WIFF, mzML,
mzXML, ANDI/netCDF, Shimadzu LCD) are recognised so a mixed folder fails
loudly, but readers for those formats are not implemented yet.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sccp_ecni.ions import CONGENER_KEYS, KEY_INDEX, MZ_TO_KEY, N_IONS, SCCP_GROUPS

VENDOR_SUFFIXES: dict[str, str] = {
    ".raw": "Thermo RAW or Waters RAW",
    ".d": "Agilent MassHunter .d",
    ".wiff": "SCIEX WIFF",
    ".wiff2": "SCIEX WIFF2",
    ".mzml": "mzML",
    ".mzxml": "mzXML",
    ".cdf": "ANDI/netCDF",
    ".lcd": "Shimadzu LCD",
}

_RT_HEADERS = {
    "rt",
    "rtmin",
    "retentiontime",
    "time",
    "timemin",
    "minutes",
    "min",
}
_ISTD_HEADERS = {"istd", "is", "internalstandard", "internalstd"}


class VendorFormatNotImplemented(NotImplementedError):
    """Raised for instrument files that must be exported to CSV first."""


@dataclass(frozen=True)
class IntegratedChromatogram:
    """Peak areas of one injection, aligned to :data:`sccp_ecni.ions.SCCP_GROUPS`."""

    areas: np.ndarray
    istd_area: float | None

    def __post_init__(self) -> None:
        areas = np.asarray(self.areas, dtype=float).reshape(-1)
        if areas.shape != (N_IONS,):
            raise ValueError(f"expected {N_IONS} areas, got {areas.size}")
        object.__setattr__(self, "areas", areas)


def _normalize_header(name: str) -> str:
    return re.sub(r"[\s_]", "", name.strip()).lower()


def header_to_channel(name: str) -> str:
    """Map a CSV header to ``rt``, ``istd``, or a congener key such as ``C10Cl5``."""
    norm = _normalize_header(name)
    if norm in _RT_HEADERS:
        return "rt"
    if norm in _ISTD_HEADERS:
        return "istd"
    formula = re.fullmatch(r"c(\d{1,2})h(\d{1,2})cl(\d{1,2})", norm)
    if formula:
        carbon, hydrogen, chlorine = (int(part) for part in formula.groups())
        if hydrogen != 2 * carbon + 2 - chlorine:
            raise ValueError(f"column {name!r} is not a saturated chlorinated alkane")
        key = f"C{carbon}Cl{chlorine}"
        if key not in KEY_INDEX:
            raise ValueError(f"column {name!r} is outside the fixed SCCP set C10–C13, Cl5–Cl10")
        return key
    short = re.fullmatch(r"c(\d{1,2})cl(\d{1,2})", norm)
    if short:
        key = f"C{int(short.group(1))}Cl{int(short.group(2))}"
        if key not in KEY_INDEX:
            raise ValueError(f"column {name!r} is outside the fixed SCCP set C10–C13, Cl5–Cl10")
        return key
    mz_header = re.fullmatch(r"(?:m/z|mz)?(\d{2,4})", norm)
    if mz_header and (norm[0].isdigit() or norm.startswith("mz") or norm.startswith("m/z")):
        mz = int(mz_header.group(1))
        key = MZ_TO_KEY.get(mz)
        if key is None:
            raise ValueError(f"m/z {mz} from column {name!r} is not one of the 24 SCCP quantifier ions")
        return key
    raise ValueError(
        f"unrecognized column {name!r}. Use rt_min plus C10Cl5…C13Cl10, the matching m/z, "
        "and an optional ISTD column."
    )


def integrate_trace(rt: np.ndarray, intensity: np.ndarray, edge_fraction: float = 0.05) -> float:
    """Trapezoidal area above a flat baseline.

    The baseline is the median intensity of the first and last
    ``edge_fraction`` of the points. That matches a full extracted-ion
    chromatogram whose peak sits inside a quiet start and end. Intensities
    below the baseline do not contribute.
    """
    if not 0 < edge_fraction < 0.5:
        raise ValueError("edge_fraction must be between 0 and 0.5")
    time = np.asarray(rt, dtype=float).reshape(-1)
    signal = np.asarray(intensity, dtype=float).reshape(-1)
    if time.shape != signal.shape:
        raise ValueError("rt and intensity must have the same length")
    if time.size < 2:
        raise ValueError("a chromatogram needs at least two points")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        raise ValueError("rt and intensity must be finite")
    if np.any(np.diff(time) <= 0):
        order = np.argsort(time, kind="mergesort")
        time = time[order]
        signal = signal[order]
        if np.any(np.diff(time) <= 0):
            raise ValueError("rt values must be strictly increasing")
    n_edge = max(1, int(round(time.size * edge_fraction)))
    if 2 * n_edge > time.size:
        n_edge = max(1, time.size // 5)
    edges = np.concatenate((signal[:n_edge], signal[-n_edge:]))
    baseline = float(np.median(edges))
    corrected = np.clip(signal - baseline, 0.0, None)
    trap = getattr(np, "trapezoid", None)
    if trap is None:
        trap = np.trapz
    return float(trap(corrected, time))


def read_chromatogram_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    """Read a wide chromatogram CSV.

    Returns retention time, a mapping of congener key to intensity, and the
    internal-standard trace when that column is present.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"{path.name}: empty chromatogram")
    first_line = text.splitlines()[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if not rows:
        raise ValueError(f"{path.name}: empty chromatogram")
    header = [cell.strip() for cell in rows[0]]
    if not any(header):
        raise ValueError(f"{path.name}: missing header")
    channels: dict[str, int] = {}
    for index, name in enumerate(header):
        if name == "":
            continue
        channel = header_to_channel(name)
        if channel in channels:
            raise ValueError(f"{path.name}: duplicate column for {channel}")
        channels[channel] = index
    if "rt" not in channels:
        raise ValueError(f"{path.name}: missing retention-time column (rt_min)")
    missing = [key for key in CONGENER_KEYS if key not in channels]
    if missing:
        preview = ", ".join(missing[:6])
        extra = "" if len(missing) <= 6 else f" (+{len(missing) - 6} more)"
        raise ValueError(f"{path.name}: missing ion columns: {preview}{extra}")

    rt_values: list[float] = []
    traces: dict[str, list[float]] = {key: [] for key in CONGENER_KEYS}
    istd_values: list[float] | None = [] if "istd" in channels else None
    for line_number, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) < len(header):
            raise ValueError(f"{path.name}: line {line_number} has fewer cells than the header")
        try:
            rt_values.append(float(row[channels["rt"]]))
            for key in CONGENER_KEYS:
                traces[key].append(float(row[channels[key]]))
            if istd_values is not None:
                istd_values.append(float(row[channels["istd"]]))
        except ValueError as exc:
            raise ValueError(f"{path.name}: line {line_number} is not numeric") from exc
    if len(rt_values) < 2:
        raise ValueError(f"{path.name}: need at least two data rows")
    rt = np.asarray(rt_values, dtype=float)
    ion_traces = {key: np.asarray(values, dtype=float) for key, values in traces.items()}
    istd = None if istd_values is None else np.asarray(istd_values, dtype=float)
    return rt, ion_traces, istd


def integrate_chromatogram(
    rt: np.ndarray,
    traces: dict[str, np.ndarray],
    istd: np.ndarray | None = None,
) -> IntegratedChromatogram:
    """Integrate every SCCP channel and the optional internal standard."""
    missing = [key for key in CONGENER_KEYS if key not in traces]
    if missing:
        raise ValueError(f"missing ion traces: {', '.join(missing)}")
    areas = np.array([integrate_trace(rt, traces[group.key]) for group in SCCP_GROUPS], dtype=float)
    istd_area = None if istd is None else integrate_trace(rt, istd)
    if istd_area is not None and istd_area <= 0:
        raise ValueError("internal-standard area must be positive")
    return IntegratedChromatogram(areas=areas, istd_area=istd_area)


def integrate_csv(path: Path) -> IntegratedChromatogram:
    """Integrate one wide CSV chromatogram."""
    rt, traces, istd = read_chromatogram_csv(path)
    return integrate_chromatogram(rt, traces, istd)


def integrate_file(path: Path) -> IntegratedChromatogram:
    """Integrate a CSV file, or reject a known vendor format that is still TODO."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return integrate_csv(path)
    if suffix in VENDOR_SUFFIXES:
        kind = VENDOR_SUFFIXES[suffix]
        raise VendorFormatNotImplemented(
            f"TODO: vendor format {suffix} ({kind}) is not implemented. "
            "Export the 24 SCCP [M−Cl]− extracted-ion chromatograms to CSV "
            f"(see docs/naming.md). File: {path.name}"
        )
    raise ValueError(f"unsupported file type {suffix!r}: {path.name}")

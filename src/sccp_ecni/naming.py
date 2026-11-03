"""File-name contract for a batch folder.

See ``docs/naming.md``. The role, sample id, and calibration fields are encoded
in the file name so a folder can be quantified without a separate sequence list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROLES = ("std", "sample", "blank", "qc")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*")
_FIELDS = ("cl", "conc", "rep", "factor")


@dataclass(frozen=True)
class InjectionName:
    """Parsed identity of one chromatogram file."""

    role: str
    name: str
    chlorine: float | None
    concentration: float | None
    replicate: int
    factor: float
    filename: str

    @property
    def is_quantitative(self) -> bool:
        return self.role in {"std", "sample", "qc"}


def parse_injection_name(filename: str) -> InjectionName:
    """Parse ``role__id__cl-55.5__conc-10__rep-1.csv``.

    Raises:
        ValueError: the name does not follow ``docs/naming.md``.
    """
    base = Path(filename).name
    if base.lower().endswith(".csv"):
        stem = base[:-4]
    else:
        raise ValueError(f"{base}: chromatogram files must be .csv (vendor formats are TODO)")
    parts = stem.split("__")
    if len(parts) < 2:
        raise ValueError(
            f"{base}: expected role__id__field-value.csv. See docs/naming.md."
        )
    role = parts[0].lower()
    sample_id = parts[1]
    if role not in ROLES:
        raise ValueError(f"{base}: role must be one of {', '.join(ROLES)}")
    if not _ID_RE.fullmatch(sample_id):
        raise ValueError(
            f"{base}: id {sample_id!r} must start with a letter or digit and "
            "contain only letters, digits, '.' and '-'"
        )
    fields: dict[str, str] = {}
    for token in parts[2:]:
        if "-" not in token:
            raise ValueError(f"{base}: field {token!r} must look like key-value")
        key, raw_value = token.split("-", 1)
        key = key.lower()
        if key not in _FIELDS:
            raise ValueError(f"{base}: unknown field {key!r}. Use cl, conc, rep, or factor.")
        if key in fields:
            raise ValueError(f"{base}: duplicate field {key}")
        if raw_value == "":
            raise ValueError(f"{base}: field {key} has no value")
        fields[key] = raw_value

    chlorine = _parse_float(base, "cl", fields.get("cl"))
    concentration = _parse_float(base, "conc", fields.get("conc"))
    factor = _parse_float(base, "factor", fields.get("factor"))
    if chlorine is not None and not 0 < chlorine < 100:
        raise ValueError(f"{base}: cl must be a chlorine mass percent between 0 and 100")
    if concentration is not None and concentration <= 0:
        raise ValueError(f"{base}: conc must be positive")
    if factor is None:
        factor = 1.0
    elif factor <= 0:
        raise ValueError(f"{base}: factor must be positive")
    if "rep" in fields:
        if not re.fullmatch(r"[1-9]\d*", fields["rep"]):
            raise ValueError(f"{base}: rep must be a positive integer")
        replicate = int(fields["rep"])
    else:
        replicate = 1
    if role == "std" and (chlorine is None or concentration is None):
        raise ValueError(f"{base}: standards require both cl-<percent> and conc-<value>")
    return InjectionName(
        role=role,
        name=sample_id,
        chlorine=chlorine,
        concentration=concentration,
        replicate=replicate,
        factor=factor,
        filename=base,
    )


def _parse_float(filename: str, field: str, raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{filename}: {field} value {raw!r} is not a number") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{filename}: {field} value {raw!r} is not finite")
    return value

"""File-name parser."""

import pytest

from sccp_ecni.naming import parse_injection_name
from sccp_ecni.synthetic import example_files


def test_example_filenames_parse() -> None:
    files = example_files()
    assert len(files) == 9
    for item in files:
        parsed = parse_injection_name(item.filename)
        assert parsed.role == item.role
        assert parsed.name == item.name
        assert parsed.replicate == item.replicate
        assert parsed.factor == item.factor
        assert parsed.chlorine == item.declared_chlorine
        assert parsed.concentration == item.concentration


def test_defaults_and_case() -> None:
    parsed = parse_injection_name("SAMPLE__River-01.CSV")
    assert parsed.role == "sample"
    assert parsed.name == "River-01"
    assert parsed.replicate == 1
    assert parsed.factor == 1.0
    assert parsed.concentration is None


def test_rejects_bad_names() -> None:
    for name in (
        "standard__a__cl-55__conc-1.csv",
        "std__has_underscore__cl-55__conc-1.csv",
        "std__sccp__conc-10.csv",
        "std__sccp__cl-55.5__conc-0.csv",
        "sample__a__rep-0.csv",
        "sample__a__factor--1.csv",
        "qc__a__unknown-1.csv",
    ):
        with pytest.raises(ValueError):
            parse_injection_name(name)

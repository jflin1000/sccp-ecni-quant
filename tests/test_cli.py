"""Command-line demo, ion table, and vendor-format rejection."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "sccp_ecni", *args],
        cwd=cwd or ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_ions_lists_the_24_quantifiers() -> None:
    completed = _run("ions")
    assert completed.returncode == 0
    body = [line for line in completed.stdout.splitlines() if line.startswith("C1")]
    assert len(body) == 24
    assert "C10Cl5" in completed.stdout
    assert "279" in completed.stdout
    assert "C13Cl10" in completed.stdout
    assert "493" in completed.stdout


def test_demo_cli_quantifies_the_batch(tmp_path: Path) -> None:
    completed = _run("demo", "--out", str(tmp_path))
    assert completed.returncode == 0, completed.stderr
    assert "pure55" in completed.stdout
    assert "Reth Fa" in completed.stdout
    text = (tmp_path / "quantification.csv").read_text(encoding="utf-8")
    assert "mix" in text
    assert "concentration_nnls" in text
    # The mixture is 3 + 5 concentration units. Formatting keeps the leading digits.
    mix_line = next(line for line in text.splitlines() if line.startswith("sample,mix,"))
    fields = mix_line.split(",")
    # concentration_nnls is the 11th column (index 10) in quantification.csv.
    header = text.splitlines()[0].split(",")
    nnls = float(fields[header.index("concentration_nnls")])
    reth = float(fields[header.index("concentration_reth")])
    assert nnls == pytest.approx(8.0, rel=1e-4)
    assert reth > 0


def test_vendor_file_in_a_folder_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "batch").mkdir()
    (tmp_path / "batch" / "shot.mzML").write_text("<mzML/>", encoding="utf-8")
    completed = _run("run", str(tmp_path / "batch"), "--out", str(tmp_path / "out"))
    assert completed.returncode == 2
    assert "TODO" in completed.stderr

"""Command line for the SCCP GC-ECNI batch pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sccp_ecni import __version__
from sccp_ecni.ions import SCCP_GROUPS
from sccp_ecni.pipeline import format_report, run_folder
from sccp_ecni.synthetic import write_synthetic_batch


def bundled_examples() -> Path | None:
    """Locate ``examples/synthetic_batch`` next to the source tree or in the cwd."""
    candidates = [
        Path(__file__).resolve().parents[2] / "examples" / "synthetic_batch",
        Path.cwd() / "examples" / "synthetic_batch",
    ]
    marker = "std__sccp51__cl-51.5__conc-10__rep-1.csv"
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sccp-ecni",
        description=(
            "Quantify short-chain chlorinated paraffins from unit-resolution "
            "GC-ECNI chromatograms: CSV integration, NNLS deconvolution, and "
            "Reth Fa–D' calibration."
        ),
    )
    parser.add_argument("--version", action="version", version=f"sccp-ecni {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="integrate and quantify a folder of named CSV files")
    run.add_argument("folder", type=Path, help="folder of named chromatogram CSV files")
    run.add_argument("-o", "--out", type=Path, required=True, help="directory for result tables")
    run.add_argument(
        "--no-blank",
        action="store_true",
        help="do not subtract the mean blank Fa vector",
    )

    demo = commands.add_parser("demo", help="quantify the synthetic C10–C13 batch")
    demo.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("sccp_ecni_demo_out"),
        help="directory for result tables (default: ./sccp_ecni_demo_out)",
    )

    commands.add_parser("ions", help="print the fixed 24 SCCP quantifier ions")

    make = commands.add_parser("make-examples", help="write the synthetic example batch")
    make.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("examples") / "synthetic_batch",
        help="output folder (default: ./examples/synthetic_batch)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ions":
            _print_ions()
            return 0
        if args.command == "make-examples":
            files = write_synthetic_batch(args.out)
            print(f"Wrote {len(files)} chromatograms to {args.out}")
            return 0
        if args.command == "demo":
            folder = bundled_examples()
            if folder is None:
                folder = args.out / "synthetic_batch"
                write_synthetic_batch(folder)
                print(f"No bundled examples found; wrote a synthetic batch to {folder}")
            result = run_folder(folder, args.out)
            print(format_report(result))
            return 0
        if args.command == "run":
            result = run_folder(args.folder, args.out, subtract_blank=not args.no_blank)
            print(format_report(result))
            return 0
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command}")
    return 2


def _print_ions() -> None:
    print(f"{'key':<10} {'formula':<14} {'[M-Cl]-':<16} {'m/z':>6} {'Cl%':>8}")
    for group in SCCP_GROUPS:
        print(
            f"{group.key:<10} {group.formula:<14} {group.quantifier_ion:<16} "
            f"{group.mz:6d} {group.chlorine_percent:8.2f}"
        )

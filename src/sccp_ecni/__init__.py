"""Unit-resolution GC-ECNI quantification of short-chain chlorinated paraffins.

The batch pipeline reads a folder of named chromatogram files, integrates the
fixed 24 SCCP quantifier ions (C10–C13, Cl5–Cl10), reconstructs the pattern by
non-negative least squares, and applies the Reth Fa–D' calibration.
"""

from sccp_ecni.ions import N_IONS, SCCP_GROUPS

__version__ = "0.1.0"

__all__ = ["N_IONS", "SCCP_GROUPS", "__version__"]

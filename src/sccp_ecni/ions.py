"""Fixed 24-ion SCCP target list for unit-resolution GC-ECNI-MS.

Short-chain chlorinated paraffins monitored here are the saturated formula
groups C10–C13 with five to ten chlorine atoms. Each group is represented by
the most abundant nominal mass of its [M−Cl]− cluster, which is the quantifier
ion used by unit-resolution electron-capture negative ionisation methods.

Isotope abundances are the standard atomic values 35Cl = 0.7576 and
37Cl = 0.2424. Nominal masses use C = 12, H = 1 and 35Cl = 35, then add 2 u
for every 37Cl in the most abundant isotopologue. Chlorine mass percent uses
standard atomic weights so it can be compared with a bulk chlorine assay.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, log

MASS_C = 12.011
MASS_H = 1.00784
MASS_CL = 35.453
ABUNDANCE_35CL = 0.7576
ABUNDANCE_37CL = 0.2424

N_CARBONS = (10, 11, 12, 13)
N_CHLORINES = (5, 6, 7, 8, 9, 10)
N_IONS = len(N_CARBONS) * len(N_CHLORINES)


@dataclass(frozen=True)
class CongenerGroup:
    """One SCCP formula group and its unit-resolution quantifier ion."""

    carbon: int
    chlorine: int
    hydrogen: int
    formula: str
    quantifier_ion: str
    mz: int
    chlorine_percent: float

    @property
    def key(self) -> str:
        """Short label used in CSV headers and result tables, e.g. ``C10Cl5``."""
        return f"C{self.carbon}Cl{self.chlorine}"


def hydrogen_count(carbon: int, chlorine: int) -> int:
    """Hydrogen count of the saturated formula C_nH_(2n+2−m)Cl_m."""
    hydrogen = 2 * carbon + 2 - chlorine
    if hydrogen < 0:
        raise ValueError(f"C{carbon}Cl{chlorine} is not a saturated chlorinated alkane")
    return hydrogen


def chlorine_mass_percent(carbon: int, chlorine: int) -> float:
    """Chlorine mass percent of C_nH_(2n+2−m)Cl_m from standard atomic weights."""
    hydrogen = hydrogen_count(carbon, chlorine)
    molar_mass = carbon * MASS_C + hydrogen * MASS_H + chlorine * MASS_CL
    return 100.0 * chlorine * MASS_CL / molar_mass


def most_abundant_mz(carbon: int, chlorine: int) -> int:
    """Nominal m/z of the most abundant [M−Cl]− isotopologue.

    The fragment keeps every hydrogen and loses one chlorine, so it is
    C_nH_(2n+2−m)Cl_(m−1). Among the m−1 chlorine atoms, the isotopologue with
    the highest binomial probability is the quantifier.
    """
    hydrogen = hydrogen_count(carbon, chlorine)
    fragment_chlorines = chlorine - 1
    if fragment_chlorines < 1:
        raise ValueError("the [M−Cl]− quantifier needs at least one remaining chlorine")
    monoisotopic = 12 * carbon + hydrogen + 35 * fragment_chlorines
    log_p35 = log(ABUNDANCE_35CL)
    log_p37 = log(ABUNDANCE_37CL)
    best_j = 0
    best_log_p = float("-inf")
    for heavy in range(fragment_chlorines + 1):
        log_p = (
            log(comb(fragment_chlorines, heavy))
            + (fragment_chlorines - heavy) * log_p35
            + heavy * log_p37
        )
        if log_p > best_log_p:
            best_log_p = log_p
            best_j = heavy
    return monoisotopic + 2 * best_j


def _build_groups() -> tuple[CongenerGroup, ...]:
    groups: list[CongenerGroup] = []
    for carbon in N_CARBONS:
        for chlorine in N_CHLORINES:
            hydrogen = hydrogen_count(carbon, chlorine)
            formula = f"C{carbon}H{hydrogen}Cl{chlorine}"
            ion = f"C{carbon}H{hydrogen}Cl{chlorine - 1}-"
            groups.append(
                CongenerGroup(
                    carbon=carbon,
                    chlorine=chlorine,
                    hydrogen=hydrogen,
                    formula=formula,
                    quantifier_ion=ion,
                    mz=most_abundant_mz(carbon, chlorine),
                    chlorine_percent=chlorine_mass_percent(carbon, chlorine),
                )
            )
    return tuple(groups)


SCCP_GROUPS: tuple[CongenerGroup, ...] = _build_groups()
CONGENER_KEYS: tuple[str, ...] = tuple(group.key for group in SCCP_GROUPS)
KEY_INDEX: dict[str, int] = {key: index for index, key in enumerate(CONGENER_KEYS)}
MZ_TO_KEY: dict[int, str] = {group.mz: group.key for group in SCCP_GROUPS}

if len(SCCP_GROUPS) != N_IONS:
    raise RuntimeError("the SCCP target list must contain exactly 24 ions")
if len(MZ_TO_KEY) != N_IONS:
    raise RuntimeError("quantifier m/z values must be unique")


def chlorine_percents() -> tuple[float, ...]:
    """Chlorine mass percent of each formula group, in :data:`SCCP_GROUPS` order."""
    return tuple(group.chlorine_percent for group in SCCP_GROUPS)

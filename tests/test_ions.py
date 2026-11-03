"""Fixed 24-ion C10–C13 quantifier list."""

import pytest

from sccp_ecni.ions import N_IONS, SCCP_GROUPS, chlorine_mass_percent, most_abundant_mz

# Most abundant nominal [M-Cl]- mass for Cl5–Cl10.
_EXPECTED_MZ = {
    10: (279, 313, 347, 381, 417, 451),
    11: (293, 327, 361, 395, 431, 465),
    12: (307, 341, 375, 409, 445, 479),
    13: (321, 355, 389, 423, 459, 493),
}


def test_exactly_24_groups_c10_to_c13() -> None:
    assert N_IONS == 24
    assert len(SCCP_GROUPS) == 24
    carbons = [group.carbon for group in SCCP_GROUPS]
    chlorines = [group.chlorine for group in SCCP_GROUPS]
    assert carbons == sorted(carbons)
    assert set(carbons) == {10, 11, 12, 13}
    assert set(chlorines) == {5, 6, 7, 8, 9, 10}
    assert len({group.mz for group in SCCP_GROUPS}) == 24
    assert [group.key for group in SCCP_GROUPS[:6]] == [
        "C10Cl5",
        "C10Cl6",
        "C10Cl7",
        "C10Cl8",
        "C10Cl9",
        "C10Cl10",
    ]


def test_quantifier_masses_are_the_most_abundant_m_minus_cl() -> None:
    for group in SCCP_GROUPS:
        assert group.mz == _EXPECTED_MZ[group.carbon][group.chlorine - 5]
        assert group.mz == most_abundant_mz(group.carbon, group.chlorine)
        assert group.formula == f"C{group.carbon}H{group.hydrogen}Cl{group.chlorine}"
        assert group.quantifier_ion == f"C{group.carbon}H{group.hydrogen}Cl{group.chlorine - 1}-"
        assert group.hydrogen == 2 * group.carbon + 2 - group.chlorine


def test_chlorine_percent_uses_formula_weight() -> None:
    manual = 100.0 * 5 * 35.453 / (10 * 12.011 + 17 * 1.00784 + 5 * 35.453)
    assert chlorine_mass_percent(10, 5) == SCCP_GROUPS[0].chlorine_percent
    assert SCCP_GROUPS[0].chlorine_percent == pytest.approx(manual)
    low = min(group.chlorine_percent for group in SCCP_GROUPS)
    high = max(group.chlorine_percent for group in SCCP_GROUPS)
    assert 49.0 < low < 51.0
    assert 72.0 < high < 74.0
    # More chlorine on the same chain raises the chlorine mass percent.
    c10 = [group.chlorine_percent for group in SCCP_GROUPS if group.carbon == 10]
    assert c10 == sorted(c10)

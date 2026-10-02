"""Hydrogen termination must preserve the group-IV host's analysis bonds."""

import csv
from itertools import product

import pytest
from ase import Atoms

from amorphgen.analysis import StructureAnalyser
from amorphgen.analysis.structure import is_bonding_pair


@pytest.mark.parametrize("host", ["C", "Si", "Ge"])
@pytest.mark.parametrize("hydrogen", [8, 64])
def test_single_element_host_keeps_its_network(host, hydrogen):
    composition = {host: 64, "H": hydrogen}
    assert is_bonding_pair(host, host, composition)
    assert is_bonding_pair(host, "H", composition)
    assert is_bonding_pair("H", host, composition)
    assert not is_bonding_pair("H", "H", composition)


@pytest.mark.parametrize("hosts", [
    {"C"}, {"Si"}, {"Ge"}, {"Si", "C"}, {"Si", "Ge"},
    {"C", "Ge"}, {"C", "Si", "Ge"},
])
@pytest.mark.parametrize("as_set", [False, True], ids=["counts", "elements"])
def test_hydrogenated_host_pairs_follow_h_free_host(hosts, as_set):
    host_composition = {element: 32 for element in hosts}
    composition = (hosts | {"H"} if as_set
                   else {**host_composition, "H": 8})
    for first, second in product(hosts, repeat=2):
        # SiC and SiGe keep hetero bonds only; H does not turn their
        # same-element host contacts into bonds.
        expected = len(hosts) == 1 or first != second
        assert is_bonding_pair(first, second, host_composition) == expected
        assert is_bonding_pair(first, second, composition) == expected
    for host in hosts:
        assert is_bonding_pair(host, "H", composition)
        assert is_bonding_pair("H", host, composition)
    assert not is_bonding_pair("H", "H", composition)


@pytest.mark.parametrize("host", ["C", "Si", "Ge"])
def test_more_than_one_hydrogen_per_host_keeps_existing_rule(host):
    composition = {host: 64, "H": 65}
    assert not is_bonding_pair(host, host, composition)
    assert is_bonding_pair(host, "H", composition)
    assert is_bonding_pair("H", host, composition)
    assert not is_bonding_pair("H", "H", composition)


@pytest.mark.parametrize("composition, bonds", [
    ({"Li": 1, "H": 1}, {frozenset({"Li", "H"})}),
    ({"Mg": 1, "H": 2}, {frozenset({"Mg", "H"})}),
    ({"Na": 1, "Al": 1, "H": 4},
     {frozenset({"Na", "H"}), frozenset({"Al", "H"})}),
    ({"Ti": 1, "H": 2}, {frozenset({"Ti", "H"})}),
    ({"Na": 1, "O": 1, "H": 1},
     {frozenset({"Na", "O"}), frozenset({"O", "H"})}),
    ({"Mg": 1, "O": 2, "H": 2},
     {frozenset({"Mg", "O"}), frozenset({"O", "H"})}),
    ({"Ca": 1, "C": 1, "O": 3},
     {frozenset({"Ca", "O"}), frozenset({"C", "O"})}),
    ({"Si": 4, "O": 6, "C": 1},
     {frozenset({"Si", "O"}), frozenset({"C", "O"})}),
    ({"Si": 4, "O": 9, "H": 2},
     {frozenset({"Si", "O"}), frozenset({"O", "H"})}),
], ids=["LiH", "MgH2", "NaAlH4", "TiH2", "NaOH", "Mg(OH)2",
        "CaCO3", "Si4O6C", "Si4O9H2"])
def test_other_systems_keep_existing_bond_roles(composition, bonds):
    for first, second in product(composition, repeat=2):
        expected = frozenset({first, second}) in bonds
        assert is_bonding_pair(first, second, composition) == expected


@pytest.mark.parametrize("host", ["C", "Si"])
def test_host_bonds_reach_report_total_angles_and_plot(host, tmp_path):
    # One centre with three host neighbours and two H neighbours, and no
    # leaf-leaf host contacts. A nearby H-H contact must remain non-bonded.
    atoms = Atoms(
        symbols=[host] * 4 + ["H", "H"],
        positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2],
                   [-1, 0, 0], [-1, -0.5, 0]],
        cell=[12, 12, 12], pbc=True,
    )
    analyser = StructureAnalyser([atoms], cutoff=2.1)
    host_pair = f"{host}-{host}"
    coordination = analyser.coordination()
    assert coordination[host_pair]["mean"] == pytest.approx(1.5)
    assert coordination[f"{host}-H"]["mean"] == pytest.approx(0.5)
    assert coordination["H-H"]["mean"] == pytest.approx(1.0)

    total = analyser.total_coordination()
    assert total[host]["mean"] == pytest.approx(2.0)
    assert total[host]["distribution"] == {1: 75.0, 5: 25.0}
    assert total["H"]["mean"] == pytest.approx(1.0)

    angles = analyser.bond_angles()
    assert angles[f"{host}-{host}-{host}"]["count"] == 3
    assert angles[f"{host}-{host}-{host}"]["mean"] == pytest.approx(90.0)
    assert all(triplet.split("-")[1] == host for triplet in angles)

    report = analyser.summary()
    bonded, nonbonded = report.split("Bonding coordination numbers:", 1)[1].split(
        "Non-bonded contacts:", 1)
    # Later sections also report bond/angle ensemble uncertainty.
    nonbonded = nonbonded.split("Bond angles:", 1)[0]
    assert f"{host_pair}: mean=1.5" in bonded
    assert f"{host}-(" in bonded and "mean=2.0" in bonded
    assert "H-H: mean=1.0" in nonbonded
    assert f"{host_pair}:" not in nonbonded

    analyser.plot(output_dir=str(tmp_path), dpi=60)
    assert (tmp_path / "analysis_cn.png").exists()
    with (tmp_path / "analysis_cn.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    plotted_pairs = {row["pair"] for row in rows}
    assert host_pair in plotted_pairs
    total_label = f"{host}-({'+'.join(sorted([host, 'H']))})"
    assert total_label in plotted_pairs
    assert "H-H" not in plotted_pairs

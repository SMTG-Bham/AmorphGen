"""Independent geometry and direct-summation checks for EXAFS numerics."""
import itertools

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from amorphgen.analysis.feff import chi_fourier_transform, feff_cluster


def canonical_cluster(cluster):
    return sorted((symbol, *np.round(position, 8)) for symbol, position in
                  zip(cluster['symbols'], cluster['positions']))


@pytest.mark.parametrize('formula,crystal,a,neighbours,distance', [
    ('Cu', 'fcc', 3.61, 12, 3.61 / np.sqrt(2)),
    ('NiO', 'rocksalt', 4.1684, 6, 4.1684 / 2),
    ('UO2', 'fluorite', 5.4682, 8, 5.4682 * np.sqrt(3) / 4),
])
def test_crystal_cluster_is_independent_of_periodic_representation(
        formula, crystal, a, neighbours, distance):
    atoms = bulk(formula, crystal, a=a, cubic=True)
    original = feff_cluster(atoms, 0, 7.7)
    assert np.count_nonzero(np.isclose(original['distances'], distance)) == neighbours
    repeated = feff_cluster(atoms.repeat((2, 3, 2)), 0, 7.7)
    assert canonical_cluster(repeated) == canonical_cluster(original)
    shifted = atoms.copy()
    shifted.translate([1.173, -2.469, 3.719])
    shifted.wrap()
    assert canonical_cluster(feff_cluster(shifted, 0, 7.7)) == canonical_cluster(original)
    order = np.arange(len(atoms))[::-1]
    permuted = atoms[order]
    assert canonical_cluster(feff_cluster(permuted, len(atoms)-1, 7.7)) == canonical_cluster(original)


def test_skew_periodic_cluster_matches_explicit_image_enumeration():
    atoms = Atoms('GaOO', scaled_positions=[[.92,.17,.31],[.08,.22,.36],[.55,.68,.83]],
                  cell=[[3.2,0,0],[1.1,3.7,0],[-.4,.8,4.1]], pbc=True)
    radius = 5.3
    expected = [('Ga', 0., 0., 0.)]
    # Independent, deliberately oversized integer-image enumeration.
    for shift in itertools.product(range(-5, 6), repeat=3):
        for site, symbol in enumerate(atoms.get_chemical_symbols()):
            if site == 0 and shift == (0, 0, 0):
                continue
            vector = atoms.positions[site] - atoms.positions[0] + np.array(shift) @ atoms.cell
            if np.linalg.norm(vector) < radius:
                expected.append((symbol, *np.round(vector, 8)))
    assert canonical_cluster(feff_cluster(atoms, 0, radius)) == sorted(expected)


@pytest.mark.parametrize('kweight', [0, 1, 2, 3])
@pytest.mark.parametrize('offset', [0., .017])
def test_fourier_complex_amplitude_matches_direct_quadrature(kweight, offset):
    k = np.arange(0, 16.001, .025) + offset
    chi = np.sin(2 * 2.13 * k + .31) * np.exp(-.007*k*k)
    out = chi_fourier_transform(k, chi, kweight=kweight, kmin=3, kmax=12,
                                dk=1, kstep=.05, nfft=2048)
    # Define the window independently, and evaluate the Fourier integral
    # directly: no FFT, no returned production window/weighted signal.
    grid = np.arange(0, 16.001, .05)
    window = np.where((grid >= 3.5) & (grid <= 11.5), 1., 0.)
    rise = (grid > 2.5) & (grid < 3.5)
    fall = (grid > 11.5) & (grid < 12.5)
    window[rise] = np.sin(np.pi*(grid[rise]-2.5)/2)**2
    window[fall] = np.cos(np.pi*(grid[fall]-11.5)/2)**2
    signal = np.interp(grid, k, chi, left=0, right=0) * grid**kweight * window
    r = out['r'][::13]
    expected = .05/np.sqrt(np.pi) * (np.exp(-2j*np.outer(r, grid)) @ signal)
    actual = out['chir_re'][::13] + 1j*out['chir_im'][::13]
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


def potential_populations(text):
    section = text.split('POTENTIALS')[1].split('ATOMS')[0]
    return {int(f[0]): (f[2], float(f[5])) for line in section.splitlines()
            if (f := line.split()) and f[0].isdigit()}


def test_explicit_stoichiometry_is_independent_of_cell_and_cluster_size():
    from amorphgen.analysis.feff import write_feff_input
    atoms = bulk('UO2', 'fluorite', a=5.4682, cubic=True)
    expected = {0: ('U', .001), 1: ('U', 1.), 2: ('O', 2.)}
    for cell in (atoms, atoms.repeat((2, 2, 2))):
        for radius in (5., 8.):
            text = write_feff_input(cell, 0, radius=radius, stoichiometry='structure')
            assert potential_populations(text) == expected


@pytest.mark.parametrize('mode', ['bad', None, 3])
def test_unknown_stoichiometry_is_rejected(mode):
    from amorphgen.analysis.feff import write_feff_input
    with pytest.raises(ValueError, match='stoichiometry must be'):
        write_feff_input(bulk('Cu', 'fcc', a=3.61), 0, stoichiometry=mode)


def test_structure_stoichiometry_rejects_finite_system_and_missing_potential():
    from amorphgen.analysis.feff import write_feff_input
    atoms = bulk('NiO', 'rocksalt', a=4.1684)
    with pytest.raises(ValueError, match='omits'):
        write_feff_input(atoms, 0, radius=1, stoichiometry='structure')
    atoms.pbc = [True, True, False]
    with pytest.raises(ValueError, match='periodicity'):
        write_feff_input(atoms, 0, stoichiometry='structure')


def test_stoichiometry_change_requires_new_tree(tmp_path):
    from amorphgen.analysis.feff import write_feff_jobs
    atoms = bulk('NiO', 'rocksalt', a=4.1684)
    first = write_feff_jobs([atoms], 'Ni', tmp_path, radius=5)
    assert first['stoichiometry'] == 'cluster'
    before = (tmp_path/'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='input changed'):
        write_feff_jobs([atoms], 'Ni', tmp_path, radius=5, stoichiometry='structure')
    assert (tmp_path/'manifest.json').read_bytes() == before


def test_yaml_stoichiometry_values_are_validated():
    from amorphgen.configs.yaml_config import _validate_config
    for mode in ('cluster', 'structure'):
        assert _validate_config({'analysis': {'feff_stoichiometry': mode}}, 'test.yaml') == ([], [])
    _, errors = _validate_config({'analysis': {'feff_stoichiometry': 'bad'}}, 'test.yaml')
    assert any('feff_stoichiometry' in error for error in errors)


@pytest.mark.parametrize('override,expected', [(False, 'structure'), (True, 'cluster')])
def test_cli_stoichiometry_yaml_and_override(tmp_path, monkeypatch, override, expected):
    import json
    import sys
    from ase.io import write
    from amorphgen.cli import main
    structures = tmp_path/'structures'
    structures.mkdir()
    write(structures/'nio.xyz', bulk('NiO', 'rocksalt', a=4.1684), format='extxyz')
    config = tmp_path/'config.yaml'
    config.write_text('analysis:\n  feff_stoichiometry: structure\n')
    output = tmp_path/'feff'
    args = ['amorphgen', '--analyse', '--input-dir', str(structures),
            '--config', str(config), '--feff-write', str(output),
            '--feff-absorber', 'Ni', '--feff-sites', '1']
    if override:
        args.extend(['--feff-stoichiometry', 'cluster'])
    monkeypatch.setattr(sys, 'argv', args)
    main()
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['stoichiometry'] == expected
    text = (output/manifest['jobs'][0]['dir']/'feff.inp').read_text()
    assert ('xnatph' in text) == (expected == 'structure')

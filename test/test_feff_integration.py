"""Opt-in check against the upstream FEFF8.5L non-SCF copper reference."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from amorphgen.analysis.feff import (
    collect_feff_results, feff_cluster, read_chi, run_feff_jobs, write_feff_jobs,
)


@pytest.mark.feff
def test_feff85_copper_reference(tmp_path):
    binary = os.environ.get("FEFF_EXECUTABLE", "feff")
    if shutil.which(binary) is None:
        pytest.fail(f"--run-feff requested but {binary!r} is unavailable")
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    cluster = feff_cluster(atoms, 0, radius=8)
    assert len(cluster["symbols"]) == 177
    assert np.count_nonzero(np.isclose(cluster["distances"], 3.61 / np.sqrt(2))) == 12
    manifest = write_feff_jobs([atoms], "Cu", str(tmp_path), n_sites=1, radius=8,
                               cards={"S02": "0", "RPATH": "5.2"})
    outcome = run_feff_jobs(str(tmp_path), executable=binary, timeout=90)
    assert outcome == {"ran": 1, "skipped": 0, "failed": []}
    k, chi = read_chi(tmp_path / manifest["jobs"][0]["dir"] / "chi.dat")
    kr, cr = read_chi(Path(__file__).parent / "data/feff/copper_chi.dat",
                     kcol=0, chicol=1)
    expected = np.interp(k, kr, cr)
    band = (k >= 2) & (k <= 18)
    relative_rms = np.linalg.norm(chi[band] - expected[band]) / np.linalg.norm(expected[band])
    assert relative_rms < .001, f"FEFF8.5L copper mismatch: {relative_rms:.3%} RMS"
    result = collect_feff_results(tmp_path, kmax=18)
    assert result["n_sites"] == 1 and result["n_structures"] == 1
    assert np.isnan(result["chi_sem"]).all()
    assert result["external_imports"] == []


@pytest.mark.feff
@pytest.mark.parametrize('formula,crystal,a,absorber,edge,rpath,scf,natoms,reference', [
    ('NiO', 'rocksalt', 4.1684, 'Ni', 'K', '5.9', None, 251, 'nio_chi.dat'),
    ('UO2', 'fluorite', 5.4682, 'U', 'L3', '5.5', None, 143, 'uo2_chi.dat'),
    ('Cu', 'fcc', 3.61, 'Cu', 'K', '5.2', '4', 177, 'copper_scf_chi.dat'),
])
def test_feff85_explicit_composition_reference(tmp_path, formula, crystal, a,
        absorber, edge, rpath, scf, natoms, reference):
    binary = os.environ.get('FEFF_EXECUTABLE', 'feff')
    if shutil.which(binary) is None:
        pytest.fail(f'--run-feff requested but {binary!r} is unavailable')
    atoms = bulk(formula, crystal, a=a, cubic=True)
    assert len(feff_cluster(atoms, 0, radius=8)['symbols']) == natoms
    cards = {'S02': '0', 'RPATH': rpath, 'NLEG': '8'}
    if scf is not None:
        cards['SCF'] = scf
    manifest = write_feff_jobs([atoms], absorber, tmp_path, n_sites=1,
        radius=8, edge=edge, cards=cards, stoichiometry='structure')
    assert run_feff_jobs(tmp_path, executable=binary, timeout=180) == {'ran': 1, 'skipped': 0, 'failed': []}
    k, chi = read_chi(tmp_path / manifest['jobs'][0]['dir'] / 'chi.dat')
    kr, cr = read_chi(Path(__file__).parent / 'data/feff' / reference,
                     kcol=0, chicol=1)
    band = (k >= 2) & (k <= 18)
    expected = np.interp(k[band], kr, cr)
    error = np.linalg.norm(chi[band]-expected) / np.linalg.norm(expected)
    assert error < .001, f'{formula} {edge} reference mismatch: {error:.4%} RMS'
    result = collect_feff_results(tmp_path, kmax=18)
    assert result['edge'] == edge
    assert result['external_imports'] == []

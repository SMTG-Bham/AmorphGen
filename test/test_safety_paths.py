"""Regression coverage for safety wiring, output ordering, and YAML settings."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE

from amorphgen.utils.common import DivergenceError


class BadAfterMove(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces']

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        bad = self.atoms.positions[0, 0] > 0.05
        self.results = dict(energy=np.nan if bad else 0.0,
                            free_energy=np.nan if bad else 0.0,
                            forces=np.zeros((len(self.atoms), 3)))


class MoveOnce(FIRE):
    def step(self, *args, **kwargs):
        self.atoms.positions[0, 0] += 0.1


@pytest.mark.parametrize('elapsed', [0, 100])
def test_velocity_initialization_does_not_hide_nan_momenta(elapsed):
    from amorphgen.utils.common import needs_velocity_init
    atoms = bulk('Cu', cubic=True)
    atoms.set_momenta(np.full((len(atoms), 3), np.nan))
    with pytest.raises(DivergenceError, match='Non-finite momenta'):
        needs_velocity_init(atoms, elapsed)


def test_opt_checks_before_trajectory_observers(tmp_path, monkeypatch):
    from amorphgen.pipeline import opt_cell
    monkeypatch.setattr(opt_cell, '_get_optimizer', lambda name: MoveOnce)
    with pytest.raises(DivergenceError, match='step 1'):
        opt_cell.run(bulk('Cu', cubic=True), calc=BadAfterMove(),
                     cfg_override={'opt': {'cell_filter': 'none', 'max_steps': 2}},
                     work_dir=tmp_path)
    # ASE opens the file at construction but must never put the bad frame in it.
    traj = tmp_path / 'stage1_opt.traj'
    if traj.exists():
        from ase.io.trajectory import Trajectory
        with Trajectory(str(traj)) as frames:
            assert len(frames) == 0
    assert not (tmp_path / 'stage1_opt.xyz').exists()


@pytest.mark.parametrize('initial_bad', [False, True])
def test_random_relax_checks_initial_and_each_step(tmp_path, monkeypatch, initial_bad):
    from amorphgen.pipeline import random_gen
    atoms = bulk('Cu', cubic=True)
    if initial_bad:
        atoms.positions[0, 0] = 0.1
    monkeypatch.setattr(random_gen, 'generate_random', lambda *args, **kwargs: atoms.copy())
    monkeypatch.setattr(random_gen, '_get_optimizer_class', lambda name: MoveOnce)
    with pytest.raises(DivergenceError, match='random structure 0000'):
        random_gen.batch_random({'Cu': 4}, output_dir=str(tmp_path), relax=True,
                                calc=BadAfterMove(), cell_filter='none',
                                max_relax_steps=0 if initial_bad else 2)
    assert not list((tmp_path / 'random_opt').glob('*.xyz'))
    assert (tmp_path / 'random_initial' / 'random_0000.xyz').exists()


@pytest.mark.parametrize('module,stage,config,output', [
    ('equilibrate', {'stage': 'low'}, {'eq_low': {'steps': 1, 'T': 300}}, 'stage6_eq.xyz'),
    ('melt_cell', {}, {'melt': {'ensemble': 'NVT', 'T_start': 300, 'T_end': 400,
                              'steps_per_T': 1}}, 'stage3_melted.xyz'),
    ('quench', {}, {'quench': {'T_start': 400, 'T_end': 300, 'steps_per_T': 1}},
     'stage5_quenched.xyz'),
])
def test_md_stages_honor_safety_before_output(tmp_path, module, stage, config, output):
    import importlib
    from ase.calculators.emt import EMT
    run = importlib.import_module('amorphgen.pipeline.' + module).run
    config['safety'] = {'max_temperature': 1.0}
    with pytest.raises(DivergenceError, match='[Tt]emperature'):
        run(bulk('Cu', cubic=True), cfg_override=config, calc=EMT(), work_dir=tmp_path, **stage)
    assert not (tmp_path / output).exists()
    assert not list(tmp_path.glob('*traj.xyz'))


@pytest.mark.parametrize('block', [
    {'safety': {'min_distance': -1}},
    {'safety': {'max_temperatur': 5000}},
    {'safety': {'reference': {'model': 'chgnet', 'interval': 0}}},
    {'safety': {'max_energy_jump_per_atom': float('nan')}},
    {'repulsive_core': {'enabled': 'yes'}},
    {'repulsive_core': {'cutoff': 0}},
    {'repulsive_core': {'strength': True}},
])
def test_yaml_rejects_bad_safety_settings(tmp_path, block):
    import yaml
    from amorphgen.configs.yaml_config import load_yaml_config
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(block))
    with pytest.raises(ValueError):
        load_yaml_config(str(path))


def test_yaml_accepts_optional_controls(tmp_path):
    from amorphgen.configs.yaml_config import load_yaml_config
    path = tmp_path / 'config.yaml'
    path.write_text('''safety:
  min_distance: 0.6
  max_energy_jump_per_atom: null
  reference:
    model: chgnet
    interval: 20
    max_force_rmse: 0.5
repulsive_core:
  enabled: true
  cutoff: 1.2
  strength: 2.0
''')
    cfg = load_yaml_config(str(path))
    assert cfg['safety']['reference']['interval'] == 20
    assert cfg['repulsive_core']['enabled'] is True


def test_random_batch_shares_one_reference_model(tmp_path, monkeypatch):
    from amorphgen.pipeline import random_gen
    atoms = bulk('Cu', cubic=True)
    monkeypatch.setattr(random_gen, 'generate_random', lambda *args, **kwargs: atoms.copy())
    loaded = []

    def load_reference(**kwargs):
        calc = BadAfterMove()
        loaded.append(calc)
        return calc

    monkeypatch.setattr('amorphgen.utils.calculators.get_calculator', load_reference)
    paths = random_gen.batch_random(
        {'Cu': 4}, n_structures=2, output_dir=str(tmp_path), relax=True,
        calc=BadAfterMove(), cell_filter='none', max_relax_steps=0,
        safety={'reference': {'model': 'custom-reference'}})
    assert len(paths) == 2
    assert len(loaded) == 1


@pytest.mark.parametrize('engine', ['ase', 'torchsim'])
def test_cli_passes_safety_and_core_to_random_relaxation(tmp_path, monkeypatch, engine):
    import sys
    import yaml
    from unittest.mock import Mock
    from amorphgen.cli import main

    controls = {'safety': {'min_distance': 0.7},
                'repulsive_core': {'enabled': True, 'cutoff': 1.2}}
    cfg = tmp_path / 'settings.yaml'
    cfg.write_text(yaml.safe_dump(controls))
    generate = Mock(return_value=[])
    optimize = Mock(return_value=[])
    monkeypatch.setattr('amorphgen.pipeline.random_gen._batch_random_unlocked', generate)
    monkeypatch.setattr('amorphgen.pipeline.opt_cell.batch_optimize', optimize)
    monkeypatch.setattr('amorphgen.utils.get_calculator', Mock(return_value=object()))
    monkeypatch.setattr(sys, 'argv', [
        'amorphgen', '--random-gen', '--relax', '--composition', 'Cu=4',
        '--model', 'lj', '--engine', engine, '--config', str(cfg),
        '-o', str(tmp_path / 'out')])
    main()
    for key, value in controls.items():
        assert generate.call_args.kwargs[key] == value
        if engine == 'torchsim':
            assert optimize.call_args.kwargs['cfg_override'][key] == value

"""Selected structures can seed a fresh population search each iteration."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

from gdpx.data.array import AtomsNDArray
from gdpx.workflow.configuration import load_workflow
from gdpx.workflow.nodes.exploration import ExplorationVariable
from gdpx.workflow.state_store import encode_state, decode_state


def search_options(tmp_path):
    source = Path(__file__).resolve().parents[2] / 'examples/workflows/active-learning-dpa4c/sa.yaml'
    options = copy.deepcopy(dict(load_workflow(source).resources['search'].options))
    seed = tmp_path / 'seed.xyz'
    write(seed, Atoms('Cu2', positions=[[4, 5, 5], [6.4, 5, 5]], cell=[10]*3, pbc=True))
    options['system']['builders']['init']['fname'] = str(seed)
    options['strategy']['minimisation']['runtime']['executor']['parameters']['stop']['steps'] = 20
    return options


def test_structure_state_preserves_selection(tmp_path):
    from ase.constraints import FixAtoms
    from ase.calculators.singlepoint import SinglePointCalculator
    frames = [Atoms('Cu', positions=[[x, 0, 0]]) for x in range(3)]
    frames[2].set_constraint(FixAtoms(indices=[0]))
    frames[2].new_array('atom_types', np.array([7]))
    frames[2].calc = SinglePointCalculator(frames[2], energy=1.5)
    value = AtomsNDArray(frames, markers=[[2]])
    record = encode_state(value, tmp_path, 'seeds', 0)
    restored = decode_state(record, tmp_path)
    np.testing.assert_array_equal(restored.markers, value.markers)
    assert restored.get_marked_structures()[0].positions[0, 0] == 2
    selected = restored.get_marked_structures()[0]
    assert selected.get_potential_energy() == 1.5
    assert selected.arrays['atom_types'].tolist() == [7]
    assert selected.constraints[0].get_indices().tolist() == [0]
    assert restored.tolist()[0].calc is None
    assert (tmp_path / 'artifacts/structures/seeds/versions/0000.json').exists()


def test_binding_keeps_other_builders_and_allocations(tmp_path):
    options = search_options(tmp_path)
    system = options['system']
    system['builders']['fresh'] = copy.deepcopy(system['builders']['init'])
    system['initial'] = {'total_size': 2, 'builder_allocations': [
        {'builder': 'init', 'size': 1}, {'builder': 'fresh', 'size': 1}]}
    engine = ExplorationVariable(**options).value
    fresh = engine.builders['fresh']
    allocations = copy.deepcopy(engine.population_config.initial_builder_allocations)
    candidates = [Atoms('Cu2', positions=[[4, 5, 5], [6.2, 5, 5]], cell=[10]*3, pbc=True),
                  Atoms('Cu2', positions=[[4, 5, 5], [6.3, 5, 5]], cell=[10]*3, pbc=True)]
    engine.bind_initial_builder('init', AtomsNDArray(candidates, markers=[[1]]), tmp_path / 'bound')
    assert engine.builders['fresh'] is fresh
    assert engine.population_config.initial_builder_allocations == allocations
    frames = engine.population_config._prepare_initial_population(engine.builders)
    assert frames[0].positions[1, 0] == 6.3
    assert frames[1].positions[1, 0] == 6.4
    assert frames[0].calc is None
    assert isinstance(engine.builders['init'].as_dict()['frames'], str)
    with pytest.raises(ValueError, match='requires 1 structures'):
        engine.bind_initial_builder('init', [], tmp_path)
    with pytest.raises(ValueError, match='must exist'):
        engine.bind_initial_builder('missing', candidates, tmp_path)


def test_repeat_sa_uses_selected_candidates_and_resumes(tmp_path, monkeypatch):
    from gdpx.workflow.nodes import data, driver, selector, runtime
    from gdpx.workflow.session.repeat import RepeatSession
    from gdpx.exploration.population.config import PopulationConfig
    from gdpx.exploration.population.exploration import PopulationBasedExploration
    options = search_options(tmp_path)
    starts = []
    original = PopulationConfig._prepare_initial_population
    def capture(self, builders):
        frames = original(self, builders)
        starts.append([a.copy() for a in frames])
        return frames
    monkeypatch.setattr(PopulationConfig, '_prepare_initial_population', capture)
    spec_data = {
        'workflow': {'mode': 'repeat', 'targets': 'pick', 'max_iterations': 2},
        'state': {'seeds': {'initial': None, 'update': 'pick'}},
        'resources': {
            'search': {'__type__': 'exploration', 'options': options},
            'runtime': {'__type__': 'runtime', 'options': {
                'potential': {'provider': 'emt'},
                'executor': {'provider': 'ase', 'method': 'md', 'parameters': {
                    'ensemble': {'method': 'nvt', 'temperature': 600.},
                    'setup': {'timestep': 0.5, 'regulator': {'name': 'langevin', 'parameters': {'friction': 0.01}}},
                    'stop': {'steps': 2}, 'output': {'trajectory': {'period': 1}}}}}},
            'selector': {'__type__': 'selector', 'options': {'selection': {'method': 'locate', 'indices': '-1'}}}},
        'steps': {
            'sample': {'__type__': 'explore', 'inputs': {'exploration': 'search', 'runtime': 'runtime', 'builder_init': 'seeds'}},
            'pick': {'__type__': 'select', 'inputs': {'structures': 'sample', 'selector': 'selector'}}}}
    path = tmp_path / 'loop.yaml'
    path.write_text(yaml.safe_dump(spec_data))
    spec = load_workflow(path)
    session = RepeatSession(spec, tmp_path / 'run')
    bind = PopulationBasedExploration.bind_initial_builder
    def interrupt(*args, **kwargs):
        raise RuntimeError('simulated interruption before iteration 1')
    monkeypatch.setattr(PopulationBasedExploration, 'bind_initial_builder', interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        session.run()
    assert len(starts) == 1
    monkeypatch.setattr(PopulationBasedExploration, 'bind_initial_builder', bind)
    RepeatSession(spec, tmp_path / 'run').run()
    assert len(starts) == 2
    selected = decode_state({'kind': 'structures', 'artifact': 'artifacts/structures/seeds/versions/0000.json'}, tmp_path / 'run').get_marked_structures()
    np.testing.assert_allclose(starts[1][0].positions, selected[0].positions)
    assert not np.allclose(starts[0][0].positions, starts[1][0].positions)
    # Restarting a completed workflow preserves committed seeds without rerunning SA.
    RepeatSession(spec, tmp_path / 'run').run()
    assert len(starts) == 2

"""Hybrid biased MC keeps MD stages and relaxes proposals inside MC blocks."""
import copy
from pathlib import Path

import pytest
import yaml

from gdpx.execution import resolve_runtime
from gdpx.execution.monte_carlo.biased_monte_carlo import BiasedMonteCarlo
from gdpx.execution.monte_carlo.hybrid_monte_carlo import HybridBiasedMonteCarlo
from gdpx.exploration.factory import create_exploration
from test_sampling_config import sampling_config


def test_hybrid_biased_configuration_and_roundtrip():
    config = sampling_config('hybrid_biased_monte_carlo')
    original = copy.deepcopy(config)
    runtime = resolve_runtime(config)
    engine = runtime.executor
    assert isinstance(engine, HybridBiasedMonteCarlo)
    assert [child.config.executor.method for child in engine.children] == ['md', 'biased_monte_carlo']
    child = engine.children[1].executor
    assert isinstance(child, BiasedMonteCarlo)
    assert child.worker.runtime.config.executor.method == 'min'
    assert child.worker.runtime.config.executor.parameters['setup']['optimizer']['name'] == 'bfgs'
    assert engine.initializer.config.executor.method == 'min'
    assert all(stage.config.potential == runtime.config.potential for stage in engine.children)
    assert resolve_runtime(engine.as_dict()).executor.as_dict() == engine.as_dict()
    assert config == original


@pytest.mark.parametrize('ensemble', ['nvt', 'uvt', 'suvt'])
def test_ensemble_is_inherited_by_md_and_biased_mc(ensemble):
    config = sampling_config('hybrid_biased_monte_carlo')
    settings = {'method': ensemble, 'temperature': 900.}
    if ensemble != 'nvt':
        settings['chemical_potentials'] = {'Cu': -3.}
    config['executor']['parameters']['ensemble'] = settings
    children = resolve_runtime(config).executor.children
    assert children[0].config.executor.parameters['ensemble']['method'] == 'nvt'
    assert children[1].config.executor.parameters['ensemble']['method'] == ensemble


@pytest.mark.parametrize('stage,method', [(0, 'min'), (1, 'monte_carlo')])
def test_rejects_wrong_stage_methods(stage, method):
    config = sampling_config('hybrid_biased_monte_carlo')
    config['executor']['parameters']['setup'][stage]['method'] = method
    with pytest.raises(ValueError):
        resolve_runtime(config)


def test_requires_biased_mc_minimisation_limits():
    config = sampling_config('hybrid_biased_monte_carlo')
    del config['executor']['parameters']['setup'][1]['parameters']['setup']['evaluation']['stop']
    with pytest.raises(ValueError, match='minimisation requires'):
        resolve_runtime(config)


def test_hybrid_biased_is_a_compute_method():
    with pytest.raises(ValueError, match='compute'):
        create_exploration({'method': 'hybrid_biased_monte_carlo'})


def test_documented_example_resolves():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / 'examples/monte_carlo/hybrid-biased-canonical.yaml').read_text())
    assert isinstance(resolve_runtime(config).executor, HybridBiasedMonteCarlo)

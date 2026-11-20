import copy

import pytest

from gdpx.exploration.simulated_annealing.protocol import simulation_executor


def protocol():
    return dict(provider="ase", method="md", parameters=dict(
        setup=dict(ensemble=dict(method="nvt")), stop=dict(steps=10)))


@pytest.mark.parametrize("ensemble", ["nvt", "npt"])
def test_injects_temperature_without_mutating_protocol(ensemble):
    value = protocol()
    value['parameters']['setup']['ensemble']['method'] = ensemble
    if ensemble == 'npt':
        value['parameters']['setup']['ensemble']['pressure'] = 1.0
    original = copy.deepcopy(value)
    prepared = simulation_executor(value, 600)
    assert prepared['parameters']['setup']['ensemble']['temperature'] == 600
    assert value == original


@pytest.mark.parametrize('field', ['potential', 'dispatch', 'scheduler', 'executor'])
def test_rejects_runtime_overrides(field):
    value = protocol()
    value[field] = {}
    with pytest.raises(ValueError, match='accepts only'):
        simulation_executor(value)


def test_rejects_duplicate_temperature():
    value = protocol()
    value['parameters']['setup']['ensemble']['temperature'] = 600
    with pytest.raises(ValueError, match='strategy.temperatures'):
        simulation_executor(value)


@pytest.mark.parametrize('fmax', [None, True, 0, -1, float('nan'), float('inf')])
def test_invalid_minimum_force_tolerance(fmax):
    from gdpx.providers.configuration import RuntimeConfig
    from gdpx.exploration.simulated_annealing.protocol import validate_minimum
    value = RuntimeConfig.from_mapping(dict(potential=dict(provider='emt'), executor=dict(
        provider='ase', method='min', parameters=dict(stop=dict(steps=10, fmax=fmax)))))
    with pytest.raises(ValueError, match='positive stop.steps and stop.fmax'):
        validate_minimum(value)

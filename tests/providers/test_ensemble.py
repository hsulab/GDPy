"""Shared thermodynamic validation and MD parameter translation."""
import copy

import pytest

from gdpx.providers.configuration import resolve_executor_parameters
from gdpx.providers.ensemble import resolve_ensemble


@pytest.mark.parametrize("ensemble, expected", [
    ({"method": "nve"}, {"ensemble": "nve"}),
    ({"method": "nvt", "temperature": 600}, {"ensemble": "nvt", "temp": 600}),
    ({"method": "npt", "temperature": 600, "pressure": 2},
     {"ensemble": "npt", "temp": 600, "press": 2}),
])
def test_md_ensemble_translation(ensemble, expected):
    source = {"ensemble": ensemble, "setup": {}, "stop": {"steps": 4}}
    if ensemble["method"] != "nve":
        source["setup"]["regulator"] = {"name": "berendsen", "parameters": {"Tdamp": 100}}
    original = copy.deepcopy(source)
    params = resolve_executor_parameters(source, "md")
    assert all(params[key] == value for key, value in expected.items())
    assert params["steps"] == 4
    assert source == original


@pytest.mark.parametrize("ensemble", [
    None, "nvt", {}, {"method": "canonical"}, {"method": "custom"},
    {"method": "nvt"}, {"method": "nvt", "temperature": True},
    {"method": "nvt", "temperature": 0}, {"method": "nvt", "temperature": -1},
    {"method": "nvt", "temperature": float("nan")},
    {"method": "nvt", "temperature": float("inf")},
    {"method": "nvt", "temperature": 300, "pressure": 1},
    {"method": "nve", "temperature": 300},
    {"method": "npt", "temperature": 300},
    {"method": "npt", "temperature": 300, "pressure": True},
    {"method": "npt", "temperature": 300, "pressure": float("nan")},
])
def test_invalid_md_ensembles(ensemble):
    with pytest.raises(ValueError, match="ensemble"):
        resolve_ensemble(ensemble, "md")


@pytest.mark.parametrize("chemicals", [None, {}, {"Cu": True}, {"Cu": float("inf")}, {"": 0}, {"Cu": "0"}])
def test_invalid_chemical_potentials(chemicals):
    with pytest.raises(ValueError, match="chemical_potentials"):
        resolve_ensemble({"method": "uvt", "temperature": 300, "chemical_potentials": chemicals}, "monte_carlo")


@pytest.mark.parametrize("setup", [
    {"ensemble": "nvt"},
    {"regulator": {"name": "berendsen", "targets": {"temperature": 300}}},
    {"temp": 500},
    {"velocities": {"temperature": 500}},
])
def test_duplicate_or_legacy_md_targets_rejected(setup):
    with pytest.raises(ValueError):
        resolve_executor_parameters({"ensemble": {"method": "nvt", "temperature": 300}, "setup": setup}, "md")


def test_nve_initial_velocity_temperature_is_separate():
    params = resolve_executor_parameters({"ensemble": {"method": "nve"},
        "setup": {"velocities": {"temperature": 300}}}, "md")
    assert params["ensemble"] == "nve" and params["temp"] == 300


@pytest.mark.parametrize("method", ["uvt", "suvt"])
def test_md_rejects_particle_exchange_ensembles(method):
    with pytest.raises(ValueError, match="does not support"):
        resolve_executor_parameters({"ensemble": {"method": method, "temperature": 300,
            "chemical_potentials": {"Cu": 0}}, "setup": {}}, "md")


@pytest.mark.parametrize("parameters", [{}, {"ensemble": None}, {"ensemble": "nvt"}])
def test_mc_requires_ensemble_mapping(parameters):
    with pytest.raises(ValueError, match="ensemble"):
        resolve_executor_parameters(parameters, "monte_carlo")


def test_md_rejects_explicit_null_ensemble():
    with pytest.raises(ValueError, match="ensemble"):
        resolve_executor_parameters({"ensemble": None, "setup": {}}, "md")

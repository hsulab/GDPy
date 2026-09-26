"""xreac adapter contract and optional real-calculator integration."""

import copy
import sys
import types

import numpy as np
import pytest
from ase.build import molecule

from gdpx.providers import CapabilityKind, get_provider_manager
from gdpx.providers.reax.manager import ReaxManager


def test_reax_exposes_one_implementation_per_executor_target():
    targets = get_provider_manager().list_capabilities("reax")[CapabilityKind.MATERIALIZER]
    assert set(targets) == {"ase.calculator", "lammps.potential"}


def test_reax_config_round_trips_without_backend():
    from gdpx.providers import RuntimeConfig

    source = {
        "schema_version": 4,
        "potential": {"provider": "reax", "parameters": {"model": "ffield"}},
        "executor": {"provider": "ase", "method": "min", "parameters": {}},
        "modifiers": [],
        "dispatch": {
            "worker": "batch",
            "batch_size": 1,
            "share_workdir": False,
            "retain_info": False,
        },
    }
    config = RuntimeConfig.from_mapping(source)
    assert config.to_dict() == source
    assert config.potential_spec().provider == "reax"


def test_lammps_executor_selects_reax_c(tmp_path):
    model = tmp_path / "ffield"
    model.touch()
    runtime = get_provider_manager().resolve_runtime({
        "schema_version": 4,
        "potential": {"provider": "reax", "parameters": {
            "model": str(model), "type_list": ["H", "O"], "command": "lmp",
        }},
        "executor": {"provider": "lammps", "method": "min", "parameters": {}},
    })
    result = runtime.materialization
    assert result.commands == ("pair_style reax/c NULL", f"pair_coeff * * {model}")
    assert result.calculator.units == "real"
    assert result.calculator.atom_style == "charge"


@pytest.mark.parametrize("model", [None, "", " ", []])
def test_ase_requires_explicit_model(model):
    with pytest.raises(ValueError, match="non-empty"):
        ReaxManager().register_calculator({"backend": "xreac", "model": model})


def test_missing_optional_dependency_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "xreac", None)
    with pytest.raises(ModuleNotFoundError, match="gdpx\\[reax\\]"):
        ReaxManager().register_calculator({"backend": "xreac", "model": "bundled:ffield"})


def test_local_model_is_resolved_and_options_forwarded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loaded = []
    field = object()
    module = types.ModuleType("xreac")
    module.ForceField = types.SimpleNamespace(from_file=lambda path: loaded.append(path) or field)
    adapter = types.ModuleType("xreac.ase")
    adapter.ReaxFFCalculator = lambda force_field, **kwargs: (force_field, kwargs)
    monkeypatch.setitem(sys.modules, "xreac", module)
    monkeypatch.setitem(sys.modules, "xreac.ase", adapter)
    parameters = {"backend": "xreac", "model": "ffield", "neighbor_backend": "replicated"}
    original = copy.deepcopy(parameters)
    manager = ReaxManager()
    manager.register_calculator(parameters)
    assert loaded == [str(tmp_path / "ffield")]
    assert manager.calc_params["model"] == loaded[0]
    assert manager.calc == (field, {"neighbor_backend": "replicated"})
    assert parameters == original


def test_real_xreac_runtime_matches_core_units_and_forces():
    xreac = pytest.importorskip("xreac")
    from ase.units import kcal, mol

    runtime = get_provider_manager().resolve_runtime({
        "schema_version": 4,
        "potential": {"provider": "reax", "parameters": {"model": "bundled:ffield.reax.HO.2015"}},
        "executor": {"provider": "ase", "method": "spc", "parameters": {}},
    })
    atoms = molecule("H2O")
    atoms.calc = runtime.materialization.calculator
    reference = xreac.Calculator(xreac.ForceField.bundled("ffield.reax.HO.2015")).evaluate(
        atoms.get_chemical_symbols(), atoms.positions
    )
    assert atoms.get_potential_energy() == pytest.approx(reference.energy * kcal / mol)
    np.testing.assert_allclose(atoms.get_forces(), reference.forces * kcal / mol, atol=1e-10)
    np.testing.assert_allclose(atoms.get_charges(), reference.charges, atol=1e-10)
    assert "stress" not in atoms.calc.implemented_properties


def test_missing_local_file_is_not_treated_as_bundled(tmp_path):
    pytest.importorskip("xreac")
    with pytest.raises(FileNotFoundError):
        ReaxManager().register_calculator({"backend": "xreac", "model": str(tmp_path / "missing")})

import copy
from unittest.mock import patch

import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp

from gdpx.providers.configuration import resolve_executor_parameters
from gdpx.providers.errors import ProviderConfigurationError
from gdpx.providers.vasp.driver import VaspDriver


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("is_restart", [False, True])
def test_common_policy_maps_to_wavefunction_only(tmp_path, enabled, is_restart):
    public = {"output": {"electronic_checkpoint": {"enabled": enabled}}, "stop": {"steps": 20}}
    original = copy.deepcopy(public)
    params = resolve_executor_parameters(public, "min")
    assert public == original
    calc = Vasp(lwave=not enabled, lcharg=True)
    driver = VaspDriver(calc, dict(params, task="min"), directory=tmp_path)
    atoms = Atoms("H", cell=[5, 5, 5], pbc=True)
    observed = []
    def inspect(*args, **kwargs):
        observed.append((calc.bool_params["lwave"], calc.bool_params["lcharg"]))
    checkpoint = None
    if is_restart:
        checkpoint = tmp_path / "0000.run"
        checkpoint.mkdir()
        (checkpoint / "INCAR").write_text("LWAVE = .FALSE.\nLCHARG = .TRUE.\n")
    with patch.object(calc, "write_input", side_effect=inspect), patch(
        "gdpx.providers.vasp.driver.prepare_restart_inputs", side_effect=inspect
    ), patch("gdpx.providers.vasp.driver.run_ase_calculator"):
        driver._irun(atoms, ckpt_wdir=checkpoint, cache_traj=[atoms])
    assert observed == [(enabled, False)]


def test_omitted_policy_preserves_native_settings(tmp_path):
    calc = Vasp(lwave=False, lcharg=True)
    driver = VaspDriver(calc, {"task": "min"}, directory=tmp_path)
    driver._apply_electronic_checkpoint_policy()
    assert calc.bool_params["lwave"] is False
    assert calc.bool_params["lcharg"] is True


def test_other_output_controls_remain_available():
    result = resolve_executor_parameters({
        "output": {
            "trajectory": {"period": 2},
            "checkpoint": {"period": 10, "keep": 3},
            "electronic_checkpoint": {"enabled": True},
        }
    }, "min")
    assert result == {
        "dump_period": 2, "ckpt_period": 10, "ckpt_number": 3,
        "electronic_checkpoint": {"enabled": True, "keep": 2},
    }


@pytest.mark.parametrize("value", [True, None, [], {"enabled": "true"}, {"enabled": 1}, {"wavecar": True}])
def test_invalid_common_policy_rejected(value):
    with pytest.raises(ProviderConfigurationError):
        resolve_executor_parameters({"output": {"electronic_checkpoint": value}}, "min")


def test_empty_block_is_disabled():
    assert resolve_executor_parameters(
        {"output": {"electronic_checkpoint": {}}}, "min"
    ) == {"electronic_checkpoint": {"enabled": False, "keep": 2}}


def test_unsupported_hdf5_checkpoint_is_rejected(tmp_path):
    calc = Vasp()
    calc.bool_params["lh5"] = True
    driver = VaspDriver(calc, {"task": "min", "electronic_checkpoint": {"enabled": True}}, directory=tmp_path)
    with pytest.raises(ValueError, match="LH5"):
        driver._apply_electronic_checkpoint_policy()

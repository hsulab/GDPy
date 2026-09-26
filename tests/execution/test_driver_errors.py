from types import SimpleNamespace

import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from gdpx.execution.driver import BaseDriver, DriverSetting
from gdpx.execution.workers import DriverBatchError, run_computation_in_commandline


class _Calculator(Calculator):
    implemented_properties = []
    command = "engine"


class _FailingDriver(BaseDriver):
    name = "failing"
    setting_cls = DriverSetting

    def _irun(self, atoms, *args, **kwargs):
        self.calc.parameters["changed"] = True
        raise ValueError("backend exploded")

    def read_trajectory(self, *args, **kwargs):
        return []


class _BatchDriver:
    name = "batch-test"

    def __init__(self, failures=()):
        self.setting = SimpleNamespace(machine_prefix="original-prefix")
        self.random_seed = 17
        self.failures = set(failures)
        self.attempted = []

    def set_rng(self, seed):
        self.random_seed = seed

    def reset(self):
        pass

    def run(self, atoms, **kwargs):
        index = atoms.info["index"]
        self.attempted.append(index)
        if index in self.failures:
            raise ValueError(f"failure-{index}")


def _structures(count):
    frames = []
    for index in range(count):
        atoms = Atoms("H")
        atoms.info["index"] = index
        frames.append(atoms)
    return frames


def test_driver_run_propagates_original_error_and_restores_calculator(tmp_path):
    calculator = _Calculator(test_value=1)
    driver = _FailingDriver(calculator, {}, directory=tmp_path / "calculation")

    with pytest.raises(ValueError, match="backend exploded"):
        driver.run(Atoms("H"))

    assert calculator.command == "engine"
    assert calculator.parameters == {"test_value": 1}


def test_batch_attempts_all_structures_then_raises_aggregate(tmp_path):
    driver = _BatchDriver(failures={0, 2})
    messages = []

    with pytest.raises(DriverBatchError) as caught:
        run_computation_in_commandline(
            identifier="input-id",
            structures=_structures(3),
            computation_dirnames=["cand0", "cand1", "cand2"],
            rng_states=[100, 101, 102],
            drivers=[driver],
            driver_indices=None,
            directory=tmp_path,
            share_wdir=False,
            print_func=messages.append,
        )

    error = caught.value
    assert driver.attempted == [0, 1, 2]
    assert [failure.workdir for failure in error.failures] == ["cand0", "cand2"]
    assert [str(failure.exception) for failure in error.failures] == ["failure-0", "failure-2"]
    assert "ValueError: failure-0" in error.failures[0].traceback
    assert isinstance(error.__cause__, ValueError)
    assert driver.random_seed == 17
    assert driver.setting.machine_prefix == "original-prefix"
    assert sum("ERROR: driver computation failed" in message for message in messages) == 2


def test_batch_does_not_swallow_process_control_exceptions(tmp_path):
    driver = _BatchDriver()

    def interrupt(atoms, **kwargs):
        raise KeyboardInterrupt

    driver.run = interrupt

    with pytest.raises(KeyboardInterrupt):
        run_computation_in_commandline(
            identifier="input-id",
            structures=_structures(1),
            computation_dirnames=["cand0"],
            rng_states=[100],
            drivers=[driver],
            driver_indices=None,
            directory=tmp_path,
            share_wdir=False,
            print_func=lambda message: None,
        )

    assert driver.random_seed == 17
    assert driver.setting.machine_prefix == "original-prefix"

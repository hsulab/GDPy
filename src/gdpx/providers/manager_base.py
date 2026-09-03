#!/usr/bin/env python3
# -*- coding: utf-8 -*


import abc
import copy
import warnings
from typing import Any, Generic, TypeVar, cast

from ase.calculators.calculator import Calculator

from gdpx.providers.ase.backend import DummyCalculator


CalcT = TypeVar("CalcT", bound=Calculator)


class BasePotentialManager(abc.ABC, Generic[CalcT]):
    """
    Create various potential instances
    """

    #: Name of the potential.
    name: str = "potential"

    #: Supported calculator backends.
    implemented_backends: tuple[str, ...] = ()

    #: Supported combinations of calculator backend and driver/engine.
    valid_combinations: tuple = ()

    #: The attached calculator.
    _calc: CalcT

    def __init__(self):
        """"""
        #: Attached calculator.
        self._calc: CalcT = cast(CalcT, DummyCalculator())

        #: The default backend.
        self._default_backend: str = self.implemented_backends[0]

        return

    @property
    def calc(self) -> CalcT:
        """Attached calculator."""

        return self._calc

    @calc.setter
    def calc(self, calc: CalcT) -> None:
        """Set the attached calculator."""
        self._calc = calc

        return

    @property
    def calc_backend(self) -> str:
        """Backend of attached calculator."""

        return self.calc_params.get("backend", self._default_backend)

    @abc.abstractmethod
    def register_calculator(self, calc_params: dict, *agrs, **kwargs):
        """Register the host calculator."""
        # Save the original copy of calc_params and pop the backend keyword
        # as it is not for calculator.
        self.calc_params = copy.deepcopy(calc_params)
        calc_params.pop("backend", None)

        if self.calc_backend not in self.implemented_backends:
            raise RuntimeError(
                f"Unknown backend {self.calc_backend} for potential {self.name} with {self.implemented_backends}."
            )

        return

    def create_driver(self, dyn_params: dict = None) -> Any:
        """Create a driver for dynamics.

        The default dynamics backend will be the same as the calculator.
        The ase-based dynamics can be used for all calculators.

        Args:
            dyn_params: Parameters for driver.

        Returns:
            A driver instance.

        """
        warnings.warn(
            "PotentialManager.create_driver() is deprecated; resolve a Runtime through ProviderManager instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from gdpx.execution.compat import create_compat_executor

        return create_compat_executor(self, dyn_params or {})

    def as_dict(self):
        """"""
        params = {}
        params["name"] = self.name
        params["params"] = copy.deepcopy(self.calc_params)

        return params


if __name__ == "__main__":
    ...

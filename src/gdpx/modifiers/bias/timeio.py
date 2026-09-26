#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

from ase.calculators.calculator import Calculator


class TimeIOCalculator(Calculator):

    def __init__(self, pace: int = 1, delay: int = 0, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.pace = pace

        self.delay = delay

        self._num_steps = 0

        return

    @property
    def num_steps(self) -> int:
        """Finished steps that match the host driver e.g. MD."""

        return self._num_steps

    @property
    def log_fpath(self):
        """"""

        return pathlib.Path(self.directory) / "calc.log"

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=["positions", "numbers", "cell"],
    ):
        super().calculate(atoms, properties, system_changes)

        if self.num_steps == 0:
            self._write_first_step()

        self.results, self.step_info = self._icalculate(
            atoms, properties, system_changes
        )

        if self.num_steps % self.pace == 0:
            self._write_step()

        self._num_steps += 1

        return

    def _icalculate(self, atoms, properties, system_changes):
        """"""

        raise NotImplementedError()

    def _write_first_step(self):
        """"""

        raise NotImplementedError()

    def _write_step(self):
        """"""

        raise NotImplementedError()


if __name__ == "__main__":
    ...

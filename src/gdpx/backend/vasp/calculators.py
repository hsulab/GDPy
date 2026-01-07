import pathlib
from typing import Optional

from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.mixing import LinearCombinationCalculator


class VaspInteractiveWithDispersion(LinearCombinationCalculator):
    def __init__(self, calcs, save_host: bool = True, directory: str = "./"):
        """"""
        self._directory = directory

        super().__init__(calcs=calcs, weights=[1.0, 1.0])

        self.save_host = save_host

        return

    @property
    def directory(self) -> str:
        """"""

        return self._directory

    @directory.setter
    def directory(self, directory):
        """"""
        self._directory = directory

        return

    def reset(self):
        """Clear all information from old calculation."""

        self.atoms = None
        self.results = {}

        for calc in self.mixer.calcs:
            calc.reset()

        return

    def calculate(self, atoms: Optional[Atoms] = None, properties=["energy"], system_changes=all_changes):
        """"""
        for i, calc in enumerate(self.mixer.calcs):
            calc.directory = str((pathlib.Path(self.directory) / (f"{i:>02d}.{calc.__class__.__name__}")).resolve())

        assert isinstance(atoms, Atoms)

        vasp_calc, disp_calc = self.mixer.calcs

        vasp_calc.calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        with vasp_calc.pause():
            disp_calc.calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        results = {}
        for prop in properties:
            results[prop] = vasp_calc.results[prop] + disp_calc.results[prop]
        self.results = results

        if self.save_host:
            self.results["host_energy"] = self.mixer.calcs[0].get_property("energy", atoms)
            self.results["host_forces"] = self.mixer.calcs[0].get_property("forces", atoms)

        return

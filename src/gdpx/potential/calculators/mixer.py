#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np

from ase.calculators.calculator import all_changes
from ase.calculators.mixing import MixedCalculator, LinearCombinationCalculator

from .. import config as GDPCONFIG

class AddonCalculator(MixedCalculator):

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """"""
        super().calculate(atoms, properties, system_changes)

        if "forces" in properties:
            forces1 = self.calcs[0].get_property("forces", atoms)
            forces2 = self.calcs[1].get_property("forces", atoms)
            self.results["force_contributions"] = np.hstack([forces1,forces2])

        return


class EnhancedCalculator(LinearCombinationCalculator):

    def __init__(self, calcs, save_host=True, weights=None, atoms=None, directory: str="./"):
        """Init the enhanced calculator.

        Args:
            calcs: Calculators.
            save_host: Whether save host energy and forces.
        """
        if weights is None:
            weights = np.ones(len(calcs))
        self._directory = directory

        super().__init__(calcs, weights)

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


    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """"""
        # - for sub calculators...
        for i, subcalc in enumerate(self.mixer.calcs):
            subcalc.directory = str(
                (pathlib.Path(self.directory)/(str(i).zfill(2)+"."+subcalc.__class__.__name__)).resolve()
            )
        # NOTE: nequip requires that atoms has NequipCalculator or None
        #       thus, we set atoms.calc to None and restore it later
        prev_calc = atoms.calc
        atoms.calc = None

        super().calculate(atoms, properties, system_changes)
        atoms.calc = prev_calc

        if self.save_host:
            self.results["host_energy"] = self.mixer.calcs[0].get_property("energy", atoms)
            self.results["host_forces"] = self.mixer.calcs[0].get_property("forces", atoms)
        
        # - save deviation if the host calculator is a committee
        if isinstance(self.mixer.calcs[0], CommitteeCalculator):
            natoms = len(atoms)
            for k, v in self.mixer.calcs[0].results.items():
                if k in GDPCONFIG.VALID_DEVI_FRAME_KEYS:
                    self.results[k] = v
            for k, v in self.mixer.calcs[0].results.items():
                if k in GDPCONFIG.VALID_DEVI_ATOMIC_KEYS:
                    self.results[k] = np.reshape(v, (natoms, -1))

        return
    

class CommitteeCalculator(LinearCombinationCalculator):

    def __init__(self, calcs, use_avg=False, save_atomic=True, ddof=0, directory: str="./"):
        """Init the committee calculator.

        Args:
            calcs: ASE calculators.
            use_avg: Whether use average results instead of those by the first calc.
            save_atomic: Whether save atomic deviation.
            ddof: Dela Degrees of Freedom that affects the deviations of results.

        """
        weights = np.ones(len(calcs))
        if use_avg:
            weights = weights / np.sum(weights)
        else:
            weights[1:] = 0.
        self.ddof = ddof
        self.save_atomic = save_atomic

        self._directory = directory

        super().__init__(calcs, weights)

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

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """"""
        # NOTE: nequip requires that atoms has NequipCalculator or None
        #       thus, we set atoms.calc to None and restore it later
        prev_calc = atoms.calc
        atoms.calc = None

        # TODO: set wdir for each subcalc
        super().calculate(atoms, properties, system_changes)
        atoms.calc = prev_calc

        self._compute_deviation(atoms, properties)

        return
    
    def _compute_deviation(self, atoms, properties):
        """Compute the RMSE deviation of calculator properties."""
        # We directly check contributions in results as some properties not in `properties`
        # are still calculated.
        if "energy_contributions" in self.results:
            # tot_energies = np.array([c.results["energy"] for c in self.calcs])
            tot_energies = np.array(self.results["energy_contributions"])
            self.results["devi_te"] = np.sqrt(np.var(tot_energies, ddof=self.ddof))

        if "forces_contributions" in self.results:
            # cmt_forces = np.array([c.results["forces"].flatten() for c in self.calcs])
            cmt_forces = np.array([contrib.flatten() for contrib in self.results["forces_contributions"]])
            frc_devi = np.sqrt(np.var(np.array(cmt_forces), axis=0))
            self.results["max_devi_f"] = np.max(frc_devi)
            self.results["min_devi_f"] = np.min(frc_devi)
            self.results["avg_devi_f"] = np.mean(frc_devi)
            if self.save_atomic:
                self.results["devi_f"] = np.reshape(frc_devi, (-1,3))
        
        # TODO: atomic energies?

        return


if __name__ == "__main__":
    ...

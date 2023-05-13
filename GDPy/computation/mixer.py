#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ase.calculators.calculator import all_changes
from ase.calculators.mixing import MixedCalculator

class AddonCalculator(MixedCalculator):

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """"""
        super().calculate(atoms, properties, system_changes)

        if "forces" in properties:
            forces1 = self.calcs[0].get_property("forces", atoms)
            forces2 = self.calcs[1].get_property("forces", atoms)
            self.results["force_contributions"] = np.hstack([forces1,forces2])

        return


if __name__ == "__main__":
    ...
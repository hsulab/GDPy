from ase.calculators.calculator import Calculator, all_changes
import numpy as np


class GPCalculator(Calculator):

    implemented_properties = ["energy", "forces", "stds"]

    def __init__(self, gp_model, **kwargs):
        super().__init__(**kwargs)
        self.gp_model = gp_model

    implemented_properties = ["energy", "forces", "stds", "energy_std"]

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        pred_forces, pred_std = self.gp_model.predict([atoms], return_std=True)
        pred_forces = pred_forces.reshape(-1, 3)
        pred_std = pred_std.reshape(-1, 3)

        try:
            pred_energy, pred_en_std = self.gp_model.predict_energy([atoms], return_std=True)
            pred_energy = float(pred_energy)
            pred_en_std = float(pred_en_std)
        except Exception:
            pred_energy = 0.0
            pred_en_std = 0.0

        self.results = {
            "energy": pred_energy,
            "forces": pred_forces,
            "stds": pred_std,
            "energy_std": pred_en_std,
        }

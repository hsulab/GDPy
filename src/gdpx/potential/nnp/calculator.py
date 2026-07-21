import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .descriptor import compute_symmetry_functions, compute_forces, compute_n_features
from .nn import SimpleNN


class ACSFNN(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(self, model_file, type_map=None, **kwargs):
        super().__init__(**kwargs)

        loaded = np.load(model_file)

        self.model_elements = [str(e) for e in loaded["elements"]]
        self.type_map = (
            list(type_map) if type_map is not None
            else list(self.model_elements)
        )
        for e in self.type_map:
            if e not in self.model_elements:
                raise ValueError(
                    f"Element '{e}' in type_map not found in model elements "
                    f"{self.model_elements}."
                )

        from .descriptor import G2Param, G4Param

        n_g2 = len(loaded["g2_eta"])
        self.g2_params = [
            G2Param(eta=float(loaded["g2_eta"][i]), Rs=float(loaded["g2_Rs"][i]))
            for i in range(n_g2)
        ]
        n_g4 = len(loaded["g4_eta"])
        self.g4_params = [
            G4Param(eta=float(loaded["g4_eta"][i]),
                    zeta=float(loaded["g4_zeta"][i]),
                    lambda_=float(loaded["g4_lambda_"][i]))
            for i in range(n_g4)
        ]
        self.r_cut = float(np.asarray(loaded["r_cut"]).flat[0])

        hidden_sizes = [int(x) for x in loaded["hidden_sizes"]]
        n_weights = sum(1 for k in loaded if k.startswith("W"))
        weights = [loaded[f"W{i}"] for i in range(n_weights)]
        biases = [loaded[f"b{i}"] for i in range(n_weights)]

        n_features = compute_n_features(
            self.model_elements, self.g2_params, self.g4_params
        )
        self.nn = SimpleNN(n_features, hidden_sizes=hidden_sizes)
        self.nn.set_params({"weights": weights, "biases": biases})

    def _compute_descriptor(self, atoms):
        return compute_symmetry_functions(
            atoms, self.model_elements, self.g2_params, self.g4_params, self.r_cut
        )

    def _validate_atoms(self, atoms):
        for s in atoms.get_chemical_symbols():
            if s not in self.type_map:
                raise ValueError(
                    f"Atom '{s}' not in type_map {self.type_map}. "
                    f"Model was trained on {self.model_elements}."
                )

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._validate_atoms(self.atoms)

        G = self._compute_descriptor(self.atoms)
        energy_per_atom, dE_dG = self.nn.energy_and_gradient(G)
        self.results["energy"] = float(np.sum(energy_per_atom))

        if "forces" in properties:
            self.results["forces"] = compute_forces(
                self.atoms, self.model_elements, self.g2_params, self.g4_params,
                self.r_cut, dE_dG,
            )

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .descriptor import (
    compute_n_features,
    compute_symmetry_functions,
    compute_symmetry_functions_and_derivatives,
)
from .nn import ElementwiseNN


MODEL_FORMAT_VERSION = 2


class ACSFNN(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(self, model_file, type_map=None, **kwargs):
        super().__init__(**kwargs)

        loaded = np.load(model_file)
        if "format_version" not in loaded:
            raise ValueError(
                "Legacy NNP model format is not supported. Retrain the model "
                "with the current NnpTrainer to create a version-2 model."
            )
        format_version = int(np.asarray(loaded["format_version"]).flat[0])
        if format_version != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported NNP model format version {format_version}; "
                f"expected {MODEL_FORMAT_VERSION}."
            )

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
        n_features = compute_n_features(
            self.model_elements, self.g2_params, self.g4_params
        )
        self.model = ElementwiseNN(
            n_features,
            self.model_elements,
            hidden_sizes=hidden_sizes,
            feature_mean=loaded["feature_mean"],
            feature_scale=loaded["feature_scale"],
            atomic_offsets=loaded["atomic_offsets"],
            rng=np.random.default_rng(0),
        )
        self.model.load_parameters(loaded)
        # ``nn`` was previously an internal single-network attribute. Keep a
        # readable alias while callers migrate to the element-wise model.
        self.nn = self.model

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

        if "forces" in properties:
            G, jacobian = compute_symmetry_functions_and_derivatives(
                self.atoms,
                self.model_elements,
                self.g2_params,
                self.g4_params,
                self.r_cut,
            )
            energy_per_atom, dE_dG = self.model.energy_and_gradient(
                G, self.atoms.get_chemical_symbols()
            )
            self.results["forces"] = jacobian.forces(dE_dG)
        else:
            G = self._compute_descriptor(self.atoms)
            energy_per_atom = self.model.forward(
                G, self.atoms.get_chemical_symbols()
            )
        self.results["energy"] = float(np.sum(energy_per_atom))

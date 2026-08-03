import copy
import json
import pathlib

import numpy as np
from ase.io import write

from .trainer import BasePotentialTrainer

WEIGHTS_NAME = "nn_weights.npz"
INPUT_DATASET_NAME = "input_dataset.xyz"


class NnpTrainer(BasePotentialTrainer):
    name = "nnp"

    def __init__(
        self,
        config,
        type_list=None,
        train_epochs=1000,
        directory=".",
        command="train",
        freeze_command="freeze",
        random_seed=None,
        calculator_params=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            type_list=type_list,
            train_epochs=train_epochs,
            directory=directory,
            command=command,
            freeze_command=freeze_command,
            random_seed=random_seed,
        )
        self.calculator_params = calculator_params or {}

    @property
    def frozen_name(self):
        return WEIGHTS_NAME

    def _resolve_train_command(self, *args, **kwargs):
        return ""

    def _resolve_freeze_command(self, *args, **kwargs):
        return ""

    def write_input(self, dataset, *args, **kwargs):
        if dataset:
            write(self.directory / INPUT_DATASET_NAME, dataset)

    def read_convergence(self) -> bool:
        return True

    def train(self, dataset, init_model=None, *args, **kwargs):
        train_dir = self.directory
        train_dir.mkdir(parents=True, exist_ok=True)

        self.write_input(dataset)

        calculator_params = self.calculator_params
        train_config = self.config
        n_epochs = train_config.get("n_epochs", self.train_epochs)
        learning_rate = train_config.get("learning_rate", 0.001)
        energy_weight = train_config.get("energy_weight", 1.0)
        force_weight = train_config.get("force_weight", 0.1)
        verbose = train_config.get("verbose", 100)
        max_grad_norm = train_config.get("max_grad_norm", None)

        if not calculator_params:
            raise ValueError(
                "NnpTrainer requires `calculator_params` in the trainer config with keys: "
                "elements, g2_params, g4_params, r_cut, hidden_sizes."
            )

        config_dump = dict(
            name=self.name,
            training=dict(
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                energy_weight=energy_weight,
                force_weight=force_weight,
            ),
            dataset=dict(
                n_structures=len(dataset),
                max_atoms=max(len(a) for a in dataset),
                min_atoms=min(len(a) for a in dataset),
            ),
            calculator=copy.deepcopy(calculator_params),
        )
        with open(train_dir / "train_config.json", "w") as f:
            json.dump(config_dump, f, indent=2)

        elements = calculator_params["elements"]
        g2_params_raw = calculator_params["g2_params"]
        g4_params_raw = calculator_params.get("g4_params", [])
        r_cut = calculator_params["r_cut"]
        hidden_sizes = calculator_params.get("hidden_sizes", (64, 64))

        from gdpx.potential.nnp.descriptor import G2Param, G4Param, compute_n_features

        g2_norm = [G2Param(*p) if not isinstance(p, G2Param) else p for p in g2_params_raw]
        g4_norm = [G4Param(*p) if not isinstance(p, G4Param) else p for p in g4_params_raw]
        n_features = compute_n_features(elements, g2_norm, g4_norm)

        from gdpx.potential.nnp.nn import SimpleNN

        nn = SimpleNN(n_features, hidden_sizes=hidden_sizes)

        ref_energies = np.zeros(len(dataset))
        for idx, atoms in enumerate(dataset):
            if atoms.calc is None or "energy" not in atoms.calc.results:
                raise ValueError(
                    f"Structure {idx} has no reference energy "
                    "(atoms.get_potential_energy() unavailable). "
                    "Provide reference energies in the dataset."
                )
            ref_energies[idx] = atoms.get_potential_energy()
            if force_weight > 0 and (
                atoms.calc is None or "forces" not in atoms.calc.results
            ):
                raise ValueError(
                    f"Structure {idx} has no reference forces "
                    "(atoms.get_forces(apply_constraint=False) unavailable); "
                    f"required since force_weight={force_weight}."
                )

        energy_shift = float(np.mean(ref_energies))

        model_path = train_dir / WEIGHTS_NAME
        save_dict = {}
        params = nn.get_params()
        for i, w in enumerate(params["weights"]):
            save_dict[f"W{i}"] = w
        for i, b in enumerate(params["biases"]):
            save_dict[f"b{i}"] = b
        save_dict["hidden_sizes"] = np.array(hidden_sizes)
        save_dict["elements"] = np.array(elements)
        save_dict["g2_eta"] = np.array([p.eta for p in g2_norm])
        save_dict["g2_Rs"] = np.array([p.Rs for p in g2_norm])
        save_dict["g4_eta"] = np.array([p.eta for p in g4_norm])
        save_dict["g4_zeta"] = np.array([p.zeta for p in g4_norm])
        save_dict["g4_lambda_"] = np.array([p.lambda_ for p in g4_norm])
        save_dict["r_cut"] = np.float64(r_cut)
        save_dict["energy_shift"] = np.float64(energy_shift)
        np.savez_compressed(model_path, **save_dict)

        from gdpx.potential.nnp.calculator import ACSFNN
        from gdpx.potential.nnp.descriptor import (
            compute_forces,
            compute_force_gradient_weights,
        )

        calc = ACSFNN(model_file=model_path)

        adam_state = None

        for epoch in range(n_epochs):
            total_loss = 0.0
            total_grads = None

            for atoms in dataset:
                G = calc._compute_descriptor(atoms)
                E_pred = float(np.sum(calc.nn.forward(G)))
                E_ref = atoms.get_potential_energy() - energy_shift
                dE = E_pred - E_ref
                num_atoms = max(len(atoms), 1)
                loss = energy_weight * dE**2 / num_atoms

                grad_output = np.full(len(atoms), 2.0 * energy_weight * dE / num_atoms)
                grads = calc.nn.backward(grad_output)

                if total_grads is None:
                    total_grads = _copy_grads(grads)
                else:
                    _add_grads(total_grads, grads)

                if force_weight > 0:
                    num_atoms = len(atoms)
                    forces_ref = atoms.get_forces(apply_constraint=False)

                    _, dE_dG = calc.nn.energy_and_gradient(G)
                    forces_pred = compute_forces(
                        atoms,
                        elements,
                        g2_params_raw,
                        g4_params_raw,
                        r_cut,
                        dE_dG,
                    )

                    dF = forces_pred - forces_ref
                    loss += force_weight * np.mean(dF**2)

                    B = compute_force_gradient_weights(
                        atoms,
                        elements,
                        g2_params_raw,
                        g4_params_raw,
                        r_cut,
                        dF,
                    )
                    force_grads = calc.nn.double_backward(B)
                    _scale_grads(force_grads, -2.0 * force_weight / (3.0 * num_atoms))
                    _add_grads(total_grads, force_grads)

                total_loss += loss

            n = len(dataset)
            _scale_grads(total_grads, 1.0 / n)

            if max_grad_norm is not None:
                grad_norm = _global_norm(total_grads)
                if grad_norm > max_grad_norm:
                    _scale_grads(total_grads, max_grad_norm / grad_norm)

            calc.nn.adam_update(total_grads, learning_rate, adam_state)

            if verbose and epoch % verbose == 0:
                avg_loss = total_loss / n
                self._print(f"Epoch {epoch:5d}: loss = {avg_loss:.8f}  (E_weight={energy_weight}, F_weight={force_weight})")

        params = calc.nn.get_params()
        save_dict = {}
        for i, w in enumerate(params["weights"]):
            save_dict[f"W{i}"] = w
        for i, b in enumerate(params["biases"]):
            save_dict[f"b{i}"] = b

        hidden_sizes = [w.shape[1] for w in params["weights"][:-1]]
        save_dict["hidden_sizes"] = np.array(hidden_sizes)
        save_dict["elements"] = np.array(calc.model_elements)
        save_dict["g2_eta"] = np.array([p.eta for p in calc.g2_params])
        save_dict["g2_Rs"] = np.array([p.Rs for p in calc.g2_params])
        save_dict["g4_eta"] = np.array([p.eta for p in calc.g4_params])
        save_dict["g4_zeta"] = np.array([p.zeta for p in calc.g4_params])
        save_dict["g4_lambda_"] = np.array([p.lambda_ for p in calc.g4_params])
        save_dict["r_cut"] = np.float64(calc.r_cut)
        save_dict["energy_shift"] = np.float64(energy_shift)

        np.savez_compressed(train_dir / WEIGHTS_NAME, **save_dict)

    def freeze(self):
        model_path = (self.directory / self.frozen_name).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Trained weights not found at {model_path}. Run train() first.")
        return model_path


def _global_norm(grads):
    total = 0.0
    for arr_list in grads.values():
        for arr in arr_list:
            total += np.sum(arr**2)
    return np.sqrt(total)


def _copy_grads(grads):
    return {
        "weights": [w.copy() for w in grads["weights"]],
        "biases": [b.copy() for b in grads["biases"]],
    }


def _add_grads(target, source):
    for i in range(len(target["weights"])):
        target["weights"][i] += source["weights"][i]
    for i in range(len(target["biases"])):
        target["biases"][i] += source["biases"][i]


def _scale_grads(grads, factor):
    for i in range(len(grads["weights"])):
        grads["weights"][i] *= factor
    for i in range(len(grads["biases"])):
        grads["biases"][i] *= factor

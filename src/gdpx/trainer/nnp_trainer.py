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
        learning_rate = train_config.get("learning_rate", 0.0001)
        energy_weight = train_config.get("energy_weight", 1.0)
        force_weight = train_config.get("force_weight", 0.1)
        verbose = train_config.get("verbose", 100)
        max_grad_norm = train_config.get("max_grad_norm", 1.0)

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
        np.savez_compressed(model_path, **save_dict)

        from gdpx.potential.nnp.calculator import ACSFNN
        from gdpx.potential.nnp.descriptor import compute_forces

        calc = ACSFNN(model_file=model_path)

        for epoch in range(n_epochs):
            total_loss = 0.0
            total_grads = None

            for atoms in dataset:
                G = calc._compute_descriptor(atoms)
                E_pred = float(np.sum(calc.nn.forward(G)))
                E_ref = atoms.info.get("energy", 0.0)
                dE = E_pred - E_ref
                num_atoms = max(len(atoms), 1)
                loss = energy_weight * dE**2 / num_atoms

                grad_output = np.full(len(atoms), 2.0 * energy_weight * dE / num_atoms)
                grads = calc.nn.backward(grad_output)

                if force_weight > 0 and "forces" in atoms.arrays:
                    num_atoms = len(atoms)
                    forces_ref = atoms.arrays["forces"]
                    pos0 = atoms.positions.copy()

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

                    if total_grads is None:
                        total_grads = _copy_grads(grads)
                    else:
                        _add_grads(total_grads, grads)

                    h_fd = 1e-5
                    for ci in range(num_atoms):
                        for d in range(3):
                            atoms.positions[ci, d] = pos0[ci, d] + h_fd
                            Gp = calc._compute_descriptor(atoms)
                            calc.nn.forward(Gp)
                            dE_dW_plus = calc.nn.backward()

                            atoms.positions[ci, d] = pos0[ci, d] - h_fd
                            Gm = calc._compute_descriptor(atoms)
                            calc.nn.forward(Gm)
                            dE_dW_minus = calc.nn.backward()

                            atoms.positions[ci, d] = pos0[ci, d]

                            gf = 2.0 * force_weight * dF[ci, d] / (3.0 * num_atoms)
                            dF_dW = _sub_grads(dE_dW_plus, dE_dW_minus)
                            _scale_grads(dF_dW, gf / (2.0 * h_fd))
                            _add_grads(total_grads, dF_dW)
                else:
                    if total_grads is None:
                        total_grads = _copy_grads(grads)
                    else:
                        _add_grads(total_grads, grads)

                total_loss += loss

            n = len(dataset)
            _scale_grads(total_grads, 1.0 / n)

            grad_norm = _global_norm(total_grads)
            if grad_norm > max_grad_norm:
                _scale_grads(total_grads, max_grad_norm / grad_norm)

            calc.nn.update(total_grads, learning_rate)

            if verbose and epoch % verbose == 0:
                avg_loss = total_loss / n
                print(f"Epoch {epoch:5d}: loss = {avg_loss:.8f}  (E_weight={energy_weight}, F_weight={force_weight})")

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


def _sub_grads(a, b):
    return {
        "weights": [aw - bw for aw, bw in zip(a["weights"], b["weights"])],
        "biases": [ab - bb for ab, bb in zip(a["biases"], b["biases"])],
    }


def _scale_grads(grads, factor):
    for i in range(len(grads["weights"])):
        grads["weights"][i] *= factor
    for i in range(len(grads["biases"])):
        grads["biases"][i] *= factor

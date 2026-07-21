import copy
import json
import pathlib

import numpy as np


TRAIN_CONFIG_NAME = "train_config.json"
WEIGHTS_NAME = "nn_weights.npz"


class NnpTrainer:

    name = "nnp"

    def __init__(self, config=None, type_list=None, train_epochs=1000,
                 directory=".", command="train", freeze_command="freeze",
                 random_seed=None, **kwargs):
        if config is None:
            config = {}
        self.config = config
        self.directory = pathlib.Path(directory)
        self.train_epochs = train_epochs
        self.command = command
        self.freeze_command = freeze_command
        self.random_seed = random_seed

    @property
    def frozen_name(self):
        return WEIGHTS_NAME

    def train(self, dataset, calc=None, calculator_params=None, *args, **kwargs):
        train_dir = self.directory
        train_dir.mkdir(parents=True, exist_ok=True)

        train_config = self.config
        n_epochs = train_config.get("n_epochs", self.train_epochs)
        learning_rate = train_config.get("learning_rate", 0.001)
        energy_weight = train_config.get("energy_weight", 1.0)
        force_weight = train_config.get("force_weight", 0.0)
        verbose = train_config.get("verbose", 100)

        if calc is None:
            raise ValueError("NnpTrainer requires a `calc` (ACSFNN instance) passed to train().")

        config_dump = dict(
            name=self.name,
            training=dict(
                n_epochs=n_epochs, learning_rate=learning_rate,
                energy_weight=energy_weight, force_weight=force_weight,
            ),
            dataset=dict(
                n_structures=len(dataset),
                max_atoms=max(len(a) for a in dataset),
                min_atoms=min(len(a) for a in dataset),
            ),
        )
        if calculator_params is not None:
            config_dump["calculator"] = copy.deepcopy(calculator_params)

        with open(train_dir / TRAIN_CONFIG_NAME, "w") as f:
            json.dump(config_dump, f, indent=2)

        for epoch in range(n_epochs):
            total_loss = 0.0
            total_grads = None

            for atoms in dataset:
                G = calc._compute_descriptor(atoms)
                E_pred = float(np.sum(calc.nn.forward(G)))
                E_ref = atoms.info.get("energy", 0.0)
                dE = E_pred - E_ref
                loss = energy_weight * dE ** 2

                grad_output = np.full(len(atoms), 2.0 * energy_weight * dE)
                grads = calc.nn.backward(grad_output)

                if force_weight > 0 and "forces" in atoms.arrays:
                    n_atoms = len(atoms)
                    _, dE_dG = calc.nn.energy_and_gradient(G)
                    forces_ref = atoms.arrays["forces"]
                    pos0 = atoms.positions.copy()
                    h = calc.fd_h

                    forces_pred = np.zeros((n_atoms, 3))
                    for ci in range(n_atoms):
                        for d in range(3):
                            atoms.positions[ci, d] = pos0[ci, d] + h
                            Gp = calc._compute_descriptor(atoms)
                            atoms.positions[ci, d] = pos0[ci, d] - h
                            Gm = calc._compute_descriptor(atoms)
                            atoms.positions[ci, d] = pos0[ci, d]
                            dG = (Gp - Gm) / (2.0 * h)
                            forces_pred[ci, d] = -float(np.sum(dE_dG * dG))

                    dF = forces_pred - forces_ref
                    loss += force_weight * np.mean(dF ** 2)

                    if total_grads is None:
                        total_grads = _copy_grads(grads)
                    else:
                        _add_grads(total_grads, grads)

                    for ci in range(n_atoms):
                        for d in range(3):
                            atoms.positions[ci, d] = pos0[ci, d] + h
                            Gp = calc._compute_descriptor(atoms)
                            calc.nn.forward(Gp)
                            dE_dW_plus = calc.nn.backward()

                            atoms.positions[ci, d] = pos0[ci, d] - h
                            Gm = calc._compute_descriptor(atoms)
                            calc.nn.forward(Gm)
                            dE_dW_minus = calc.nn.backward()

                            atoms.positions[ci, d] = pos0[ci, d]

                            gf = (2.0 * force_weight * dF[ci, d]
                                  / (3.0 * n_atoms))
                            dF_dW = _sub_grads(dE_dW_plus, dE_dW_minus)
                            _scale_grads(dF_dW, gf / (2.0 * h))
                            _add_grads(total_grads, dF_dW)
                else:
                    if total_grads is None:
                        total_grads = _copy_grads(grads)
                    else:
                        _add_grads(total_grads, grads)

                total_loss += loss

            n = len(dataset)
            _scale_grads(total_grads, 1.0 / n)
            calc.nn.update(total_grads, learning_rate)

            if verbose and epoch % verbose == 0:
                avg_loss = total_loss / n
                print(f"Epoch {epoch:5d}: loss = {avg_loss:.8f}  "
                      f"(E_weight={energy_weight}, F_weight={force_weight})")

        params = calc.nn.get_params()
        save_dict = {}
        for i, w in enumerate(params["weights"]):
            save_dict[f"W{i}"] = w
        for i, b in enumerate(params["biases"]):
            save_dict[f"b{i}"] = b
        np.savez(train_dir / WEIGHTS_NAME, **save_dict)

    def freeze(self, calc, train_dir=None):
        if train_dir is None:
            train_dir = self.directory
        train_dir = pathlib.Path(train_dir)
        weights_file = train_dir / WEIGHTS_NAME
        if not weights_file.exists():
            raise FileNotFoundError(
                f"Trained weights not found at {weights_file}. "
                f"Run train() first."
            )
        loaded = np.load(weights_file)
        n_weights = sum(1 for k in loaded if k.startswith("W"))
        weights = [loaded[f"W{i}"] for i in range(n_weights)]
        biases = [loaded[f"b{i}"] for i in range(n_weights)]
        calc.nn.set_params({"weights": weights, "biases": biases})


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

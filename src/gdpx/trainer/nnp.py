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
                        total_grads = {k: grads[k].copy() for k in grads}
                    else:
                        for k in total_grads:
                            total_grads[k] += grads[k]

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
                            for k in total_grads:
                                total_grads[k] += (gf
                                                   * (-(dE_dW_plus[k]
                                                        - dE_dW_minus[k])
                                                      / (2.0 * h)))
                else:
                    if total_grads is None:
                        total_grads = {k: grads[k].copy() for k in grads}
                    else:
                        for k in total_grads:
                            total_grads[k] += grads[k]

                total_loss += loss

            n = len(dataset)
            for k in total_grads:
                total_grads[k] /= n
            calc.nn.update(total_grads, learning_rate)

            if verbose and epoch % verbose == 0:
                avg_loss = total_loss / n
                print(f"Epoch {epoch:5d}: loss = {avg_loss:.8f}  "
                      f"(E_weight={energy_weight}, F_weight={force_weight})")

        np.savez(train_dir / WEIGHTS_NAME, **calc.nn.get_params())

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
        calc.nn.set_params({
            k: loaded[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"]
        })

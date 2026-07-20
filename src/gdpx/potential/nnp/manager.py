import copy
import numpy as np

from ..manager import BasePotentialManager


class NnAcsfManager(BasePotentialManager):

    name = "nnp"
    implemented_backends = ("ase",)
    valid_combinations = (("ase", "ase"),)

    def __init__(self):
        super().__init__()

    def register_calculator(self, calc_params, *args, **kwargs):
        super().register_calculator(calc_params, *args, **kwargs)
        from .calculator import ACSFNN
        self.calc = ACSFNN(**calc_params)

    def train(self, dataset, n_epochs=1000, lr=0.001,
              energy_weight=1.0, force_weight=0.0, verbose=100):
        calc = self.calc

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
            calc.nn.update(total_grads, lr)

            if verbose and epoch % verbose == 0:
                avg_loss = total_loss / n
                print(f"Epoch {epoch:5d}: loss = {avg_loss:.8f}  "
                      f"(E_weight={energy_weight}, F_weight={force_weight})")

    def as_dict(self):
        params = copy.deepcopy(self.calc_params)
        params.pop("nn_weights", None)
        params["nn_weights"] = self.calc.nn.get_params()
        return {"name": self.name, "params": params}

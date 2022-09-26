#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
from pathlib import Path

import yaml

import numpy as np

from GDPy.potential.manager import AbstractPotentialManager

class NequipManager(AbstractPotentialManager):

    name = "nequip"
    implemented_backends = ["ase", "lammps"]

    valid_combinations = [
        ["ase", "ase"], # calculator, dynamics
        ["lammps", "ase"],
        ["lammps", "lammps"]
    ]
    
    def __init__(self):
        """"""
        self.committee = None

        return

    def register_calculator(self, calc_params):
        """"""
        super().register_calculator(calc_params)

        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())
        atypes = calc_params.pop("type_list", [])

        models = calc_params.get("file", None)

        type_map = {}
        for i, a in enumerate(atypes):
            type_map[a] = i

        if self.calc_backend == "ase":
            # return ase calculator
            from nequip.ase import NequIPCalculator
            calc = NequIPCalculator.from_deployed_model(
                model_path=models
            )
        elif self.calc_backend == "lammps":
            from GDPy.computation.lammps import Lammps
            pair_style = calc_params.get("pair_style", None)
            if pair_style:
                calc = Lammps(
                    command=command, directory=directory, **calc_params
                )
                # - update several params
                calc.set(newton="off")
            else:
                calc = None
        
        self.calc = calc

        return
    
    def register_trainer(self, train_params_: dict):
        """"""
        super().register_trainer(train_params_)
        # print(self.train_config)

        return
    
    def train(self, dataset=None, train_dir=Path.cwd()):
        """"""
        self._make_train_files(dataset, train_dir)

        return

    def _make_train_files(self, dataset=None, train_dir=Path.cwd()):
        """ make files for training
        """
        # - add dataset to config
        if not dataset: # NOTE: can be a path or a List[Atoms]
            dataset = self.train_dataset
        assert dataset, f"No dataset has been set for the potential {self.name}."

        # - check dataset
        from ase.io import read, write
        write(train_dir/"dataset.xyz", dataset)

        nframes = len(dataset)
        n_train = int(nframes*0.9)
        n_val = nframes - n_train

        # - check train config
        # params: root, run_name, seed, dataset_seed, n_train, n_val, batch_size
        #         dataset, dataset_file_name
        train_config = copy.deepcopy(self.train_config)

        train_config["root"] = str(train_dir.resolve())
        train_config["run_name"] = "auto"

        train_config["seed"] = np.random.randint(0,10000)
        train_config["dataset_seed"] = np.random.randint(0,10000)

        train_config["dataset"] = "ase"
        train_config["dataset_file_name"] = str((train_dir/"dataset.xyz").resolve())

        train_config["n_train"] = n_train
        train_config["n_val"] = n_val

        with open(train_dir/"config.yaml", "w") as fopen:
            yaml.safe_dump(train_config, fopen)

        return
    
    def freeze(self, train_dir=Path.cwd()):
        """ freeze model and update current attached calc?
        """
        # - find subdirs
        train_dir = Path(train_dir)
        mdirs = []
        for p in train_dir.iterdir():
            if p.is_dir() and p.name.startswith("m"):
                mdirs.append(p.resolve())
        assert len(mdirs) == self.train_size, "Number of models does not equal model size..."

        # - find models and form committee
        models = []
        for p in mdirs:
            models.append(str(p/"deployed_model.pth"))

        committee = []
        for i, m in enumerate(models):
            calc_params = copy.deepcopy(self.calc_params)
            calc_params.update(backend=self.calc_backend)
            if self.calc_backend == "ase":
                calc_params["file"] = m
            elif self.calc_backend == "lammps":
                calc_params["pair_style"] = "nequip"
                calc_params["pair_coeff"] = "* * {}".format(m)
            saved_calc_params = copy.deepcopy(calc_params)
            # NOTE: self.calc will be the last one
            self.register_calculator(calc_params)
            self.calc.directory = Path.cwd()/"c{i}".format(i)
            committee.append(self.calc)
        # NOTE: do not share calculator...
        self.register_calculator(saved_calc_params) # use last model
        if len(committee) > 1:
            self.committee = committee

        return
    
    def estimate_uncertainty(self, frames):
        """ use committee to estimate uncertainty
        """
        from ase.calculators.calculator import PropertyNotImplementedError
        if self.committee:
            # max_devi_e, max_devi_f
            # TODO: directory where estimate?
            for atoms in frames:
                cmt_tot_energy = []
                cmt_energies = []
                cmt_forces = []
                for c in self.committee:
                    c.reset()
                    atoms.calc = c
                    # - total energy
                    energy = atoms.get_potential_energy()
                    cmt_tot_energy.append(energy)
                    # - atomic energies
                    try:
                        energies = atoms.get_potential_energies()
                    except PropertyNotImplementedError:
                        energies = [1e8]*len(atoms)
                    cmt_energies.append(energies)
                    # - atomic forces
                    forces = atoms.get_forces()
                    cmt_forces.append(forces)
                cmt_tot_energy = np.array(cmt_tot_energy)
                tot_energy_devi = np.sqrt(np.var(cmt_tot_energy))
                atoms.info["te_devi"] = tot_energy_devi

                cmt_energies = np.array(cmt_energies)
                ae_devi = np.sqrt(np.var(cmt_energies, axis=0))
                atoms.arrays["ae_devi"] = ae_devi
                atoms.info["max_devi_e"] = np.max(ae_devi)

                cmt_forces = np.array(cmt_forces)
                force_devi = np.sqrt(np.var(cmt_forces, axis=0))
                atoms.arrays["force_devi"] = force_devi
                atoms.info["max_devi_f"] = np.max(force_devi)
        else:
            pass

        return frames


if __name__ == "__main__":
    pass
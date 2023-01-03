#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import pathlib
from pathlib import Path
from typing import NoReturn, List, Union

import numpy as np

import matplotlib
matplotlib.use('Agg') #silent mode
import matplotlib.pyplot as plt
#plt.style.use("presentation")

from ase import Atoms
from ase.io import read, write

from ase.calculators import calculator

import ase.optimize
from ase.constraints import FixAtoms
from ase.constraints import UnitCellFilter

from abc import ABC
from abc import abstractmethod

from GDPy.potential.register import PotentialRegister
from GDPy.utils.command import parse_input_file

from GDPy.validator.validator import AbstractValidator

"""
Various properties to be validated

Atomic Energy and Crystal Lattice constant

Elastic Constants

Phonon Calculations

Point Defects (vacancies, self interstitials, ...)

Surface energies

Diffusion Coefficient

Adsorption, Reaction, ...
"""


class MinimaValidator(AbstractValidator):

    def __init__(self, task_outpath: str, task_params: dict, pot_worker=None):
        """ run minimisation on various configurations and
            compare relative energy
            how to postprocess
        """
        super().__init__(task_outpath, task_params, pot_worker)

        #self.calc = self.__parse_calculator(self.valid_dict)
        self.driver = self.pm.create_worker(
            dyn_params = task_params["dynamics"]
        )

        return
    
    def __run_dynamics(
        self, atoms, dyn_cls, dyn_opts: dict
    ):
        """"""
        init_positions = atoms.get_positions().copy()

        self.calc.reset()
        atoms.calc = self.calc
        # check bulk optimisation
        check_bulk = atoms.info.get("constraint", None)
        if check_bulk is not None:
            print("use {}".format(check_bulk))
            atoms = UnitCellFilter(atoms, constant_volume=False)
        dyn = dyn_cls(atoms)
        dyn.run(**dyn_opts)

        opt_positions = atoms.get_positions().copy()
        rmse = np.sqrt(np.var(opt_positions - init_positions))

        return atoms, rmse
    
    def __parse_dynamics(self, dyn_dict: dict):
        """"""
        cur_dict = dyn_dict.copy()
        dyn_name = cur_dict.pop('name')

        return getattr(ase.optimize, dyn_name), cur_dict
    
    def _run_group(self, group_data: dict, dyn_dict: dict):
        """ run group of structures
        """
        group_output = [] # [[name, atoms],...,[]]
        if False:
            for stru_file, cons_data in zip(group_data['structures'], group_data['constraints']):
                atoms_name = pathlib.Path(stru_file).stem
                print('===== ', atoms_name, ' =====')
                frames = read(stru_file, ':')
                assert len(frames) == 1, 'only one structure at a time now'
                atoms = frames[0]
                atoms.calc = self.calc
                if cons_data[0] == "FixAtoms":
                    if cons_data[1] is not None:
                        cons = FixAtoms(
                            indices = [atom.index for atom in atoms if atom.z < cons_data[1]]
                        )
                        atoms.set_constraint(cons)
                        print('constraint: natoms', cons)
                    else:
                        pass
                elif cons_data[0] == "UnitCellFilter":
                    atoms = UnitCellFilter(atoms, constant_volume=False)
                    print('constraint: UnitcellFilter')
                else:
                    raise ValueError("unsupported constraint type.")
                dynamics = getattr(ase.optimize, dyn_data[0])
                dyn = dynamics(atoms)
                dyn.run(dyn_data[1], dyn_data[2])
                #if self.pot.uncertainty:
                #    print(atoms.calc.results['energy_stdvar'])
                group_output.append([atoms_name, atoms])
        else:
            # read structures
            frames = []
            if isinstance(group_data['structures'], list):
                for stru_file in group_data['structures']:
                    stru_name = pathlib.Path(stru_file).stem
                    cur_frames = read(stru_file, ':')
                    assert len(cur_frames) == 1, 'only one structure in this mode' 
                    atoms = cur_frames[0]
                    atoms.info['description'] = stru_name
                    frames.append(atoms)
            else:
                #print(group_data['structures'])
                cur_frames = read(group_data['structures'], ':')
                frames.extend(cur_frames)
            
            # parse dynamics inputs
            # optimise/dynamics class, run_params
            dyn_cls, dyn_opts = self.__parse_dynamics(dyn_dict)
            
            # start dynamics
            for i, atoms in enumerate(frames):
                atoms_name = atoms.info.get('description', 'structure %d' %i)
                print(
                    'calculating {} ...'.format(atoms_name)
                )
                opt_atoms, rmse = self.__run_dynamics(
                    atoms, dyn_cls, dyn_opts
                )
                print("Structure Deviation: ", rmse)
                if self.pm.uncertainty:
                    self.calc.reset()
                    self.calc.calc_uncertainty = True
                    opt_atoms.calc = self.calc
                    energy = opt_atoms.get_potential_energy()
                    stdvar = opt_atoms.calc.results["en_stdvar"]
                    print("Final energy: {:.4f} Deviation: {:.4f}".format(energy, stdvar))
                
                group_output.append([atoms_name, atoms])

        return group_output
    
    def run(self):
        self.my_references = []
        self.outputs = []
        #for (task_name, task_data) in self.tasks.items():
        #    print('start task ', task_name)
        #    basics_output = self._run_group(task_data['basics'], task_data['dynamics'])
        #    composites_output = self._run_group(task_data['composites'], task_data['dynamics'])
        #    self.outputs.append({'basics': basics_output, 'composites': composites_output})

        print("=== initialisation ===")
        stru_paths = self.task_params["structures"]
        frames = []
        for p in stru_paths:
            cur_frames = read(p, ":")
            print(f"nframes {len(cur_frames)} in {p}")
            frames.extend(cur_frames)
        print("total nframes: ", len(frames))

        #for a in frames:
        #    print(a.info)
        ref_energies = [a.get_potential_energy() for a in frames]
        #ref_energies = [a.info["energy"] for a in frames]
        names = []
        for a in frames:
            name = a.info.get("name", None)
            if name is None:
                name = a.info.get("description", None)
            if name is None:
                name = a.get_chemical_formula()
            names.append(name)

        print("=== minimisation ===")
        calc_frames = []
        for name, atoms in zip(names, frames):
            self.driver.directory = self.task_outpath / name
            # NOTE: ase dynamics wont create new atoms
            new_atoms = self.driver.run(atoms.copy()) 
            calc_frames.append(new_atoms)
        new_energies = [a.get_potential_energy() for a in calc_frames]

        #print(names)
        #print(ref_energies)
        #print(new_energies)

        # analysis
        print("=== analysis ===")
        for i, name in enumerate(names):
            # energetic data
            a, b = ref_energies[i], new_energies[i]
            print("{:24s}\n    energy [eV]  {:>12.4f}  {:>12.4f}  {:>12.4f}".format(name, a, b, b-a))
            # geometric data
            #geo_devi = np.fabs(frames[i].positions - calc_frames[i].positions)
            #print(geo_devi)
            geo_devi = np.mean(np.fabs(frames[i].positions - calc_frames[i].positions))
            fmax = np.max(np.fabs(calc_frames[i].get_forces(apply_constraint=True)))
            print("    fmax  {:<8.4f} eV/AA  GMAE {:<8.4f} AA  ".format(fmax, geo_devi))

        return
    
    def analyse(self):
        # check data
        saved_frames = []
        for (task_name, task_data), output_data in zip(self.tasks.items(), self.outputs):
            print("\n\n===== Task {0} Summary =====".format(task_name))
            basics_output = output_data['basics']
            basics_energies = []
            for (atoms_name, atoms), coef in zip(basics_output, task_data['basics']['coefs']):
                basics_energies.append(atoms.get_potential_energy()*coef)
            composites_output = output_data['composites']
            composites_references = task_data['composites'].get('references', None)
            for idx, ((atoms_name, atoms), coef) in enumerate(zip(composites_output, task_data['composites']['coefs'])):
                assert len(basics_energies) == len(coef)
                relative_energy = atoms.get_potential_energy()
                for en, c in zip(basics_energies, coef):
                    relative_energy -= c*en
                saved_frames.append(atoms)
                if composites_references is not None:
                    #if self.pot.uncertainty > 1:
                    #    # print(atoms_name, relative_energy, atoms.info['energy_stdvar'], composites_references[idx])
                    #    print(atoms_name, relative_energy, composites_references[idx])
                    #else:
                    #    print(atoms_name, relative_energy, composites_references[idx])
                    print(
                        "{0:<20s}  {1:.4f}  {2:.4f}  {3:.4f}".format(
                            atoms_name, atoms.get_potential_energy(),
                            relative_energy, composites_references[idx]
                        )
                    )
                else:
                    print(atoms_name, relative_energy)
        write(self.output_path / 'saved.xyz', saved_frames)

        return

class RunCalculation():

    def __init__(self):

        return 
    
    def run(self, frames, func_name):
        """"""
        func = getattr(self, func_name)
        return func(frames)
    
    @staticmethod
    def dimer(frames):
        """turn xyz into dimer data"""
        data = []
        for atoms in frames:
            # donot consider minimum image
            distance = np.linalg.norm(atoms[0].position-atoms[1].position) 
            energy = atoms.get_potential_energy()
            data.append([distance,energy])
        data = np.array(data)
    
        return np.array(data[:,0]), np.array(data[:,1])

    @staticmethod
    def volume(frames):
        """turn xyz into eos data"""
        data = []
        for atoms in frames:
            # donot consider minimum image
            vol = atoms.get_volume()
            energy = atoms.get_potential_energy()
            data.append([vol,energy])
        data = np.array(data)

        return np.array(data[:,0]), np.array(data[:,1])


class SinglePointValidator(AbstractValidator):

    """
    calculate energies on each structures and save them to file
    """

    def __init__(self, task_outpath: str, task_params: dict, pot_manager=None):
        """
        """
        self.task_outpath = Path(task_outpath)
        self.task_params = task_params
        self.pm = pot_manager

        self.calc = pot_manager.calc
        
        self.structure_paths = self.task_params.get("structures", None)

        return

    def run(self):
        """
        lattice constant
        equation of state
        """
        for stru_path in self.structure_paths:
            # set output file name
            stru_path = Path(stru_path)
            stru_name = stru_path.stem
            fname = self.task_outpath / (stru_name + "-valid.dat")
            pname = self.task_outpath / (stru_name + "-valid.png")

            print(stru_path)

            # run dp calculation
            frames = read(stru_path, ":")
            natoms_array = [len(a) for a in frames]
            volumes = [a.get_volume() for a in frames]
            dft_energies = [a.get_potential_energy() for a in frames]

            mlp_energies = []
            self.calc.reset()
            for a in frames:
                a.calc = self.calc
                mlp_energies.append(a.get_potential_energy())

            # save to data file
            data = np.array([natoms_array, volumes, dft_energies, mlp_energies]).T
            np.savetxt(fname, data, fmt="%12.4f", header="natoms Prop DFT MLP")

            self.plot_dimer(
                "Bulk EOS", volumes, 
                {
                    "DFT": dft_energies, 
                    "MLP": mlp_energies
                },
                pname
            )

        return

    @staticmethod
    def plot_dimer(task_name, distances, energies: dict, pname):
        """"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
        ax.set_title(
            task_name,
            fontsize=20, 
            fontweight='bold'
        )
    
        ax.set_xlabel('Distance [Å]', fontsize=16)
        ax.set_ylabel('Energyr [eV]', fontsize=16)

        for name, en in energies.items():
            ax.scatter(distances, en, label=name)
        ax.legend()

        plt.savefig(pname)

        return

    def analyse(self):        
        # plot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,12))
        plt.suptitle(
            #"Birch-Murnaghan (Constant-Volume Optimisation)"
            "Energy-Volume Curve"
        )
    
        ax.set_xlabel("Volume [Å^3/atom]")
        ax.set_ylabel("Energy [eV/atom]")

        ax.scatter(volumes/natoms_array, dft_energies/natoms_array, marker="*", label="DFT")
        ax.scatter(volumes/natoms_array, mlp_energies/natoms_array, marker="x", label="MLP")

        ax.legend()

        plt.savefig('bm.png')
        """

        return

def run_validation(
    directory: Union[str, pathlib.Path], input_json: Union[str, pathlib.Path],
    pot_manager
):
    """ This is a factory to deal with various validations...
    """
    # parse potential
    pm = pot_manager

    # run over validations
    valid_dict = parse_input_file(input_json)

    wdir = pathlib.Path(directory)
    if wdir.exists():
        warnings.warn("Validation wdir exists.", UserWarning)

    tasks = valid_dict.get("tasks", {})
    if len(tasks) == 0:
        raise RuntimeError(f"No tasks was found in {input_json}")
    
    for task_name, task_params in tasks.items():
        print(f"=== Run Validation Task {task_name} ===")
        task_outpath = wdir/task_name
        if not task_outpath.exists():
            task_outpath.mkdir(parents=True)
        method = task_params.get("method", "minima")
        # test surface related energies
        if method == "dimer":
            from GDPy.validator.dimer import DimerValidator
            rv = DimerValidator(task_outpath, task_params, pm)
        elif method == "rdf":
            from GDPy.validator.rdf import RdfValidator
            rv = RdfValidator(task_outpath, task_params, pm)
        elif method == "minima":
            rv = MinimaValidator(task_outpath, task_params, pm)
        elif method == "rxn":
            from GDPy.validator.rxn import ReactionValidator
            rv = ReactionValidator(task_outpath, task_params, pm)
        elif method == "bulk":
            rv = SinglePointValidator(task_outpath, task_params, pm)
        rv.run()
        #rv.analyse()

    return


if __name__ == "__main__":
    pass

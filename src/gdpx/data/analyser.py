#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import groupby
import time

import numpy as np
import scipy as sp

from gdpx.selector.abstract import create_selector
from gdpx.utils.comparasion import parity_plot_dict, rms_dict
from gdpx.data.operators import append_predictions, merge_predicted_forces, xyz2results
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read, write
from ase import Atoms
from tqdm import tqdm
import pathlib
from pathlib import Path

from typing import NoReturn, Union, List, overload
from joblib import Parallel, delayed

from collections import Counter

from gdpx.utils.command import parse_input_file

# global settings
from gdpx import config


import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    #print("Used default matplotlib style.")
    ...


"""
properties we may be interested in

maximum force if it is a local minimum

energy and force distribution (both train and test)

use DeepEval to check uncertainty

"""

def estimate_number_composition():
    mini = 320
    base = 32


    for coef in range(1,19):
        #print(mini*coef**0.5)
        print(np.ceil(mini*coef**0.5/base)*base)

    return


def transform_forces(symbols, forces):
    """"""
    type_list = ["O", "Pt"]
    type_map = {"O": 0, "Pt": 1}
    # print(symbols)
    forces = np.array(forces)
    print(forces.shape)
    elemental_forces = {}
    for elem in type_map.keys():
        elemental_forces[elem] = []
    for idx, symbol in enumerate(symbols):
        # print(type_list[symbol])
        elemental_forces[type_list[symbol]].extend(
            list(forces[:, 3*idx:3*(idx+1)].flatten())
        )
    for elem in elemental_forces.keys():
        elemental_forces[elem] = np.array(elemental_forces[elem])
        print(elemental_forces[elem].shape)
    elemental_forces = elemental_forces

    return elemental_forces


def plot_hist(ax, data, xlabel, ylabel, label=None):
    num_bins = 50
    n, bins, patches = ax.hist(data, num_bins, density=False, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return

def parse_constraint_info(cons_info):
    """ python convention starts from 0 and excludes end
        lammps convention starts from 1 and includes end
    """
    cons_indices = None
    if cons_info is not None:
        print("apply constraint...")
        cons_indices = []
        for c in cons_info.split():
            s, e = c.split(":")
            cons_indices.extend(list(range(int(s)-1, int(e))))

    return

class DatasetSystems():

    """ PlainTextDatasetSystem
    """

    def __init__(
        self,
        system_config_file
    ):
        input_dict = parse_input_file(system_config_file)
        self.database_path = Path(input_dict["database"])
        self.systems = input_dict["systems"]
        self.sift_settings = input_dict["sift"]

        for sys_name in self.systems.keys():
            self.__check_system_name(sys_name, self.systems[sys_name]["composition"])

        return
    
    def __check_system_name(self, sys_name, composition):
        """ system name should be one of three formats
            1. simple chemical formula
            2. custom name plus chemical formula
            3. custom name, chemical formula and system type
        """
        name_parts = sys_name.split("-")
        if len(name_parts) == 1:
            chemical_formula = name_parts[0]
        elif len(name_parts) == 2:
            custom_name, chemical_formula = name_parts
        elif len(name_parts) == 3:
            custom_name, chemical_formula, system_type = name_parts
        else:
            raise ValueError("directory name must be as xxx, xxx-xxx, or xxx-xxx-xxx")

        composition_from_name = self.__parse_composition_from_name(chemical_formula)
        assert composition_from_name == composition, f"{sys_name}'s composition is not consistent."

        return

    def __parse_composition_from_name(self, chemical_formula: str):
        """"""
        from itertools import groupby
        formula_list = [
            "".join(g) for _, g in groupby(chemical_formula, str.isalpha)
        ]
        iformula = iter(formula_list)
        number_dict = dict(zip(iformula, iformula))  # number of atoms per type

        type_list = []
        for atype, num in number_dict.items():
            number_dict[atype] = int(num)
            if number_dict[atype] > 0:
                type_list.append(atype)
        assert len(type_list) > 0, "no atomic types were found."

        return number_dict

    def read_all_systems(
        self, system_path,
        sys_pattern = "*", file_pattern = "*.xyz"
    ) -> List[Atoms]:
        print("===== load all systems =====")
        # TODO: should sort dirs to make frames consistent
        #system_path = self.database_path
        system_path = Path(system_path)
        print(system_path)

        nframes = 0
        total_frames = {}
        #dirs = list(system_path.glob(sys_pattern)) 
        dirs = []
        for sys_name in self.systems.keys():
            dirs.append(system_path / sys_name)
            print(system_path / sys_name)
        dirs.sort()
        for p in dirs:
            cur_frames = self.read_single_system(p, file_pattern)
            total_frames[p.name] = cur_frames
            nframes += len(cur_frames)
        print("===== Total number: ", nframes)

        return total_frames

    def read_single_system(
        self, system_path: Union[str, pathlib.Path],
        pattern: int = "*.xyz"
    ) -> List[Atoms]:
        print("----- read frames -----")
        print("system path: ", str(system_path))
        # TODO: should sort dirs to make frames consistent

        # TODO: check composition is the same
        total_frames = []
        stru_files = list(system_path.glob(pattern)) # structure files, xyz for now
        print("wocao: ", stru_files)
        stru_files.sort()
        for p in stru_files:
            frames = read(p, ":")
            total_frames.extend(frames)
            print("structure path: ", p)
            print("number of frames: ", len(frames))
        print("Total number: ", len(total_frames))

        # sift structures based on atomic energy and max forces
        sift_criteria = self.sift_settings
        use_sift = sift_criteria.get("enabled", False)
        
        if use_sift:
            total_frames = self.sift_structures(
                total_frames,
                energy_tolerance = sift_criteria["atomic_energy"],
                force_tolerance = sift_criteria["max_force"]
            )
        exit()

        return total_frames

    def sift_structures(
        self, 
        frames,
        energy_tolerance = None, 
        force_tolerance = None
    ) -> NoReturn:
        """ sift structures based on atomic energy and max forces
        """
        # remove large-force frames
        print("----- sift structures -----")
        if force_tolerance is None and energy_tolerance is None:
            print("no criteria is applied...")
        else:
            if energy_tolerance is None:
                energy_tolerance = 1e8
            if force_tolerance is None:
                force_tolerance = 1e8
            print("energy criteria: ", energy_tolerance, " [eV]")
            print("force criteria: ", force_tolerance, " [eV/Å]")
            new_frames = []
            for idx, atoms in enumerate(frames):
                avg_energy = atoms.get_potential_energy() / len(atoms)
                max_force = np.max(np.fabs(atoms.get_forces()))
                if max_force > force_tolerance: 
                    print(f"skip {idx} with large forces {max_force}")
                    continue
                elif avg_energy > energy_tolerance:
                    print(f"skip {idx} with large energy {avg_energy}")
                    continue
                else:
                    new_frames.append(atoms)
            print("number of frames after sift: ", len(new_frames))

        return new_frames
    
    def __str__(self):
        content = "===== DataSystems =====\n"
        content += f"nsystems {len(self.systems.keys())}\n"
        content += "===== End DataSystems =====\n"

        return content

class DataOperator():

    """
    """

    def __init__(
        self, main_dir, 
        systems, global_type_list,
        name: str,
        pattern, 
        calc,
        geometric_settings,
        sift_settings,
        compress_settings = None,
        selection_settings = None
    ):
        """ parse names...
        """
        # get global type list
        self.calc = calc

        # TODO: in a simpler way...
        self.sift_settings = sift_settings

        # geometric convergence
        self.geometric_settings = geometric_settings
        self.fmax = geometric_settings.get("fmax", 0.05)

        # compress settings
        self.compress_params = compress_settings

        self.name = name # system name

        name_parts = name.split("-")
        if len(name_parts) == 1:
            chemical_formula = name_parts[0]
        elif len(name_parts) == 2:
            custom_name, chemical_formula = name_parts
        elif len(name_parts) == 3:
            custom_name, chemical_formula, system_type = name_parts
        else:
            raise ValueError("directory name must be as xxx-xxx-xxx")

        if name == "ALL":
            self.type_list, self.type_numbers = [], []
        else:
            self.type_list, self.type_numbers = self.__parse_type_list(
                chemical_formula)

        # TODO: results dir
        self.prefix = Path.cwd() / self.name

        # selection settings
        if selection_settings is not None:
            self.selector = create_selector(selection_settings, directory=self.prefix)

        # --- system specific ---
        self.systems = systems # contains composition and constraint

        # parse structures
        self.frames = None  # List of Atoms
        self.main_dir = main_dir
        self.pattern = pattern
        if name != "ALL":
            self.is_single = True
            self.system_names = [self.name]
            #self.frames = self.read_single_system(
            #    Path(main_dir) / name,
            #    pattern = pattern
            #)
        else:
            self.is_single = False # whether a single dataset is read
            self.system_names = list(self.systems.keys())
            #self.global_type_list = []
            #for sys_name in self.system_names:
            #    # update type_list
            #    chemical_formula = "".join(
            #        [k+str(v) for k, v in self.systems[sys_name]["composition"].items()]
            #    )
            #    self.type_list, self.type_numbers = self.__parse_type_list(
            #        chemical_formula
            #    )
            #    self.global_type_list.extend(self.type_list)
            #self.global_type_list = sorted(list(set(self.global_type_list)))
        self.global_type_list = global_type_list

        # check system accessibility
        accessible_sys_names = []
        for sys_name in self.systems.keys():
            sys_path = self.main_dir / sys_name
            if sys_path.exists():
                nxyzfiles = len(list(sys_path.glob("*.xyz")))
                if nxyzfiles > 0:
                    accessible_sys_names.append(sys_name)
                else:
                    print(f"xxx Empty {sys_path}. xxx")
            else:
                print(f"xxx Cant find {sys_path}. xxx")
        acc_systems = {key: value for key, value in self.systems.items() if key in accessible_sys_names}
        self.systems = acc_systems

        for name in self.systems.keys():
            print(name)

        return

    def __parse_type_list(self, chemical_formula: str):
        from itertools import groupby
        formula_list = [
            "".join(g) for _, g in groupby(chemical_formula, str.isalpha)
        ]
        iformula = iter(formula_list)
        number_dict = dict(zip(iformula, iformula))  # number of atoms per type

        type_list = []
        for atype, num in number_dict.items():
            number_dict[atype] = int(num)
            if number_dict[atype] > 0:
                type_list.append(atype)
                assert atype in self.global_type_list, f"atype {atype} not in global type list."
        assert len(type_list) > 0, "no atomic types were found."

        return type_list, number_dict

    def read_all_systems(
        self, system_path: Union[str, pathlib.Path],
        sys_pattern = "*",
        file_pattern = "*.xyz"
    ) -> List[Atoms]:
        print("----- read frames -----")
        print(str(system_path))
        # TODO: should sort dirs to make frames consistent

        nframes = 0
        total_frames = {}
        #dirs = list(system_path.glob(sys_pattern)) 
        dirs = []
        for sys_name in self.systems.keys():
            dirs.append(system_path / sys_name)
        dirs.sort()
        for p in dirs:
            cur_frames = self.read_single_system(p, file_pattern)
            total_frames[p.name] = cur_frames
            nframes += len(cur_frames)
        print("===== Total number: ", nframes)

        return total_frames

    def read_single_system(
        self, system_path: Union[str, pathlib.Path],
        pattern: int = "*.xyz"
    ) -> List[Atoms]:
        print("----- read frames -----")
        print("system path: ", str(system_path))
        # TODO: should sort dirs to make frames consistent

        # TODO: check composition is the same
        total_frames = []
        stru_files = list(system_path.glob(pattern)) # structure files, xyz for now
        #print(pattern, " wocao: ", stru_files)
        stru_files.sort()
        for p in stru_files:
            frames = read(p, ":")
            total_frames.extend(frames)
            print("structure path: ", p)
            print("number of frames: ", len(frames))
        print("Total number: ", len(total_frames))

        # TODO: assert frames have same composition and lattice

        # sift structures based on atomic energy and max forces
        sift_criteria = self.sift_settings
        use_sift = sift_criteria.get("enabled", False)
        
        if use_sift:
            total_frames = self.sift_structures(
                total_frames,
                energy_tolerance = sift_criteria["atomic_energy"],
                force_tolerance = sift_criteria["max_force"]
            )

        return total_frames

    def sift_structures(
        self, 
        frames,
        energy_tolerance = None, 
        force_tolerance = None
    ) -> NoReturn:
        """ sift structures based on atomic energy and max forces
        """
        # remove large-force frames
        print("----- sift structures -----")
        if force_tolerance is None and energy_tolerance is None:
            print("no criteria is applied...")
        else:
            if energy_tolerance is None:
                energy_tolerance = 1e8
            if force_tolerance is None:
                force_tolerance = 1e8
            print("energy criteria: ", energy_tolerance, " [eV]")
            print("force criteria: ", force_tolerance, " [eV/Å]")
            new_frames = []
            for idx, atoms in enumerate(frames):
                avg_energy = atoms.get_potential_energy() / len(atoms)
                max_force = np.max(np.fabs(atoms.get_forces()))
                if max_force > force_tolerance: 
                    print(f"skip {idx} with large forces {max_force}")
                    continue
                elif avg_energy > energy_tolerance:
                    print(f"skip {idx} with large energy {avg_energy}")
                    continue
                else:
                    new_frames.append(atoms)
            print("number of frames after sift: ", len(new_frames))

        return new_frames

    def split_frames(self, count=0):
        merged_indices = []
        for i in range(count):
            previous_indices_path = pathlib.Path("r" + str(i) + "-indices.npy")
            indices = np.load(previous_indices_path).tolist()
            merged_indices.extend(indices)
        assert len(merged_indices) == len(set(merged_indices)), "Have duplicated structures... {0} != {1}".format(
            len(merged_indices), len(set(merged_indices))
        )
        used_frames, other_frames = [], []
        for i, atoms in enumerate(self.frames):
            if i in merged_indices:
                used_frames.append(atoms)
            else:
                other_frames.append(atoms)
        print(
            "Number of used frames: ",
            len(used_frames),
            "Number of other frames: ",
            len(other_frames)
        )

        return used_frames, other_frames

    def reduce_dataset(self, number, count, name, energy_shift):
        """"""
        frames = self.frames
        # write(patword+"-tot.xyz", frames)
        prefix = "r" + str(count) + "-"
        if count == 0:
            if number > 0:
                converged_frames = []
                converged_indices = []
                # add converged structures
                for idx, atoms in enumerate(frames):
                    # add constraint to neglect forces of frozen atoms
                    cons = FixAtoms(
                        indices=[a.index for a in atoms if a.position[2] < 2.5])
                    atoms.set_constraint(cons)
                    max_force = np.max(np.fabs(atoms.get_forces()))
                    if max_force < 0.05:
                        converged_frames.append(atoms)
                        converged_indices.append(idx)

                # select frames
                nconverged = len(converged_frames)
                print("Number of converged: ", nconverged)
                features = calc_desc(frames)
                selected_frames = select_structures(
                    name, features, number-nconverged, prefix=prefix,
                    manually_selected=converged_indices
                )  # include manually selected structures
                # selected_frames.extend(converged_frames)
                # TODO: sometims the converged one will be selected... so duplicate...

                if energy_shift > 0:
                    es = energy_shift
                    print(
                        f"add {es} [eV] energy correction to each structure!!!")
                    for atoms in selected_frames:
                        new_energy = atoms.get_potential_energy() + es
                        new_forces = atoms.get_forces(
                            apply_constraint=False).copy()
                        calc = SinglePointCalculator(
                            atoms, energy=new_energy, forces=new_forces)
                        atoms.calc = calc

                print("Writing structure file... ")
                write((prefix+'-sel.xyz'), selected_frames)
                print("")
        else:
            merged_indices = []
            for i in range(count):
                previous_indices_path = pathlib.Path(
                    "r" + str(i) + "-indices.npy")
                indices = np.load(previous_indices_path).tolist()
                merged_indices.extend(indices)
            print(
                "Number of unique frames: ",
                len(set(merged_indices))
            )

            if previous_indices_path.exists() and args.more:
                features_path = cwd / "features.npy"
                features = np.load(features_path)

                assert len(frames) == features.shape[0]

                index_map = []
                rest_features = []
                rest_frames = []
                for idx, atoms in enumerate(frames):
                    if idx not in merged_indices:
                        rest_frames.append(atoms)
                        rest_features.append(features[idx])
                        index_map.append(idx)
                if len(rest_frames) < number:
                    raise ValueError("Not enough structures left...")
                rest_features = np.array(rest_features)
                selected_frames = select_structures(
                    args.name, rest_features, number, index_map=index_map, prefix=prefix)

                print("Writing structure file... ")
                write(cwd / (prefix+patword+'-sel.xyz'), selected_frames)
                print("")

        return

    def find_converged(
        self, sys_name, sys_frames, res_dir
    ):
        # parse constraint indices
        cons_info = self.systems[sys_name].get("constraint", None)
        cons_type, cons_data = None, None
        if cons_info is not None:
            if isinstance(cons_info, dict):
                cons_type = "region"
                cons_data = float(cons_info["zmax"])
            else:
                cons_type = "index"
                if cons_info is not None:
                    print("apply constraint...")
                    cons_indices = []
                    for c in cons_info.split():
                        s, e = c.split(":")
                        cons_indices.extend(list(range(int(s)-1, int(e))))
                cons_data = cons_indices
        print(f"cons: {cons_type} data: {cons_data}")

        # check converged structures
        converged_indices = []
        converged_frames = []
        for i, atoms in enumerate(sys_frames):
            # check if it has constraint
            if cons_type is not None:
                if cons_type == "index":
                    cons = FixAtoms(indices=cons_data)
                    atoms.set_constraint(cons)
                elif cons_type == "region":
                    cons_indices = []
                    for ci, pz in enumerate(atoms.positions[:,2]):
                        if pz < cons_data:
                            cons_indices.append(ci)
                    cons = FixAtoms(indices=cons_indices)
                    atoms.set_constraint(cons)
                else:
                    raise RuntimeError(f"Unknown constraint {cons_type}")
            else:
                # TODO: attached constraint
                cons = atoms.constraints
                if cons:
                    assert len(cons) == 1, "found two constraints on one atoms object"
                    assert isinstance(cons[0], FixAtoms), "the constraint is not FixAtoms"
                else:
                    # no constraints
                    pass
            max_force = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            if (max_force < self.fmax):
                atoms.info["description"] = "local minimum"
                converged_frames.append(atoms)
                converged_indices.append(i)
        print("converged structure index: ", converged_indices)

        if len(converged_frames) > 0 and self.geometric_settings["enabled"]:
            converged_energies = [a.get_potential_energy() for a in converged_frames]
            converged_frames.sort(key=lambda a:a.get_potential_energy())
            write(res_dir / "local_minimum.xyz", converged_frames)

        return converged_frames, converged_indices
    
    def compress_systems(
        self,
        minisize = 640, # minimum extra number of structures
        energy_tolerance=0.020,
        energy_shift=0.0
    ):
        # check if selector exists
        if not hasattr(self, "selector"):
            raise RuntimeError("selector is not created...")

        if not self.is_single:
            # save data
            title = ("#{:<23s}  "+"{:<12s}  "*4+"\n").format(
                "SysName", "natoms", "nframes", "ncandidates", "nselected"
            )
            with open(Path.cwd() / "all-compress.dat", "w") as fopen:
                fopen.write(title)

        for sys_name in self.systems.keys():
            # run operation
            print(f"===== Current System {sys_name} =====")
            # update type_list and sys_name
            chemical_formula = "".join(
                [k+str(v) for k, v in self.systems[sys_name]["composition"].items()]
            )
            self.type_list, self.type_numbers = self.__parse_type_list(
                chemical_formula
            )
            natoms = sum(self.type_numbers.values())
            #global_type_list.extend(self.type_list)
            self.name = sys_name

            nframes, ncandidates, nselected, nconverged = self.compress_single_system(
                sys_name, minisize, energy_tolerance, energy_shift
            )

            if not self.is_single:
                # save data
                content = ("{:<24s}  "+"{:<12d}  "*5+"\n").format(
                    sys_name, natoms, nframes, nconverged, ncandidates, nselected
                )
                # TODO: check all
                #all_file = Path.cwd() / "all-compress.dat"
                with open(Path.cwd() / "all-compress.dat", "a") as fopen:
                    fopen.write(content)

        return

    def compress_single_system(
        self,
        sys_name,
        minisize = 640, # minimum extra number of structures
        energy_tolerance=0.020,
        energy_shift=0.0
    ):
        """ select candidates based on 
            true error or model ensemble deviation

            possible criteria 
            energy error 0.020 eV / atom
        """
        nselected, nconverged, ncandidates = 0, 0, 0

        # update a few outputs path based on system
        self.prefix = Path.cwd() / sys_name
        if self.prefix.exists():
            pass
        else:
            self.prefix.mkdir()
        self.selector.directory = self.prefix

        mlp_file = self.prefix / (self.name + "-MLP.xyz")
        out_xyz = self.prefix / ("miaow-sel.xyz")
        if out_xyz.exists():
            print(f"result dir {self.prefix} exists, may skip calculation...")
            return 0, 0, 0

        if self.calc is not None:
            print("!!!use calculator-assisted dataset compression...!!!")
            calc = self.calc

            batchsize = self.compress_params["batchsize"]

            print(f"tolerance {energy_tolerance} shift {energy_shift}")

            if not mlp_file.exists():
                print("\n\n--- calculate mlp results ---")
                # read frames
                frames = self.read_single_system(self.main_dir / sys_name, self.pattern)
                nframes = len(frames)
                
                if nframes > 0:
                    ref_energies, dft_forces = xyz2results(frames, calc=None)

                    #print(frames[0].get_potential_energy())

                    calc_frames = [a.copy() for a in frames] # NOTE: shallow copy of frames will contaminate results
                    #print(frames[0].get_potential_energy())
                    mlp_energies, mlp_forces = xyz2results(calc_frames, calc=calc)

                    # preprocess
                    ref_energies = np.array(ref_energies)
                    mlp_energies = np.array(mlp_energies)
            else:
                frames = read(mlp_file, ":") # TODO: change this into another function
                calc_name = calc.name.lower()

                ref_energies, ref_forces = xyz2results(frames, calc=None)

                mlp_energies = np.array([a.info[calc_name+"_energy"] for a in frames])
                # mlp_forces = merge_predicted_forces(frames, calc_name) # save time for now
            nframes = len(frames)
            if nframes > 0:
                # TODO: check force errors
                natoms_per_structure = np.sum(list(self.type_numbers.values()))

                print("\n\n--- data statistics ---")
                abs_error = np.fabs(ref_energies - mlp_energies) / natoms_per_structure
                print(abs_error)
                print("error shape:")
                print(abs_error.shape)
                print("mean: ", np.mean(abs_error))
                _hist, _bin_edges = np.histogram(abs_error, bins=10)
                print("histogram: ")
                print(_hist)
                print(_bin_edges)

                cand_indices = []  # structure index with large error
                cand_frames = []
                for i, x in enumerate(abs_error):
                    if x > energy_tolerance:
                        cand_indices.append(i)
                        # NOTE: copy atoms wont copy calc
                        _energy = frames[i].get_potential_energy()
                        _forces = frames[i].get_forces()
                        cand_atoms = frames[i].copy()
                        # print(cand_atoms.calc)
                        singlepoint = SinglePointCalculator(
                            cand_atoms, energy=_energy,
                            forces=_forces
                        )
                        cand_atoms.calc = singlepoint
                        # print(cand_atoms.calc)
                        # print(cand_atoms.get_potential_energy())
                        # cand_atoms.calc = None # torch results cannot be direct to joblib
                        cand_frames.append(cand_atoms)
                ncand = len(cand_indices)
                print("number of candidates: ", ncand)

                num_cur = int(
                    np.min([minisize, np.floor(ncand / batchsize)*batchsize]))
                print("number to select: ", num_cur)

                # go for cur
                if num_cur <= 0:
                    print("no selection...")
                    ncandidates, nselected = 0, 0
                else:
                    ncandidates = len(cand_frames)
                    nselected, nconverged = self.compress_system_based_on_structural_diversity(cand_frames, num_cur, energy_shift)
        else:
            print("!!!use descriptor-based dataset compression...!!!")
            # read frames
            sys_frames = self.read_single_system(self.main_dir / sys_name, self.pattern)
            nframes = len(sys_frames)
            if nframes > 0:
                ncandidates = nframes
                nselected, nconverged = self.compress_system_based_on_structural_diversity(sys_frames, minisize, energy_shift)

        # save data
        if nframes > 0:
            title = ("#{:<23s}  "+"{:<12s}  "*4+"\n").format(
                #"SysName", "natoms", "nframes", "dE_RMSE", "dE_Std"
                "SysName", "natoms", "nframes", "ncandidates", "nselected"
            )
            content = ("{:<24s}  "+"{:<12d}  "*4+"\n").format(
                self.name, sum(self.type_numbers.values()), 
                nframes, ncandidates, nselected
            )
            with open(self.prefix / "compress.dat", "w") as fopen:
                fopen.write(title+content)
        else:
            ncandidates, nselected = 0, 0

        return nframes, ncandidates, nselected, nconverged

    def compress_system_based_on_structural_diversity(
        self, 
        frames,
        number = 320,
        energy_shift = 0.0
    ):
        """ Strategy for compressing large dataset
            1. if atomic energy is way large, skip them
            2. random selection based on (
                cur_scores * bin_prop * boltzmann
                # geometric, phase volume, energy-wise
            )
        """

        print("\n\n")
        print("!!! number of frames to compress: ", len(frames))

        #print(frames[0].info)
        #write("fuccc.xyz", frames)
        #print(frames[0].get_potential_energy())

        # find converged structures TODO: move this outside
        converged_frames, converged_indices = self.find_converged(self.name, frames, self.prefix)
        nconverged = len(converged_indices)
        print("Number of converged: ", nconverged)

        # cur in each bin
        print("\n\n--- start histogram ---")
        energies = np.array([a.get_potential_energy() for a in frames])
        # convert to atomic energies
        energies = energies / np.sum(list(self.type_numbers.values()))

        en_width = self.compress_params["en_width"]  # energy width
        en_min = np.floor(np.min(energies))
        en_max = np.ceil(np.max(energies))
        bin_edges = np.arange(en_min, en_max+en_width, en_width)
        nbins = len(bin_edges) - 1  # number of bins

        print(f"AtomicEnergy Range: {en_min} _ {en_max}")
        print(f"number of bins: {nbins}")

        _, bin_edges, binnumber = sp.stats.binned_statistic(
            energies, energies, statistic="mean", bins=bin_edges,
        )

        # add boltzmann coefficient
        beta = self.compress_params["beta"]  # eV
        shifts = self.compress_params["shifts"]
        boltz_prob = np.zeros(len(bin_edges)-1)
        for es in shifts:
            boltz_prob += np.exp(-beta*(bin_edges[:-1]-(en_min+es)))
        norm_boltz = boltz_prob / np.sum(boltz_prob)
        print(f"Boltzmann {boltz_prob}, norm: {norm_boltz}")

        # get frame index in each bin
        fidx_list = []  # frame index list
        for x in range(nbins):
            fidx_list.append([])
        for k, g in groupby(enumerate(binnumber), lambda x: x[1]):
            # k starts with 1 since histogram uses right boundary
            fidx_list[k-1].extend([ig[0] for ig in g])
        num_array = np.zeros(nbins, dtype=int)
        for x in range(nbins):
            num_array[x] = len(fidx_list[x])
        num_density = num_array / np.sum(num_array)
        print(f"NumDensity: {num_array}, norm: {num_density}")

        assert len(frames) == np.sum(
            num_array), "all frames must be in the hist."

        # choose number of structures in eacn bin
        merged_probs = num_density*norm_boltz
        merged_probs = merged_probs / np.sum(merged_probs)
        print("BoltzDensity: ", merged_probs)
        print("random choice: ", number-nconverged)
        if number > nconverged:
            selected_indices = np.random.choice(
                nbins, number-nconverged, replace=True, p=merged_probs)

            final_num_array = np.zeros(nbins, dtype=int)
            for k, g in groupby(selected_indices):
                final_num_array[k] += len(list(g))
            print("Numbers selected in each bin: ", final_num_array)

            # select structures in each bin
            print("\n\n--- start cur decompostion ---")
            # check if selected numbers are valid
            new_final_selected_num = []
            rest_num = 0
            for bin_i, (bin_num, bin_idx) in enumerate(zip(final_num_array, fidx_list)):
                nframes_bin = len(bin_idx) # number of structure in the bin
                #if bin_num == 0:
                #    new_final_selected_num.append(bin_num)
                #    continue
                bin_num += rest_num
                if bin_num > nframes_bin:
                    rest_num = bin_num - nframes_bin
                    bin_num = nframes_bin
                else:
                    rest_num = 0
                #if rest_num < 0:
                #    bin_num -= rest_num
                #rest_num = nframes_bin - bin_num # all minus selected
                #if rest_num < 0:
                #    bin_num = nframes_bin
                new_final_selected_num.append(bin_num)
            print("NEW: Numbers selected in each bin: ", new_final_selected_num)
            all_indices = []
            for bin_i, (bin_num, bin_idx) in enumerate(zip(new_final_selected_num, fidx_list)):
                nframes_bin = len(bin_idx)
                if bin_num == 0 or nframes_bin == 0:
                    continue
                print(
                    "{} candidates are selected from {} structures in bin {}...".format(
                        bin_num, nframes_bin, bin_i
                    )
                )
                #bin_frames = []
                #for i in bin_idx:
                #    bin_frames.append(frames[i])
                bin_frames = [frames[i] for i in bin_idx]
                #features = self.selector.calc_desc(bin_frames)
                #if len(bin_frames) == 1:
                #    features = features[:, np.newaxis]
                #    selected_indices = [0]
                #else:
                #    selected_indices = self.selector.select_structures(
                #        features, bin_num
                #    )
                selected_indices = self.selector.select(bin_frames, ret_indices=True)
                # NOTE: map selected into global indices
                selected_indices = [bin_idx[s] for s in selected_indices]
                    # TODO: sometims the converged one will be selected... so duplicate...
                all_indices.extend(selected_indices)
            all_indices.extend(converged_indices)
        else:
            print("nconverged is greater than number...")
            converged_energies = np.array([a.get_potential_energy() for a in converged_frames])
            converged_probs = -converged_energies + np.max(converged_energies) + 0.1
            converged_probs = converged_probs / np.sum(converged_probs)
            all_indices = np.random.choice(
                converged_indices, number, replace=False, p=converged_probs
            )
        print("number selected: ", len(all_indices))
        print("number no-duplicated: ", len(set(all_indices)))

        selected_frames = []
        for i in all_indices:
            selected_frames.append(frames[i])

        # output final candidates
        if energy_shift > 0:
            es = energy_shift
            print(f"add {es} [eV] energy correction to each structure!!!")
            for atoms in selected_frames:
                new_energy = atoms.get_potential_energy() + es
                new_forces = atoms.get_forces(apply_constraint=False).copy()
                calc = SinglePointCalculator(
                    atoms, energy=new_energy, forces=new_forces)
                atoms.calc = calc

        print("Writing structure file... ")
        selected_frames.sort(key=lambda a:a.get_potential_energy()) # sort energy
        write(self.prefix / ("miaow-sel.xyz"), selected_frames)
        print("")

        return len(all_indices), nconverged

    @staticmethod
    def __binned_statistic(x, values, func, nbins, limits):
        '''The usage is nearly the same as scipy.stats.binned_statistic'''
        from scipy.sparse import csr_matrix

        N = len(values)
        r0, r1 = limits

        digitized = (float(nbins)/(r1 - r0)*(x - r0)).astype(int)
        S = csr_matrix((values, [digitized, np.arange(N)]), shape=(nbins, N))

        return [func(group) for group in np.split(S.data, S.indptr[1:-1])]
    
    def __run_scalculation(self):
        """ run calculation to obtain dataset statistics (rmse)
        """

        return
    
    def show_statistics(self):
        """
        """
        if not self.is_single:
            # analyse all results
            #energies, forces, en_rmse, force_rmse
            title = ("#{:<23s}  {:<12s}  {:<12s}  {:<12s}  {:<12s}  ").format(
                "SysName", "natoms", "nframes", "dE_RMSE", "dE_Std"
            )
            for x in self.global_type_list:
                title += "{:<12s}  {:<12s}  ".format(x+"_dF_RMSE", x+"_dF_Std")
            title += "\n"
            with open("all_rmse.dat", "w") as fopen:
                fopen.write(title)

        nframes_dict = {}
        rmse_results = {}
        all_energies = None
        all_forces = None
        for sys_name in self.systems.keys():
            print(f"\n\n===== System {sys_name} =====")
            # update chemical info to this system
            # TODO: change this part to the DataSystem
            chemical_formula = "".join(
                [k+str(v) for k, v in self.systems[sys_name]["composition"].items()]
            )
            self.type_list, self.type_numbers = self.__parse_type_list(
                chemical_formula
            )
            # update a few outputs path based on system
            skipped = False
            self.prefix = Path.cwd() / sys_name
            if self.prefix.exists():
                print(f"result dir {self.prefix} exists, skip compression...")
                # check MLP data
                print("Use saved property data...")
                existed_data = self.prefix / (sys_name + "-MLP.xyz")
                # TODO: check
                if existed_data.exists():
                    new_frames = read(existed_data, ":")
                    calc_name = self.calc.name.lower()

                    ref_energies, ref_forces = xyz2results(new_frames, calc=None)
                    natoms_array = np.array([len(x) for x in new_frames])

                    mlp_energies = [a.info[calc_name+"_energy"] for a in new_frames]
                    mlp_forces = merge_predicted_forces(new_frames, calc_name) 

                    energies = np.array([ref_energies, mlp_energies])
                    energies = np.array([ref_energies, mlp_energies]) / natoms_array
                    forces = [ref_forces, mlp_forces]
                    
                    en_rmse = {}
                    en_rmse["energy"] = rms_dict(energies[0], energies[1])
                    force_rmse = {}
                    for key in forces[0].keys():
                        force_rmse[key] = rms_dict(
                            forces[0][key], forces[1][key]
                        )
                else:
                    # read frames
                    sys_frames = self.read_single_system(self.main_dir / sys_name, self.pattern)
                    if sys_frames:
                        energies, forces, en_rmse, force_rmse = self.show_single_statistics(
                            sys_name, sys_frames
                        )
                    else:
                        skipped = True
            else:
                self.prefix.mkdir()
                # read frames
                sys_frames = self.read_single_system(self.main_dir / sys_name, self.pattern)
                if sys_frames:
                    energies, forces, en_rmse, force_rmse = self.show_single_statistics(
                        sys_name, sys_frames
                    )
                else:
                    skipped = True

            # --- add info ---
            if not self.is_single and not skipped:
                # add system-wise rmse results
                nframes_dict[sys_name] = energies.shape[1]
                rmse_results[sys_name] = (en_rmse, force_rmse)
                # merge energy and forces
                if all_energies is None:
                    all_energies = energies
                else:
                    all_energies = np.hstack([all_energies, energies])
                print("energy shape: ", all_energies.shape)
                if all_forces is None:
                    all_forces = forces
                    for key in forces[0].keys():
                        print(f"ref {key} force length: ", len(all_forces[0][key]))
                    for key in forces[1].keys():
                        print(f"mlp {key} force length: ", len(all_forces[1][key]))
                else:
                    # ref
                    for key in forces[0].keys():
                        if key in all_forces[0].keys():
                            all_forces[0][key].extend(forces[0][key])
                        else:
                            all_forces[0][key] = forces[0][key]
                        print(f"ref {key} force length: ", len(all_forces[0][key]))
                    # mlp
                    for key in forces[1].keys():
                        if key in all_forces[1].keys():
                            all_forces[1][key].extend(forces[1][key])
                        else:
                            all_forces[1][key] = forces[1][key]
                        print(f"mlp {key} force length: ", len(all_forces[1][key]))

                natoms = sum(self.systems[sys_name]["composition"].values())
                content = "{:<24s}  {:<12d}  {:<12d}  ".format(sys_name, natoms, nframes_dict[sys_name])
                content += ("{:<12.4f}  "*2).format(en_rmse["energy"]["rmse"], en_rmse["energy"]["std"])
                for x in self.global_type_list:
                    force_rmse_x = force_rmse.get(x, {})
                    x_rmse = force_rmse_x.get("rmse", np.NaN)
                    x_std = force_rmse_x.get("std", np.NaN)
                    content += ("{:<12.4f}  "*2).format(x_rmse, x_std)
                content += "\n"

                with open("all_rmse.dat", "a") as fopen:
                    fopen.write(content)

        if not self.is_single and all_energies is not None:
            # all 
            en_rmse, force_rmse = self.__plot_comparasion(
                "ALL", all_energies, all_forces, "ALL-cmp.png"
            )
            content = "{:<24s}  {:<12d}  {:<12d}  ".format("ALL", 0, sum(nframes_dict.values()))
            content += ("{:<12.4f}  "*2).format(en_rmse["energy"]["rmse"], en_rmse["energy"]["std"])
            for x in self.global_type_list:
                force_rmse_x = force_rmse.get(x, {})
                x_rmse = force_rmse_x.get("rmse", np.NaN)
                x_std = force_rmse_x.get("std", np.NaN)
                content += ("{:<12.4f}  "*2).format(x_rmse, x_std)
            content += "\n"

            with open("all_rmse.dat", "a") as fopen:
                fopen.write(content)

        return

    def show_single_statistics(
        self, 
        sys_name,
        frames
    ):
        """ parameters
            fmax: 0.05
        """
        print(f"----- {sys_name} statistics -----")
        # results
        res_dir = self.prefix

        # check converged structures
        # NOTE: ignore converged structures now
        #converged_frames, converged_indices = self.find_converged(self.name, frames, self.prefix)
        #nconverged = len(converged_indices)
        #print("Number of converged: ", nconverged)

        #if len(converged_frames) > 0:
        #    converged_energies = [a.get_potential_energy() for a in converged_frames]
        #    converged_frames.sort(key=lambda a:a.get_potential_energy())
        #    write(res_dir / "local_minimum.xyz", converged_frames)

        # use loaded frames
        nframes = len(frames)
        ref_energies, ref_forces = xyz2results(frames, calc=None)
        natoms_array = np.array([len(x) for x in frames])

        if self.calc is not None:
            st = time.time()
            # TODO: compare with MLP
            existed_data = res_dir / (sys_name + "-MLP.xyz")
        
            # use loaded frames
            calc_name = self.calc.name.lower()
            print("Calculating with MLP...")
            new_frames = append_predictions(frames, calc=self.calc)
            #mlp_energies, mlp_forces = xyz2results(frames, calc=calc)
            mlp_energies = [a.info[calc_name+"_energy"] for a in new_frames]
            mlp_forces = merge_predicted_forces(new_frames, calc_name) 
            energies = np.array([ref_energies, mlp_energies])

            # save data
            write(existed_data, new_frames)

            forces = np.array([ref_forces, mlp_forces])
            energies = np.array([ref_energies, mlp_energies]) / natoms_array

            et = time.time()
            print("calculation time: ", et-st)

            en_rmse, force_rmse = self.__plot_comparasion(calc_name, energies, forces, res_dir / (sys_name + "-cmp.png"))

        # plot stat
        fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))
        axarr = axarr.flatten()

        plt.suptitle("Dataset Overview")

        plot_hist(axarr[0], np.array(ref_energies).flatten(),
                  "Energy [eV]", "Number of Frames")
        for x in self.type_list:
            plot_hist(
                axarr[1], np.array(ref_forces[x]).flatten(
                ), "Force [eV/AA]", "Number of Frames",
                label=x
            )
        axarr[1].legend()

        plt.savefig(res_dir / (sys_name + "-stat.png"))
        
        # save rmse dat
        self.__write_rmse_results(
            sys_name, natoms_array[0], nframes, en_rmse, force_rmse
        )

        return energies, forces, en_rmse, force_rmse

    @staticmethod
    def test_uncertainty_consistent(frames, calc, saved_figure=None):
        # use loaded frames
        ref_energies, ref_forces = xyz2results(frames, calc=None)
        natoms_array = np.array([len(x) for x in frames])
        ref_energies = np.array(ref_energies)

        mlp_energies, mlp_forces, mlp_svars = xyz2results(
            frames, calc=calc, other_props=["en_stdvar", "force_stdvar"]
        )
        mlp_energies = np.array(mlp_energies)
        en_stdvar = np.array(mlp_svars["en_stdvar"])

        # save energy data
        ener_data = np.vstack([natoms_array, ref_energies, mlp_energies, en_stdvar])
        np.savetxt("en_devi.dat", ener_data.T)

        # save forces data
        forces_O = np.vstack(
            [np.array(ref_forces["O"]), np.array(mlp_forces["O"])]
        )
        np.savetxt("forces_O.dat", forces_O.T)
        forces_Pt = np.vstack(
            [np.array(ref_forces["Pt"]), np.array(mlp_forces["Pt"])]
        )
        np.savetxt("forces_Pt.dat", forces_Pt.T)

        # save data
        #with open("./ref_data.json", "w") as fopen:
        #    json.dump(fopen, ref_energies)

        # plot data
        fig, axarr = plt.subplots(
            nrows=2, ncols=2,
            gridspec_kw={'hspace': 0.3}, figsize=(16, 16)
        )
        axarr = axarr.flat[:]

        fig.suptitle("Committee Model Uncertainty Estimation")

        ax = axarr[0]
        parity_plot_dict(
            {'energy': np.array(ref_energies)/natoms_array},
            {'energy': np.array(mlp_energies)/natoms_array},
            ax,
            {
                "xlabel": "DFT [eV]",
                "ylabel": "%s [eV]" % calc.name.upper(),
                "title": "(a) Energy Comparasion"
            }
        )

        ax = axarr[1]
        true_deviations = np.fabs(
            np.array(ref_energies) - np.array(mlp_energies)) / natoms_array
        predicted_deviations = mlp_svars["en_stdvar"] / natoms_array
        parity_plot_dict(
            {"deviation": predicted_deviations},
            {"deviation": true_deviations},  # absolute?
            ax,
            {
                "xlabel": "%s Uncertainty [eV]" % calc.name.upper(),
                "ylabel": "True Error [eV]",
                "title": "(b) Energy Uncertainty"
            },
            [1.0, 10.0],
            write_info=False
        )

        # TODO: force deviation
        ax = axarr[2]
        parity_plot_dict(
            {"Pt": ref_forces["Pt"]},
            {"Pt": mlp_forces["Pt"]},
            ax,
            {
                "xlabel": "DFT [eV/Å]",
                "ylabel": "%s [eV/Å]" % calc.name.upper(),
                "title": "(c) Platinum Forces"
            }
        )

        ax = axarr[3]
        parity_plot_dict(
            {"O": ref_forces["O"]},
            {"O": mlp_forces["O"]},
            ax,
            {
                "xlabel": "DFT [eV/Å]",
                "ylabel": "%s [eV/Å]" % calc.name.upper(),
                "title": "(d) Oxygen Forces"
            }
        )

        plt.savefig(saved_figure)

        return

    @staticmethod
    def test_uncertainty_evolution(frames, calc, saved_file=None):
        def recalc_atoms(calc, atoms):
            # can't pickle torch or tf
            calc.reset()
            calc.calc_uncertainty = True
            atoms.calc = calc
            dummy = atoms.get_forces()
            singlepoint = SinglePointCalculator(
                atoms, energy=atoms.calc.results["energy"],
                forces=atoms.calc.results["forces"],
            )
            atoms.info["en_stdvar"] = atoms.calc.results["en_stdvar"]
            atoms.calc = singlepoint
            return atoms
        # overwrite info in frames
        print("Test Uncertainty Evolution...")
        saved_file = Path(saved_file)
        if saved_file.exists():
            frames = read(saved_file, ":")
        else:
            # frames = Parallel(n_jobs=4)(delayed(recalc_atoms)(calc, atoms) for atoms in frames)
            for atoms in tqdm(frames):
                calc.reset()
                calc.calc_uncertainty = True
                atoms.calc = calc
                dummy = atoms.get_forces()
                singlepoint = SinglePointCalculator(
                    atoms, energy=atoms.calc.results["energy"],
                    forces=atoms.calc.results["forces"],
                )
                atoms.info["en_stdvar"] = atoms.calc.results["en_stdvar"]
                atoms.calc = singlepoint
            write(saved_file, frames)

        # write to file
        content = "#{}  {}  {}  {}\n".format(
            "step   ", "natoms  ", "energy  ", "deviation")
        for i, atoms in enumerate(frames):
            content += ("{:8d}  "*2+"{:8.4f}  "*2+"\n").format(
                i, len(atoms), atoms.get_potential_energy(
                ), atoms.info["en_stdvar"]
            )
        with open("evolution.dat", "w") as fopen:
            fopen.write(content)

        fig, axarr = plt.subplots(
            nrows=2, ncols=1,
            gridspec_kw={'hspace': 0.3}, figsize=(16, 12)
        )
        axarr = axarr.flatten()
        plt.suptitle("GCMC Evolution")

        ax = axarr[0]
        # element_numbers = Counter()
        natoms_array = np.array([len(a) for a in frames])
        # oxygen_array = np.array([Counter(a.get_chemical_symbols())["O"] for a in frames])
        steps = range(len(frames))
        energies = np.array([a.get_potential_energy()
                            for a in frames]) / natoms_array
        en_stdvars = np.array([a.info["en_stdvar"]
                              for a in frames]) / natoms_array

        ax.set_xlabel("MC Step")
        ax.set_ylabel("Energy [eV]")

        ax.plot(steps, energies, label="energy per atom")
        apex = 10.
        ax.fill_between(
            steps, energies-apex*en_stdvars, energies+apex*en_stdvars, alpha=0.2,
            label="10 times deviation per atom"
        )
        for i in range(1, 5001, 500):
            ax.text(
                steps[i], energies[i]+apex *
                en_stdvars[i], "%.4f" % en_stdvars[i],
                fontsize=16
            )

        ax.legend()

        # coverage and number of oxygen atoms
        ax = axarr[1]

        nsurfatoms = int(len(frames[0])/6)
        print("Number of surface atoms: ", nsurfatoms)
        noxygen_array = np.array(
            [Counter(a.get_chemical_symbols())["O"] for a in frames])
        coverages = noxygen_array / nsurfatoms
        ax.plot(steps, coverages, color="r", label="coverage of O")

        ax.set_xlabel("MC Step")
        ax.set_ylabel("Coverage [ML]")

        ax.legend()

        plt.savefig("m-stdvar.png")

        return
    
    def __write_rmse_results(
        self, sys_name, natoms, nframes,
        en_rmse, force_rmse
    ):
        """"""
        # save rmse dat
        title = ("#{:<23s}  "+"{:<12s}  "*4).format(
            "SysName", "natoms", "nframes", "dE_RMSE", "dE_Std"
        )
        for x in self.type_list:
            title += "{:<12s}  {:<12s}  ".format(x+"_dF_RMSE", x+"_dF_Std")
        title += "\n"

        content = "{:<24s}  {:<12d}  {:<12d}  ".format(sys_name, natoms, nframes)
        content += ("{:<12.4f}  "*2).format(en_rmse["energy"]["rmse"], en_rmse["energy"]["std"])
        for x in self.type_list:
            content += ("{:<12.4f}  "*2).format(force_rmse[x]["rmse"], force_rmse[x]["std"])
        content += "\n"

        with open(self.prefix / "rmse.dat", "w") as fopen:
            fopen.write(title+content)
        
        print(title+content)

        return

    @staticmethod
    def __plot_comparasion(calc_name, energies, forces, saved_figure):
        """"""
        fig, axarr = plt.subplots(
            nrows=1, ncols=2,
            gridspec_kw={'hspace': 0.3}, figsize=(16, 12)
        )
        axarr = axarr.flatten()

        _, nframes = energies.shape
        plt.suptitle(f"Number of frames {nframes}")

        ax = axarr[0]
        en_rmse_results = parity_plot_dict(
            {"energy": energies[0, :]}, {"energy": energies[1, :]}, 
            ax, 
            {
                "xlabel": "DFT [eV]", "ylabel": "%s [eV]" % calc_name, "title": "Energy"
            }
        )

        ax = axarr[1]
        for_rmse_results = parity_plot_dict(
            forces[0], forces[1],
            ax,
            {
                "xlabel": "DFT [eV/AA]",
                "ylabel": "%s [eV/AA]" % calc_name,
                "title": "Forces"
            }
        )

        plt.savefig(saved_figure)

        return en_rmse_results, for_rmse_results


if __name__ == "__main__":
    # print("See usage in utils.anal_data")
    # test DatasetSystems
    ds = DatasetSystems("/mnt/scratch2/users/40247882/PtOx-dataset/systems.yaml")
    print(ds)
    ds.read_all_systems("/users/40247882/scratch2/oxides/eann-main/Latest-Compressed")

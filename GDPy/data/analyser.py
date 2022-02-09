#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from genericpath import exists
from itertools import groupby
import json
from pickletools import read_decimalnl_long

import numpy as np
import scipy as sp

from GDPy.selector.structure_selection import calc_feature, cur_selection, select_structures
from GDPy.utils.comparasion import parity_plot_dict
from GDPy.data.dpsets import append_predictions, merge_predicted_forces, xyz2results
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read, write
from ase import Atoms
from tqdm import tqdm
import pathlib
from pathlib import Path
import argparse

from typing import NoReturn, Union, List, overload
from joblib import Parallel, delayed

from collections import Counter


import matplotlib as mpl
mpl.use('Agg')  # silent mode
from matplotlib import pyplot as plt
plt.style.use('presentation')


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


def check_convergence(forces):
    """
    forces nframes*(3*natoms) array
    """
    max_forces = np.max(np.fabs(forces), axis=1)

    return max_forces


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


# TODO: move this to the input file
DESC_JSON_PATH = "/mnt/scratch2/users/40247882/catsign/eann-main/soap_param.json"
DESC_JSON_PATH = "/mnt/scratch2/users/40247882/oxides/eann-main/soap_param.json"


def calc_desc(frames):
    # calculate descriptor to select minimum dataset
    with open(DESC_JSON_PATH, "r") as fopen:
        desc_dict = json.load(fopen)
    desc_dict = desc_dict["soap"]

    # TODO: inputs
    njobs = 4

    cwd = pathlib.Path.cwd()
    features_path = cwd / "features.npy"
    # if features_path.exists():
    #    print("use precalculated features...")
    #    features = np.load(features_path)
    #    assert features.shape[0] == len(frames)
    # else:
    #    print('start calculating features...')
    #    features = calc_feature(frames, desc_dict, njobs, features_path)
    #    print('finished calculating features...')
    print('start calculating features...')
    features = calc_feature(frames, desc_dict, njobs, features_path)
    print('finished calculating features...')

    return features


def select_structures(
    features, num, zeta=-1, strategy="descent", index_map=None,
    prefix="",  # make prefix to the directory path
):
    """ 
    """
    # cur decomposition
    cur_scores, selected = cur_selection(features, num, zeta, strategy)

    # map selected indices
    if index_map is not None:
        selected = [index_map[s] for s in selected]
    # if manually_selected is not None:
    #    selected.extend(manually_selected)

    # TODO: if output
    # content = '# idx cur sel\n'
    # for idx, cur_score in enumerate(cur_scores):
    #     stat = 'F'
    #     if idx in selected:
    #         stat = 'T'
    #     if index_map is not None:
    #         idx = index_map[idx]
    #     content += '{:>12d}  {:>12.8f}  {:>2s}\n'.format(idx, cur_score, stat)
    # with open((prefix+"cur_scores.txt"), 'w') as writer:
    #    writer.write(content)
    #np.save((prefix+"indices.npy"), selected)

    #selected_frames = []
    # for idx, sidx in enumerate(selected):
    #    selected_frames.append(frames[int(sidx)])

    return selected


class DataOperator():

    """
    """

    def __init__(self, name: str):
        """ parse names...
        """
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

        self.frames = None  # List of Atoms

        self.prefix = Path.cwd()

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
        assert len(type_list) > 0, "no atomic types were found."

        return type_list, number_dict
    
    def register_calculator(self, calc):
        """"""
        self.calc = calc

        return

    def read_all_frames(
        self, system_path: Union[str, pathlib.Path],
        pattern: int = "*.xyz"
    ) -> List[Atoms]:
        print("----- read frames -----")
        print(str(system_path))
        # TODO: should sort dirs to make frames consistent

        total_frames = []
        dirs = list(system_path.glob("O*"))
        dirs.sort()
        for p in dirs:
            print(p)
            for x in p.glob("*.xyz"):
                frames = read(x, ":")
                print("number of frames: ", len(frames))
                total_frames.extend(frames)
        print("Total number: ", len(total_frames))

        return total_frames

    def read_frames(
        self, system_path: Union[str, pathlib.Path],
        pattern: int = "*.xyz"
    ) -> List[Atoms]:
        print("----- read frames -----")
        print(str(system_path))
        # TODO: should sort dirs to make frames consistent

        total_frames = []
        dirs = list(system_path.glob(pattern))
        dirs.sort()
        for p in dirs:
            print(p)
            frames = read(p, ":")
            print("number of frames: ", len(frames))
            total_frames.extend(frames)
        print("Total number: ", len(total_frames))

        return total_frames

    def remove_large_force_structures(self) -> NoReturn:
        # remove large-force frames
        # TODO: move this part to the main reading procedure
        force_tolerance = 50
        print("force criteria: ", force_tolerance, " [eV/Å]")
        new_frames = []
        for atoms in self.frames:
            max_force = np.max(np.fabs(atoms.get_forces()))
            if max_force > force_tolerance:  # TODO
                # skip large-force structure
                continue
            else:
                new_frames.append(atoms)
        self.frames = new_frames
        print("number of frames after removing large-force ones: ", len(self.frames))

        return

    def remove_large_energy_structures(self) -> NoReturn:
        """ remove structure with large atomic energy
        """
        MAX_AEAVG = -1.4  # TODO: add this to input
        remove_unstable = True  # if remove structures with large energy and forces
        if remove_unstable:
            new_frames = []
            for idx, atoms in enumerate(self.frames):
                ae_avg = atoms.get_potential_energy() / len(atoms)
                if ae_avg < MAX_AEAVG:
                    new_frames.append(atoms)
                else:
                    print(
                        "Find {} unstable structure with energy {}".format(
                            idx, ae_avg)
                    )
            self.frames = new_frames

        return

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

    def compress_based_on_deviation(
        self, calc,
        minisize = 640, # minimum extra number of structures
        energy_tolerance=0.020,
        energy_shift=0.0
    ):
        """ select candidates based on 
            true error or model ensemble deviation

            possible criteria 
            energy error 0.020 eV / atom
        """
        batchsize = 32
        # minisize = 320  # minimum extra number of structures

        print(f"tolerance {energy_tolerance} shift {energy_shift}")

        frames = self.frames
        # calc_frames = frames.copy() # WARN: torch-based calc will contaminate results

        dft_energies, dft_forces = xyz2results(frames, calc=None)
        mlp_energies, mlp_forces = xyz2results(frames, calc=calc)

        natoms_array = np.array([len(x) for x in frames])
        forces = [dft_forces, mlp_forces]
        energies = np.array([dft_energies, mlp_energies]) / natoms_array

        self.__plot_comparasion(calc.name, energies, forces, "test.png")

        # preprocess
        print("\n\n--- data statistics ---")
        dft_energies = np.array(dft_energies)
        mlp_energies = np.array(mlp_energies)
        natoms_per_structure = np.sum(list(self.type_numbers.values()))
        natoms_array = np.ones(dft_energies.shape) / natoms_per_structure

        abs_error = np.fabs(dft_energies - mlp_energies) / natoms_per_structure
        print(abs_error.shape)
        print("mean: ", np.mean(abs_error))
        _hist, _bin_edges = np.histogram(abs_error, bins=10)
        print("histogram: ")
        print(_hist)
        print(_bin_edges)

        cand_indices = []  # structure index with large error
        cand_frames = []
        for i, x in enumerate(abs_error):
            # TODO: make this a parameter
            if x > energy_tolerance:
                cand_indices.append(i)
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
        # exit()

        num_cur = int(
            np.min([minisize, np.floor(ncand / batchsize)*batchsize]))
        print("number to select: ", num_cur)

        # go for cur
        self.compress_frames(num_cur, cand_frames, energy_shift)

        return

    def compress_frames(
        self, number=320,
        frames=None,
        energy_shift=0.0
    ):
        """ Strategy for compressing large dataset
            1. if atomic energy is way large, skip them
            2. random selection based on (
                cur_scores * bin_prop * boltzmann
                # geometric, phase volume, energy-wise
            )
        """
        print("\n\n")
        if frames is None:
            print("Use pre-loaded dataset...")
            frames = self.frames.copy()
        else:
            print("Use user-input dataset...")
        print("!!! number of frames in compressing: ", len(frames))
        # for atoms in frames:
        #    print(atoms)

        # find converged structures
        converged_indices = []
        for idx, atoms in enumerate(frames):
            # TODO: add constraint to neglect forces of frozen atoms
            cons = FixAtoms(
                indices=[a.index for a in atoms if a.position[2] < 2.0])
            atoms.set_constraint(cons)
            max_force = np.max(np.fabs(atoms.get_forces()))
            if max_force < 0.05:  # TODO:
                converged_indices.append(idx)

        # select frames
        nconverged = len(converged_indices)
        print("Number of converged: ", nconverged)

        # cur in each bin
        print("\n\n--- start histogram ---")
        energies = np.array([a.get_potential_energy() for a in frames])
        # convert to atomic energies
        energies = energies / np.sum(list(self.type_numbers.values()))

        en_width = 0.5  # energy width
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
        beta = 3.0  # eV
        shifts = [0., 1., 2.]  # TODO:
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
            nframes_bin = len(bin_idx)
            if bin_num == 0:
                new_final_selected_num.append(bin_num)
                continue
            if rest_num < 0:
                bin_num -= rest_num
            rest_num = nframes_bin - bin_num # all minus selected
            if rest_num < 0:
                bin_num = nframes_bin
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
            features = calc_desc(bin_frames)
            if len(bin_frames) == 1:
                # TODO:
                features = features[:, np.newaxis]
                continue
            selected_indices = select_structures(
                features, bin_num, prefix=self.prefix
            )
            # TODO: sometims the converged one will be selected... so duplicate...
            all_indices.extend(selected_indices)
        all_indices.extend(converged_indices)

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

        return

    @staticmethod
    def __binned_statistic(x, values, func, nbins, limits):
        '''The usage is nearly the same as scipy.stats.binned_statistic'''
        from scipy.sparse import csr_matrix

        N = len(values)
        r0, r1 = limits

        digitized = (float(nbins)/(r1 - r0)*(x - r0)).astype(int)
        S = csr_matrix((values, [digitized, np.arange(N)]), shape=(nbins, N))

        return [func(group) for group in np.split(S.data, S.indptr[1:-1])]

    def check_xyz(self):
        """"""
        # check converged structures
        converged_frames = []
        for i, atoms in enumerate(self.frames):
            # TODO: make fixed atom indices as parameter
            cons = FixAtoms(
                indices=[a.index for a in atoms if a.position[2] < 2.5])
            atoms.set_constraint(cons)
            max_force = np.max(np.fabs(atoms.get_forces()))
            if (max_force < 0.05):
                print("converged structure index: ", i)
                atoms.info["description"] = "local minimum"
                converged_frames.append(atoms)
        if len(converged_frames) > 0:
            converged_energies = [a.get_potential_energy() for a in converged_frames]
            converged_frames.sort(key=lambda a:a.get_potential_energy())
            write("local_minimum.xyz", converged_frames)

        # use loaded frames
        ref_energies, ref_forces = xyz2results(self.frames, calc=None)
        natoms_array = np.array([len(x) for x in self.frames])

        # TODO: if force in given interval
        # for k in ref_forces.keys():
        #    ref_forces[k] = np.ma.masked_outside(ref_forces[k], -10., 10.)
        #    print(ref_forces[k])

        fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))
        axarr = axarr.flatten()

        plt.suptitle('Dataset Overview')

        plot_hist(axarr[0], np.array(ref_energies).flatten(),
                  "Energy [eV]", "Number of Frames")
        for x in self.type_list:
            plot_hist(
                axarr[1], np.array(ref_forces[x]).flatten(
                ), "Force [eV/AA]", "Number of Frames",
                label=x
            )
        axarr[1].legend()

        plt.savefig('wang.png')

        return

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

    def test_frames(self, fig_name=None) -> NoReturn:
        """test xyz energy and force errors"""
        calc = self.calc
        frames = self.frames

        if len(frames) == 0:
            print("No frames...")
            return

        nframes = len(frames)
        natoms_array = np.array([len(x) for x in frames])

        res_dir = Path(self.name)
        if not res_dir.exists():
            res_dir.mkdir()
        existed_data = res_dir / "new_frames.xyz"
        
        # use loaded frames
        calc_name = calc.name
        if not existed_data.exists():
            dft_energies, dft_forces = xyz2results(frames, calc=None)
            new_frames = append_predictions(frames, calc=calc)
            #mlp_energies, mlp_forces = xyz2results(frames, calc=calc)
            mlp_energies = [a.info["mlp_energy"] for a in new_frames]
            mlp_forces = merge_predicted_forces(new_frames) # BUG: not forces
            energies = np.array([dft_energies, mlp_energies])

            # save data
            write(existed_data, new_frames)

            forces = [dft_forces, mlp_forces]
            energies = np.array([dft_energies, mlp_energies]) / natoms_array
        else:
            print("Use saved property data...")
            raise NotImplementedError("TODO: use pre-calculated data...")

        self.__plot_comparasion(calc_name, energies, forces, res_dir / fig_name)

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
        rmse_results = parity_plot_dict({'energy': energies[0, :]}, {'energy': energies[1, :]}, ax, {
                         'xlabel': 'DFT [eV]', 'ylabel': '%s [eV]' % calc_name, 'title': "Energy"})
        print(rmse_results)

        ax = axarr[1]
        rmse_results = parity_plot_dict(
            forces[0], forces[1],
            ax,
            {
                'xlabel': 'DFT [eV]',
                'ylabel': '%s [eV]' % calc_name,
                'title': "Energy"
            }
        )
        print(rmse_results)

        plt.savefig(saved_figure)

        return


if __name__ == "__main__":
    print("See usage in utils.anal_data")

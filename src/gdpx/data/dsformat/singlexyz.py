#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import pathlib
from typing import List, Tuple, Union

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from .. import registers
from ..dataset import AbstractDataloader


def group_structures_by_composition(frames: List[Atoms]):
    """"""
    # Check chemical symbols
    system_dict = {}  # {formula: [indices]}

    # NOTE: groupby only collects contiguous data
    #       we need aggregate by ourselves
    formulae = [a.get_chemical_formula() for a in frames]
    for k, v in itertools.groupby(enumerate(formulae), key=lambda x: x[1]):
        if k not in system_dict:
            system_dict[k] = [x[0] for x in v]
        else:
            system_dict[k].extend([x[0] for x in v])

    # # transfer data
    # acc_nframes = 0
    # for formula, curr_indices in system_dict.items():
    #     # -- TODO: check system type
    #     system_type = self.system # currently, use user input one
    #     # -- name = description+formula+system_type
    #     #dirname = "-".join([self.directory.parent.name, formula, system_type])
    #     dirname = "-".join([self.prefix, formula, system_type])
    #
    #     target_subdir = target_dir/dirname
    #     target_subdir.mkdir(parents=True, exist_ok=True)
    #
    #     # -- save frames
    #     curr_frames = [frames[i] for i in curr_indices]
    #     curr_nframes = len(curr_frames)
    #
    #     if self.set_pbc:
    #         for atoms in curr_frames:
    #             atoms.set_pbc(True)
    #
    #     if self.clean_info:
    #         self._clean_frames(curr_frames)
    #
    #     strname = self.version + ".xyz"
    #     target_destination = target_dir/dirname/strname
    #     if not target_destination.exists():
    #         write(target_destination, curr_frames)
    #         self._print(f"nframes {curr_nframes} -> {str(target_destination.relative_to(target_dir))}")
    #     else:
    #         #warnings.warn(f"{target_destination} exists.", UserWarning)
    #         self._print(f"WARN: {str(target_destination.relative_to(target_dir))} exists.")
    #
    #     acc_nframes += curr_nframes
    # assert nframes == acc_nframes

    return system_dict


def map_atoms_data(atoms: Atoms, prop_map_keys: List[Tuple[str, str]], clean_data: bool = False) -> Atoms:
    """"""
    if atoms.calc is not None:
        assert type(atoms.calc) == SinglePointCalculator, f"Atoms {atoms} has {atoms.calc}."
        stored_results = atoms.calc.results
    else:
        spc = SinglePointCalculator(atoms)
        atoms.calc = spc
        stored_results = {}
        stored_results.update(**atoms.info)
        stored_results.update(**atoms.arrays)

    # For some small dataset, structures can be used both as train and test,
    # thus, we need add a flag to avoid repeated mapping.
    is_mapped = atoms.info.get("is_mapped", False)

    if not clean_data:
        new_atoms = atoms
    else:
        new_atoms = Atoms(
            atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc()
        )
        spc = SinglePointCalculator(new_atoms)
        new_atoms.calc = spc

    if not is_mapped:
        mapped_results = dict()
        for mapping_pairs in prop_map_keys:
            dst_key, src_key = mapping_pairs
            if src_key in stored_results:
                # NOTE: The dst_data is overwritten if it exists
                src_val = copy.deepcopy(stored_results.get(src_key))
                mapped_results[dst_key] = src_val
            else:
                raise KeyError(f"Atoms {atoms} has no key {src_key}.")
        new_atoms.calc.results = mapped_results  # type: ignore
        new_atoms.info["is_mapped"] = True
    else:
        ...

    return new_atoms


class SingleXyzDataloader(AbstractDataloader):

    name: str = "single_xyz"

    """"""

    def __init__(self, dataset_path: Union[str, pathlib.Path], *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.dataset_path = pathlib.Path(dataset_path).resolve()

        return

    def load_frames(self, *args, **kwargs):
        """"""
        frames = read(self.dataset_path, ":")
        num_frames = len(frames)
        self._print(f"{num_frames=}")

        sys_dict = group_structures_by_composition(frames)

        acc_num_frames = 0
        systems = []
        for system, indices in sys_dict.items():
            sys_num_frames = len(indices)
            acc_num_frames += sys_num_frames
            self._print(f"{system=:>48s}  num_frames={sys_num_frames}")
            sys_frames = [frames[i] for i in indices]
            sys_frames = [
                map_atoms_data(
                    a,
                    prop_map_keys=[
                        ("energy", "dft_energy"),
                        ("free_energy", "dft_free_energy"),
                        ("forces", "dft_forces"),
                        ("initial_charges", "charge_bader"),
                    ],
                    clean_data=True
                ) for a in sys_frames
            ]
            systems.append((system, sys_frames))
        assert num_frames == acc_num_frames

        return systems


if __name__ == "__main__":
    ...

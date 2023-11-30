#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Union
from itertools import groupby
from operator import itemgetter

import numpy as np

from ase.constraints import constrained_indices, FixAtoms


def convert_indices(indices: Union[str,List[int]], index_convention="lmp"):
    """ parse indices for reading xyz by ase, get start for counting
        constrained indices followed by lammps convention
        "2:4 3:8"
        convert [1,2,3,6,7,8] to "1:3 6:8"
        lammps convention starts from 1 and includes end
        ---
        input can be either py or lmp
        output for indices is in py since it can be used to access atoms
        output for text is in lmp since it can be used in lammps or sth
    """
    ret = []
    if isinstance(indices, str):
        # string to List[int]
        for x in indices.strip().split():
            cur_range = list(map(int, x.split(":")))
            if len(cur_range) == 1:
                start, end = cur_range[0], cur_range[0]
            else:
                start, end = cur_range
            if index_convention == "lmp":
                ret.extend([i-1 for i in list(range(start,end+1))])
            elif index_convention == "py":
                ret.extend(list(range(start,end)))
            else:
                pass
    elif isinstance(indices, list):
        # List[int] to string
        indices = sorted(indices)
        if index_convention == "lmp":
            pass
        elif index_convention == "py":
            indices = [i+1 for i in indices]
        ret = []
        #ranges = []
        for k, g in groupby(enumerate(indices),lambda x:x[0]-x[1]):
            group = (map(itemgetter(1),g))
            group = list(map(int,group))
            #ranges.append((group[0],group[-1]))
            if group[0] == group[-1]:
                ret.append(str(group[0]))
            else:
                ret.append("{}:{}".format(group[0],group[-1]))
        ret = " ".join(ret)
    else:
        pass

    return ret

def parse_constraint_info(atoms, cons_text, ignore_ase_constraints=True, ret_text=True) -> List[int]:
    """ constraint info can be any forms below, 
        and transformed into indices that start from 1
        "2:5 8" means 2,3,4,5,8 (default uses lmp convention)
        "py 0:5 9" means 1,2,3,4,5,10
        "lmp 1:4 8" means 1,2,3,4,8
        "lowest 10" means 10 atoms with smallest z-positions
        "zpos 4.5" means all atoms with zpositions smaller than 4.5

        return lammps format text or python format list
    """
    # - set some init values
    aindices = list(range(len(atoms)))
    mobile_indices = aindices.copy()
    frozen_indices = []
    #print("constraint: ", cons_text)

    # TODO: check if atoms have constraint
    # NOTE: only need to determine which atoms are frozen then others are mobile
    cons_indices = constrained_indices(atoms, only_include=FixAtoms) # array
    if (not ignore_ase_constraints) and cons_indices.size > 0:
        # convert to lammps convention
        frozen_indices = cons_indices.copy().tolist()
    else:
        # TODO: if use region indicator
        if cons_text is None:
            pass
        else:
            # - parse constraint text type
            cons_data = cons_text.split()
            if cons_data[0] not in ["py", "lmp", "lowest", "zpos"]:
                cons_type, cons_info = "lmp", " ".join(cons_data)
            else:
                cons_type, cons_info = cons_data[0], " ".join(cons_data[1:])
            #print("cons_info: ", cons_type, cons_info)

            # NOTE: text may have different notations
            #       but indices should all be in python convention
            if cons_type == "py":
                frozen_indices = convert_indices(cons_info, index_convention="py")
            elif cons_type == "lmp":
                frozen_indices = convert_indices(cons_info, index_convention="lmp")
            elif cons_type == "lowest":
                if atoms.pbc[2]:
                    z_coordinates = atoms.get_scaled_positions()[:, 2]
                    wrapped_coordinates = []
                    for z in z_coordinates: # wrap to [-0.1, 0.9)
                        z_ = np.modf(z)[0]
                        if z_ < -0.1:
                            z_ += 1.0
                        elif z >= 0.9:
                            z_ -= 1.0
                        wrapped_coordinates.append(z_)
                    frozen_indices = sorted(aindices, key=lambda x: wrapped_coordinates[x])[:int(cons_info)]
                else:
                    frozen_indices = sorted(aindices, key=lambda x:atoms.positions[x][2])[:int(cons_info)]
            elif cons_type == "zpos":
                frozen_indices = [i for i in aindices if atoms.positions[i][2] <= float(cons_info)]
            else:
                pass
    mobile_indices = [i for i in aindices if i not in frozen_indices]
    #print(mobile_indices)
    #print(frozen_indices)

    if ret_text:
        frozen_text = convert_indices(frozen_indices, index_convention="py")
        mobile_text = convert_indices(mobile_indices, index_convention="py")

        return mobile_text, frozen_text
    else:
        return mobile_indices, frozen_indices

def set_constraint(atoms, cons_text, ignore_attached_constraints=True):
    """Set constraint based on constraint text."""
    mobile_indices, frozen_indices = parse_constraint_info(
        atoms, cons_text, ignore_ase_constraints=ignore_attached_constraints, ret_text=False
    )
    if frozen_indices:
        if ignore_attached_constraints:
            atoms._del_constraints()
        else:
            ...
        atoms.set_constraint(FixAtoms(indices=frozen_indices))
    else:
        ...

    return

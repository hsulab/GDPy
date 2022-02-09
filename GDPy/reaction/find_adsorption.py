#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np

import spglib

from ase import Atom, utils
from ase.io import read, write

"""
get adsorption sites ->
find symmetry and group structures ->
check them manually
"""
print("spglib version: ", spglib.get_version())

def find_symmetry(atoms):
    """"""
    # find symmetry
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    magmoms = None # The collinear polarizations magmoms only work with get_symmetry 
                   # and are given as a list of N floating point values
    cell = (lattice, positions, numbers)

    spg = spglib.get_spacegroup(cell, symprec=0.1)
    operations = spglib.get_symmetry(cell, symprec=0.1)
    #lattice, scaled_positions, numbers = spglib.find_primitive(cell, symprec=0.1)
    #lattice, scaled_positions, numbers = spglib.standardize_cell(cell, to_primitive=True, no_idealize=False, symprec=1e-5)
    #print("=== primitive")
    #print(lattice)
    #print(scaled_positions)
    #print(numbers)

    #return spg, operations
    return spg

# read atoms
#atoms = read("./surfaces/surf-22/O2/s1fcc1dtet-1/CONTCAR")
#atoms = read("./surfaces/surf-22/O2/surf-1fcc1hcp/CONTCAR")

surface = read("/mnt/scratch2/users/40247882/oxides/surfaces/Pt111/surf-22/surf-22/CONTCAR")
cell = surface.get_cell()

# site is on z 0.35 but O is on 0.4 after opt
#fcc_sites = [
#    [0.0, 0.0, 0.40],
#    [0.5, 0.0, 0.40],
#    [0.0, 0.5, 0.40],
#    [0.5, 0.5, 0.40]
#] # p(2x2)
fcc_sites = [
    [0.0, 0.0, 0.40],
    [0.0, 1/3., 0.40],
    [0.0, 2/3., 0.40],
    [1/3., 0.0, 0.40],
    [1/3., 1/3., 0.40],
    [1/3., 2/3., 0.40],
    [2/3., 0.0, 0.40],
    [2/3., 1/3., 0.40],
    [2/3., 2/3., 0.40]
] # p(3x3)
fcc_sites = [
    [0/4., 0.0, 0.40],
    [0/4., 1/4., 0.40],
    [0/4., 2/4., 0.40],
    [0/4., 3/4., 0.40],
    [1/4., 0.0, 0.40],
    [1/4., 1/4., 0.40],
    [1/4., 2/4., 0.40],
    [1/4., 3/4., 0.40],
    [2/4., 0.0, 0.40],
    [2/4., 1/4., 0.40],
    [2/4., 2/4., 0.40],
    [2/4., 3/4., 0.40],
    [3/4., 0.0, 0.40],
    [3/4., 1/4., 0.40],
    [3/4., 2/4., 0.40],
    [3/4., 3/4., 0.40],
] # p(4x4)
fcc_sites = np.dot(np.array(fcc_sites), cell).tolist()

hcp_sites = [
    [1./6, 1./6, 0.40],
    [2./3, 1./6, 0.40],
    [1./6, 2./3, 0.40],
    [2./3, 2./3, 0.40]
] # p(2x2)
hcp_sites = [
    [1./9, 1./9, 0.40],
    [1./9, 4./9, 0.40],
    [1./9, 7./9, 0.40],
    [4./9, 1./9, 0.40],
    [4./9, 4./9, 0.40],
    [4./9, 7./9, 0.40],
    [7./9, 1./9, 0.40],
    [7./9, 4./9, 0.40],
    [7./9, 7./9, 0.40]
] # p(3x3)
hcp_sites = [
    [1./12, 1./12, 0.40],
    [1./12, 4./12, 0.40],
    [1./12, 7./12, 0.40],
    [1./12, 10./12, 0.40],
    [4./12, 1./12, 0.40],
    [4./12, 4./12, 0.40],
    [4./12, 7./12, 0.40],
    [4./12, 10./12, 0.40],
    [7./12, 1./12, 0.40],
    [7./12, 4./12, 0.40],
    [7./12, 7./12, 0.40],
    [7./12, 10./12, 0.40],
    [10./12, 1./12, 0.40],
    [10./12, 4./12, 0.40],
    [10./12, 7./12, 0.40],
    [10./12, 10./12, 0.40],
] # p(4x4)
hcp_sites = np.dot(np.array(hcp_sites), cell).tolist()

dtet_sites = [ # down tetrahedron site
    [1./6, 1./6, 0.33],
    [2./3, 1./6, 0.33],
    [1./6, 2./3, 0.33],
    [2./3, 2./3, 0.33]
]
dtet_sites = np.dot(np.array(dtet_sites), cell).tolist()

utet_sites = [
    [1./3, 1./3, 0.30],
    [5./6, 1./3, 0.30],
    [1./3, 5./6, 0.30],
    [5./6, 5./6, 0.30]
]
utet_sites = np.dot(np.array(utet_sites), cell).tolist()

def find_sites(base_surface, num: int, sites1, sites2, site_name):
    # generate 3 O on surface
    from itertools import combinations, product
    count = 0
    for i in range(num+1):
        combins = combinations(range(len(sites1)), i)
        j = num - i
        combins2 = combinations(range(len(sites2)), j)
        cur_frames = []
        for c in product(combins, combins2):
            new_atoms = base_surface.copy()
            for x in c[0]:
                new_atoms.append(
                    Atom("O", position=sites1[x])
                )
            for x in c[1]:
                new_atoms.append(
                    Atom("O", position=sites2[x])
                )
            new_atoms.info["combination"] = c
            print("sym: ", find_symmetry(new_atoms))
            cur_frames.append(new_atoms)
            count += 1
        print(f"#frames {i}-{j}: ", len(cur_frames))
        write(site_name+"_frames_{0}-{1}.arc".format(i, j), cur_frames, format="dmol-arc")
        write(site_name+"_frames_{0}-{1}.xtd".format(i, j), cur_frames)
        #exit()
    print("number of combinations: ", count)

    return

def find_sites_symm(base_surface, num: int, sites1, sites2, site_name):
    # generate 3 O on surface
    from itertools import combinations, product
    count = 0
    for i in range(num+1):
        combins = combinations(range(len(sites1)), i)
        j = num - i
        #if i != 2 and j != 1:
        #    continue
        icount = 0
        combins2 = combinations(range(len(sites2)), j)
        cur_frames = []
        for c in product(combins, combins2):
            new_atoms = base_surface.copy()
            for x in c[0]: # site 1
                new_atoms.append(
                    Atom("C", position=sites1[x])
                )
            for x in c[1]: # site 2
                new_atoms.append(
                    Atom("O", position=sites2[x])
                )
            new_atoms.info["combination"] = c
            new_atoms = new_atoms[16:]
            # check if already have this one
            print(f"sym {icount}: ", find_symmetry(new_atoms))
            cur_frames.append(new_atoms)
            icount += 1
            count += 1
        print(f"#frames {i}-{j}: ", len(cur_frames))
        #write(site_name+"_frames_{0}-{1}.arc".format(i, j), cur_frames, format="dmol-arc")
        write(site_name+"_frames_{0}-{1}.xyz".format(i, j), cur_frames)
        #exit()
    print("number of combinations: ", count)

    return

def find_sites_on_basis(base_surface, num: int, sites1, sites2, site_name):
    # read basis structure
    import json
    basis_file = Path("./basis.json")
    if basis_file.exists():
        with open("./basis.json", "r") as fopen:
            basis_config = json.load(fopen)
        print(basis_config)
    else:
        basis_config = {}

    # generate 3 O on surface
    from itertools import combinations, product
    num_dist = [] # site number distribution
    for i in range(num+1):
        j = num - i
        cur_basis = []
        # run over basis
        for k, bc in enumerate(basis_config):
            bi, bj = len(bc["fcc"]), len(bc["hcp"])
            if i >= bi  and j >= bj:
                cur_basis.append(k)
        num_dist.append((i,j,cur_basis))
    print(num_dist)
    count = 0
    for i, j, basis in num_dist:
        if j != 0:
            print("skip hcp sites for now...")
            continue
        if len(basis) == 0:
            print("No basis structures are provided.")
            combins = combinations(range(len(sites1)), i)
            icount = 0
            combins2 = combinations(range(len(sites2)), j)
            cur_frames_sym = {}
            for c in product(combins, combins2):
                new_atoms = base_surface.copy()
                for x in c[0]: # site 1
                    new_atoms.append(
                        Atom("C", position=sites1[x])
                    )
                for x in c[1]: # site 2
                    new_atoms.append(
                        Atom("O", position=sites2[x])
                    )
                new_atoms.info["combination"] = c
                new_atoms = new_atoms[16:]
                # check if already have this one
                sym = find_symmetry(new_atoms)
                print(f"sym {icount}: ", sym)
                label = sym.split("(")[-1].split(")")[0]
                if label not in cur_frames_sym.keys():
                    cur_frames_sym[label] = [new_atoms]
                else:
                    cur_frames_sym[label].append(new_atoms)
                icount += 1
                count += 1
            for sym, cur_frames in cur_frames_sym.items():
                print(f"#frames {i}-{j} with {sym}: ", len(cur_frames))
                write(site_name+"_frames_{0}-{1}-{2}.xyz".format(i, j, "sym"+sym), cur_frames)
        else:            
            print("Use basis structures.")
            cur_frames_sym = {}
            for k in basis: # basis index
                bc = basis_config[k]

                base_atoms = base_surface.copy()
                for x in bc["fcc"]:
                    base_atoms.append(
                        Atom("C", position=sites1[x])
                    )
                for x in bc["hcp"]:
                    base_atoms.append(
                        Atom("O", position=sites2[x])
                    )

                site1_indices = [x for x in range(len(sites1)) if x not in bc["fcc"]]
                site2_indices = [x for x in range(len(sites2)) if x not in bc["hcp"]]
                combins = combinations(site1_indices, i-len(bc["fcc"]))
                combins2 = combinations(site2_indices, j-len(bc["hcp"]))

                icount = 0
                for c in product(combins, combins2):
                    new_atoms = base_atoms.copy()
                    for x in c[0]: # site 1
                        new_atoms.append(
                            Atom("C", position=sites1[x])
                        )
                    for x in c[1]: # site 2
                        new_atoms.append(
                            Atom("O", position=sites2[x])
                        )
                    new_atoms.info["combination"] = c
                    new_atoms = new_atoms[16:]
                    # check if already have this one
                    sym = find_symmetry(new_atoms)
                    print(f"sym {icount}: ", sym)
                    label = sym.split("(")[-1].split(")")[0]
                    if label not in cur_frames_sym.keys():
                        cur_frames_sym[label] = [new_atoms]
                    else:
                        cur_frames_sym[label].append(new_atoms)
                    icount += 1
                    count += 1
            for sym, cur_frames in cur_frames_sym.items():
                print(f"#frames {i}-{j} with {sym}: ", len(cur_frames))
                write(site_name+"_frames_{0}-{1}-{2}.xyz".format(i, j, "sym"+sym), cur_frames)
    print("number of combinations: ", count)

    return


site_dict = {
    "fcc": fcc_sites,
    "hcp": hcp_sites,
    "dtet": dtet_sites,
    "utet": utet_sites
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--number", required=True, type=int,
        help="number of O atoms added"
    )
    parser.add_argument(
        "-s", "--sites", nargs=2, required=True,
        help="two sites [fcc, hcp, dtet, utet]"
    )
    args = parser.parse_args()

    # generate configurations
    sites1, sites2 = site_dict[args.sites[0]], site_dict[args.sites[1]]
    site_name = args.sites[0] + "-" + args.sites[1]
    
    #find_sites(surface, args.number, sites1, sites2, site_name)
    #print(spglib.get_spacegroup_type("p_3_-2\""))
    #find_sites_symm(surface, args.number, sites1, sites2, site_name)
    find_sites_on_basis(surface, args.number, sites1, sites2, site_name)

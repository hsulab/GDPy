#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pull initial structures 
"""

import numpy as np 

import pymatgen as mg 
import pymatgen.core.surface as mg_surface 
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor 

from ase import Atoms
from ase.io import read, write 

"""
Copper 
Dimer
    distance 1.5 - 3.0
Bulk 
    fcc Fm3m mp-30
    hcp P63/mmc mp-989782 
Surface 
    miller index <= 2 
"""

def yield_dimer(symbol, distances=np.arange(1.5,3.1,0.1)):
    """generate dimer curve using ase"""
    box_length = 10. 
    cell = box_length*np.eye(3)
    positions = [[0.,0.,0.],[0.,0.,box_length/2.]]
    dimer = Atoms(symbol+'2', positions=positions, cell=cell)

    dimer_frames = [] 
    for d in distances:
        dimer_bak = dimer.copy() 
        dimer_bak[1].position[2] = d
        dimer_frames.append(dimer_bak)

    return dimer_frames

def scale_bulk(atoms, scalings=np.arange(0.8,1.25,0.05)):
    """scale bulks using ase """
    scaled_frames = []
    for scale in scalings:
        prim_cell = atoms.cell.copy() 
        scaled_atoms = atoms.copy()
        scaled_atoms.set_cell(prim_cell*scale, scale_atoms=True)
        scaled_frames.append(scaled_atoms)
        pass

    return scaled_frames 

def scale_bulk_mg(structure, scalings=np.arange(0.8,1.25,0.05)):
    """scale bulks using pymatgen """
    scaled_frames = []
    prim_latt = structure.lattice 
    prim_vol = structure.volume
    for scale in scalings:
        scaled_structure = structure.copy()
        #scaled_structure.lattice = prim_latt
        scaled_structure.scale_lattice(prim_vol*scale**3)
        #atoms = AseAtomsAdaptor.get_atoms(scaled_structure)
        #print(atoms.positions)
        #print(scaled_structure.sites)
        #print(scaled_structure.coords)
        scaled_frames.append(scaled_structure)

    return scaled_frames 

def yield_slab(frames_mg, max_miller=2):
    """cleave surface with pymatgen"""
    scaled_slabs = [] 
    for structure in frames_mg:
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
        # TODO: should cleave slabs with unstrained bulks ... 
        possible_slabs = mg_surface.generate_all_slabs(
            structure, 
            max_index = max_miller, 
            min_slab_size = 8.0, 
            min_vacuum_size = 12.0, 
            center_slab = True
        )
        #print(len(possible_slabs))
        for slab in possible_slabs:
            #print(slab.miller_index)
            slab.make_supercell([2,2,1])
            scaled_slabs.append(slab)
        #print(structure)

    return scaled_slabs

def perturb_sites():
    """mg can also perturb atoms randomly"""
    return 

def workflow_ase():
    # from ase 
    atoms = read('./mpstrus/Cu_mp-30_conventional_standard.cif')
    frames = scale_bulk(atoms)

    pass 

if __name__ == '__main__':
    # dimer
    dimer_frames = yield_dimer('Pt', distances=np.arange(1.5,3.1,0.1))
    #write('Pt-dimer.xyz', dimer_frames)
    for idx, frame in enumerate(dimer_frames):
        write('Pt2-'+str(idx).zfill(2)+'.xyz', frame)
    dimer_frames = yield_dimer('O', distances=np.arange(0.5,2.1,0.1))
    for idx, frame in enumerate(dimer_frames):
        write('O2-'+str(idx).zfill(2)+'.xyz', frame)
    #write('O-dimer.xyz', dimer_frames)
    exit()

    # read structure, generate bulks and surfaces 
    structure = mg.Structure.from_file('./mpstrus/Cu_mp-30_conventional_standard.cif')
    scaled_bulks = scale_bulk_mg(structure)
    scaled_slabs = yield_slab(scaled_bulks)

    scaled_frames_mg = scaled_bulks + scaled_slabs 
    frames = list(map(AseAtomsAdaptor.get_atoms, scaled_frames_mg))
    frames = dimer_frames + frames 
    write('basic_strus.xyz', frames)
    #print(structure.lattice)

    pass

    #atoms = AseAtomsAdaptor.get_atoms(structure)
    #print(atoms)

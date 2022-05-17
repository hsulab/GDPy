#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np

import xml.etree.ElementTree as ET
from xml.dom import minidom 

from ase import Atoms 
from ase.io import read, write
from ase.constraints import FixAtoms

from pathlib import Path

from GDPy.calculator.vasp import VaspMachine
from GDPy.utils.command import run_command

from joblib import Parallel, delayed

from GDPy import config

def read_xsd2(fd):
    """ read xsd file by Material Studio
    """
    tree = ET.parse(fd)
    root = tree.getroot()

    atomtreeroot = root.find('AtomisticTreeRoot')
    # if periodic system
    if atomtreeroot.find('SymmetrySystem') is not None:
        symmetrysystem = atomtreeroot.find('SymmetrySystem')
        mappingset = symmetrysystem.find('MappingSet')
        mappingfamily = mappingset.find('MappingFamily')
        system = mappingfamily.find('IdentityMapping')

        coords = list()
        cell = list()
        formula = str()

        names = list()
        restrictions = list() 

        for atom in system:
            if atom.tag == 'Atom3d':
                symbol = atom.get('Components')
                formula += symbol

                xyz = atom.get('XYZ')
                if xyz:
                    coord = [float(coord) for coord in xyz.split(',')]
                else:
                    coord = [0.0, 0.0, 0.0]
                coords.append(coord)

                name = atom.get('Name') 
                if name:
                    pass # find name 
                else: 
                    name = symbol + str(len(names)+1) # None due to copy atom 
                names.append(name)

                restriction = atom.get('RestrictedProperties', None)
                if restriction:
                    if restriction.startswith("FractionalXYZ"):  # TODO: may have 1-3 flags
                        restrictions.append(True)
                    else: 
                        raise ValueError('unknown RestrictedProperties')
                else: 
                    restrictions.append(False)
            elif atom.tag == 'SpaceGroup':
                avec = [float(vec) for vec in atom.get('AVector').split(',')]
                bvec = [float(vec) for vec in atom.get('BVector').split(',')]
                cvec = [float(vec) for vec in atom.get('CVector').split(',')]

                cell.append(avec)
                cell.append(bvec)
                cell.append(cvec)

        atoms = Atoms(formula, cell=cell, pbc=True)
        atoms.set_scaled_positions(coords)

        # add constraints 
        fixed_indices = [idx for idx, val in enumerate(restrictions) if val]
        if fixed_indices:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))

        # add two atoms constrained optimisation 
        constrained_indices = [
            idx for idx, name in enumerate(names) if name.endswith('_c') 
        ]
        if constrained_indices:
            assert len(constrained_indices) == 2
            atoms.info["copt"] = constrained_indices

        return atoms
        # if non-periodic system
    elif atomtreeroot.find('Molecule') is not None:
        system = atomtreeroot.find('Molecule')

        coords = list()
        formula = str()

        for atom in system:
            if atom.tag == 'Atom3d':
                symbol = atom.get('Components')
                formula += symbol

                xyz = atom.get('XYZ')
                coord = [float(coord) for coord in xyz.split(',')]
                coords.append(coord)

        atoms = Atoms(formula, pbc=False)
        atoms.set_scaled_positions(coords)
        return atoms

def read_results(pstru, input_dict, indices=None, custom_incar=None):
    """"""
    # get vasp calc params to add extra_info for atoms
    if custom_incar is None:
        input_dict["incar"] = pstru / "INCAR"
    else:
        input_dict["incar"] = Path(custom_incar)
    vasp_machine = VaspMachine(**input_dict)
    vasp_machine.init_creator()

    fmax = vasp_machine.fmax

    # read info
    # TODO: check electronic convergence
    vasprun = pstru / "vasprun.xml"
    if indices is None:
        traj_frames = read(vasprun, ":")
        energies = [a.get_potential_energy() for a in traj_frames]
        maxforces = [np.max(np.fabs(a.get_forces(apply_constraint=True))) for a in traj_frames] # TODO: applt cons?

        print(f"--- vasp info @ {pstru.name} ---")
        print("nframes: ", len(traj_frames))
        print("last energy: ", energies[-1])
        print("last maxforce: ", maxforces[-1])
        print("force convergence: ", fmax)

        last_atoms = traj_frames[-1]
        last_atoms.info["source"] = pstru.name
        last_atoms.info["maxforce"] = maxforces[-1]
        if maxforces[-1] <= fmax:
            write(pstru / (pstru.name + "_opt.xsd"), last_atoms)
            print("write converged structure...")
    else:
        print(f"--- vasp info @ {pstru.name} ---")
        last_atoms = read(vasprun, indices)
        print("nframes: ", len(last_atoms))
        nconverged = 0
        for a in last_atoms:
            maxforce = np.max(np.fabs(a.get_forces(apply_constraint=True)))
            if maxforce < fmax:
                a.info["converged"] = True
                nconverged += 1
        print("nconverged: ", nconverged)

    return last_atoms

def vasp_main(
    pstru, # path, a single file or a directory with many files
    choice,
    indices,
    incar_template,
    cinidices,
    is_sort,
    is_submit
):
    """"""
    # parse structures
    pstru = Path(pstru).resolve()

    # vasp machine
    gdpconfig = Path.home() / ".gdp"
    if gdpconfig.exists() and gdpconfig.is_dir():
        # find vasp config
        vasprc = gdpconfig / "vasprc.json"
        with open(vasprc, "r") as fopen:
            input_dict = json.load(fopen)
    else:
        input_dict = {}
    input_dict["incar"] = incar_template
    input_dict["isinteractive"] = True


    if choice == "create":
        stru_files = []
        if pstru.is_file():
            stru_files.append(pstru)
        elif pstru.is_dir():
            for p in pstru.glob("*.xsd"): # TODO: change this
                stru_files.append(p)
            stru_files.sort()

        vasp_machine = VaspMachine(**input_dict)
    
        # create calculation files
        cwd = Path.cwd()
        for struct_path in stru_files:
            if cwd != Path.cwd(): 
                directory = cwd.resolve() / struct_path.stem
            else:
                directory = Path.cwd().resolve() / struct_path.stem

            if directory.exists(): 
                raise ValueError(f"{directory} exists.")

            atoms = read_xsd2(struct_path)
            #atoms.set_constraint(FixAtoms(indices=range(len(atoms))))

            # sort atoms by symbols and z-positions especially for supercells 
            if is_sort: 
                numbers = atoms.numbers 
                xposes = atoms.positions[:, 0].tolist()
                yposes = atoms.positions[:, 1].tolist()
                zposes = atoms.positions[:, 2].tolist()
                sorted_indices = np.lexsort((xposes,yposes,zposes,numbers))
                atoms = atoms[sorted_indices]

                map_indices = dict(zip(sorted_indices, range(len(atoms))))
                copt = atoms.info.get("copt", None)
                if copt: 
                    new_copt = [map_indices[key] for key in atoms.info["copt"]]
                    atoms.info["copt"] = new_copt 

            vasp_machine.create(atoms, directory)

    elif choice == "freq":
        assert pstru.is_dir(), "input path is not a directory"
        vasprun = pstru / "vasprun.xml"
        if not vasprun.exists():
            raise RuntimeError("vasp calculation may not exist...")
        
        input_dict["incar"] = pstru / "INCAR"
        vasp_machine = VaspMachine(**input_dict)

        frames = read(vasprun, ":")
        atoms = frames[-1]

        assert len(cinidices) > 0, "at least one atom should be free for freq..."

        indices = [i for i in range(len(atoms)) if i not in cinidices]
        cons = FixAtoms(indices=indices)
        atoms.set_constraint(cons)

        directory = Path.cwd().resolve() / (pstru.name + "-freq")
        vasp_machine.create(atoms, directory, task="freq")
    elif choice == "data":
        assert pstru.is_dir(), "input path is not a directory"

        # recursively find all vasp dirs
        vasp_dirs = []
        vasprun = pstru / "vasprun.xml"
        if vasprun.exists():
            vasp_dirs.append(pstru)
        else:
            for p in pstru.iterdir():
                if p.is_dir():
                    if (p/"vasprun.xml").exists():
                        vasp_dirs.append(p)
            vasp_dirs.sort()
        print("find vasp dirs: ", len(vasp_dirs))

        frames = []
        NJOBS = config.NJOBS
        print("njobs...", NJOBS)
        if NJOBS > 1:
            ret = Parallel(n_jobs=NJOBS)(delayed(read_results)(p, input_dict, indices=indices, custom_incar=incar_template) for p in vasp_dirs)
            for frames in ret:
                if isinstance(frames, list):
                    frames.extend(frames)
                elif isinstance(frames, Atoms):
                    frames.append(frames)
                else:
                    raise RuntimeError("Unexpected object from vasp read results...")
        else:
            frames = []
            for p in vasp_dirs:
                atoms = read_results(p, input_dict, indices=indices, custom_incar=incar_template)
                if isinstance(atoms, Atoms):
                    frames.append(atoms)
                else:
                    frames.extend(atoms)
        if indices is None:
            frames = sorted(frames, key=lambda a:a.get_potential_energy(), reverse=False)
        
        if len(frames) > 1:
            write(Path.cwd().name+"_frames.xyz", frames, columns=["symbols", "positions", "move_mask"])
            print("nframes: ", len(frames))

    else:
        pass


    # submit job automatically 
    if is_submit: 
        vasp_machine.submit(directory)

    return

if __name__ == "__main__":
    pass
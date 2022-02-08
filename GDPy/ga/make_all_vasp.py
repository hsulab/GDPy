#!/usr/bin/env python3

import os
import time
import json
import logging
import warnings
import argparse
import subprocess 

import shutil 

import xml.etree.ElementTree as ET
from xml.dom import minidom 

from pathlib import Path 

import numpy as np 

from ase import Atoms 
from ase.io import read, write
from ase.io.vasp import write_vasp 
from ase.calculators.vasp import Vasp2
from ase.constraints import FixAtoms
from ase.calculators.vasp.create_input import GenerateVaspInput

# customised xsd reader 
#import xsd_plus as xsd2 
#from xsd_plus import read_xsd 

vasp_params = {
    # INCAR 
    "nwrite": 2, 
    "istart": 0, 
    "lcharg": False, 
    "lwave": False, 
    "lorbit": 10,
    "npar": 4,
    "xc": "pbe",
    "encut": 400,
    "prec": "Normal",
    "ediff": 1E-5,
    "nelm": 180, 
    "nelmin": 6, 
    "ispin": 1,
    "ismear": 1,
    "sigma": 0.2,
    "algo": "Fast", 
    "lreal": "Auto", 
    "isym": 0, 
    "ediffg": -0.05,
    "nsw": 200,
    "ibrion": 2,
    "isif": 2,
    "potim": 0.02, 
} 


def read_xsd2(fd):
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

                restriction = atom.get('RestrictedProperties')
                if restriction:
                    if restriction == 'FractionalXYZ': 
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
            atoms.info['copt'] = constrained_indices

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

def create_copt(atoms):
    """"""
    content = ""

    return content

def create_slurm(directory, partition='k2-medpri', time='24:00:00', ncpus=32):
    """create job script"""
    content = "#!/bin/bash -l \n"
    content += "#SBATCH --partition=%s        # queue\n" %partition 
    content += "#SBATCH --job-name=%s         # Job name\n" %Path(directory).name 
    content += "#SBATCH --time=%s              # Time limit hrs:min:sec\n" %time 
    content += "#SBATCH --nodes=1                    # Number of nodes\n"
    content += "#SBATCH --ntasks=%s                  # Number of cores\n" %ncpus
    content += "#SBATCH --cpus-per-task=1            # Number of cores per MPI task \n"
    content += "#SBATCH --mem-per-cpu=4G             # Number of cores per MPI task \n"
    content += "#SBATCH --output=slurm.o%j           # Standard output and error log\n"
    content += "#SBATCH --error=slurm.e%j            # Standard output and error log\n"
    content += "\n"
    content += "module purge\n"
    content += "module load services/s3cmd\n"
    content += "module load libs/intel/2016u1\n"
    content += "module load libs/intel-mkl/2016u1/bin\n"
    content += "module load mpi/intel-mpi/2016u1/bin\n"
    content += "\n"
    content += 'echo `date "+%Y-%m-%d %H:%M:%S"` `pwd` >> $HOME/submitted\n'
    content += "mpirun -n %s /mnt/scratch2/chemistry-apps/dkb01416/vasp/installed/intel-2016/5.4.1-TS/vasp_std 2>&1 > vasp.out\n" %str(ncpus)
    content += 'echo `date "+%Y-%m-%d %H:%M:%S"` `pwd` >> $HOME/finished\n'

    with open(os.path.join(directory,'vasp.slurm'), 'w') as fopen:
        fopen.write(content)

    return 

def create_by_ase(atoms, incar=None, directory=Path('vasp-test')):
    # ===== environs
    # pseudo 
    pp_path = "/mnt/scratch2/chemistry-apps/dkb01416/vasp/PseudoPotential"
    if 'VASP_PP_PATH' in os.environ.keys():
        os.environ.pop('VASP_PP_PATH')
    os.environ['VASP_PP_PATH'] = pp_path
    
    # vdw 
    vdw_envname = 'ASE_VASP_VDW'
    vdw_path = "/mnt/scratch2/chemistry-apps/dkb01416/vasp/pot"
    if vdw_envname in os.environ.keys():
        _ = os.environ.pop(vdw_envname)
    os.environ[vdw_envname] = vdw_path

    # ===== set basic params
    vasp_creator = GenerateVaspInput()
    vasp_creator.set_xc_params('PBE') # since incar may not set GGA
    if incar is not None:
        vasp_creator.read_incar(incar)
    vasp_creator.input_params['gamma'] = True
    vasp_creator.input_params['kpts'] = (2,3,1)
    vasp_creator.string_params['system'] = directory.name

    # write inputs
    if not directory.exists():
        directory.mkdir()
    else:
        warnings.warn('%s exists, skip creating...' %directory)
    vasp_creator.initialize(atoms)
    vasp_creator.write_input(atoms, directory)

    # write job script
    # TODO: replace by SlurmMachine
    create_slurm(directory, partition="k2-medpri", time="12:00:00")

    return


def create_vasp_inputs(atoms, incar=None, directory=Path('vasp-test'), verbose=True): 
    """only for molecules"""
    # pseudo 
    pp_path = "/mnt/scratch/chemistry-apps/dkb01416/vasp/PseudoPotential"
    if 'VASP_PP_PATH' in os.environ.keys():
        os.environ.pop('VASP_PP_PATH')
    os.environ['VASP_PP_PATH'] = pp_path

    # vdw 
    vdw_envname = 'ASE_VASP_VDW'
    vdw_path = "/mnt/scratch/chemistry-apps/dkb01416/vasp/pot"
    if vdw_envname in os.environ.keys():
        _ = os.environ.pop(vdw_envname)
    os.environ[vdw_envname] = vdw_path

    # ===== initialise ===== 
    calc = Vasp2(
        command = "vasp_std", 
        directory = 'dummy',  
        **vasp_params, 
    )

    # convert to pathlib usage 
    if directory != Path.cwd() and not directory.exists():
        directory.mkdir()

    calc.initialize(atoms)

    calc.string_params['system'] = Path(directory).name

    if verbose: 
        content = '\n>>>>> Modified ASE for VASP <<<<<\n'
        content += '    directory -> %s\n' %directory 
        print(content)

    # ===== POSCAR ===== 
    poscar = os.path.join(directory, 'POSCAR')
    write_vasp(poscar, atoms, direct=True, vasp5=True)

    from collections import Counter 
    cons = atoms.constraints
    assert len(cons) == 1 
    cons_indices = cons[0].get_indices() 
    symbols = atoms.get_chemical_symbols() 
    all_atoms = Counter(symbols) 
    fixed_symbols = [symbols[i] for i in cons_indices]
    fixed_atoms = Counter(fixed_symbols)

    natoms = len(atoms) 
    nfixed = np.sum(list(fixed_atoms.values()))

    if verbose:
        content = '\nPOSCAR -->\n'
        content += '    \033[4m Element       Numbers      \033[0m\n'
        for sym in all_atoms.keys():
            content += '         %2s \033[1;33m%4d\033[0m \033[32m%4d\033[0m(T)\033[31m%4d\033[0m(F)\n'\
                    %(sym, all_atoms[sym], all_atoms[sym]-fixed_atoms[sym], fixed_atoms[sym])
        content += '    \033[4m                            \033[0m\n'
        content += '      %2s \033[1;33m%4d\033[0m \033[32m%4d\033[0m(T)\033[31m%4d\033[0m(F)\n'\
                    %('Total', natoms, natoms-nfixed, nfixed)

        print(content) 

    # ===== KPOINTS =====
    kpts = np.linalg.norm(atoms.cell, axis=1).tolist()
    kpts = [int(20./k)+1 for k in kpts] 

    calc.input_params['gamma'] = True
    calc.input_params['kpts'] = kpts 
    calc.write_kpoints(atoms=atoms, directory=directory)

    if verbose:
        content = 'KPOINTS -->\n'
        content += "     Set k-point -> \033[1;35m{} {} {}\033[0m\n".format(*kpts)
        print(content)

    # ===== INCAR and 188 ===== 
    content = 'INCAR -->\n'

    if incar: 
        calc.read_incar(incar)
        calc.string_params['system'] = Path(directory).name
        content += '    read template from %s\n' %incar 

    copt = atoms.info.get('copt', None)
    if copt: 
        # If constrained get distance and create fort.188
        symbols = atoms.get_chemical_symbols() 
        ca, cb = atoms.info['copt'][0], atoms.info['copt'][1] 
        pt1, pt2 = atoms.positions[ca], atoms.positions[cb] 

        # Use Ax=b convert to cartisan coordinate
        distance = np.linalg.norm(pt1-pt2)

        # Create fort.188
        ts_content = '1\n3\n6\n4\n0.04\n%-5d%-5d%f\n0\n' % \
            (ca+1, cb+1, distance)

        with open(os.path.join(directory,'fort.188'), 'w') as f:
            f.write(ts_content)
    
        calc.int_params['ibrion'] = 1 

        content += '\n'
        content += "     fort.188 has been created.\n"
        content += '     ' + '-'*20 + '\n'
        content += "     atom number: {:<5d}{:<5d}\n".format(ca+1, cb+1)
        content += "     atom name: {} {}\n".format(symbols[ca], symbols[cb])
        content += "     distance: {:f}\n".format(distance)
        content += '     ' + '-'*20 + '\n'

        # Set IBRION = 1
        content += '     Note: IBRION has been set to 1.\n'

    if verbose:
        print(content)

    calc.write_incar(atoms, directory=directory)
    
    # ===== POTCAR =====
    calc.write_potcar(directory=directory)
    content = "POTCAR -->\n"
    content += "    Using POTCAR from %s\n" %pp_path
    if verbose:
        print(content)

    # ===== VDW =====
    calc.copy_vdw_kernel(directory=directory)

    content = "VDW -->\n"
    content += "    Using vdW Functional as %s\n" %calc.string_params['gga']
    if verbose:
        print(content)

    # ===== SLURM =====
    create_slurm(directory)

    content = "JOB -->\n"
    content += "    Write to %s\n" %str('vasp.slurm')
    if verbose:
        print(content)

    return 


if __name__ == '__main__': 
    # Set argument parser.
    parser = argparse.ArgumentParser()

    # Add optional argument.
    #parser.add_argument("-t", "--task", nargs='?', const='SC', help="set task type")
    #parser.add_argument("-k", "--kpoints", help="set k-points")
    #parser.add_argument("-q", "--queue", choices=('Gold', 'Test'), help="pbs queue type")
    #parser.add_argument("-n", "--ncpu", help="cpu number in total")

    parser.add_argument(
        "-d", "--cwd", default="./", 
        help="current working directory"
    )
    parser.add_argument(
        "-f", "--file", help="structure file in any format"
    )
    parser.add_argument(
        "-i", "--incar", 
        help="template incar file"
    )
    parser.add_argument(
        # "-c", "--copt", action='store_true', 
        "-c", "--copt", type=int, nargs=2, 
        help="use constrained optimisation for transition state search"
    )
    parser.add_argument(
        "-s", "--sort", action='store_false', 
        help="sort atoms by elemental numbers and z-positions"
    )
    parser.add_argument(
        "--sub", action='store_true', 
        help="submit the job after creating input files"
    )

    args = parser.parse_args()

    # Add all possible arguments in INCAR file.
    struct_path = Path(args.file)
    #if struct_path.suffix != '.xsd': 
    #    raise ValueError('only support xsd format now...')

    cwd = Path(args.cwd) 
    if cwd != Path.cwd(): 
        directory = cwd.resolve() / struct_path.stem
    else:
        directory = Path.cwd().resolve() / struct_path.stem
    
    if directory.exists(): 
        raise ValueError('targeted directory exists.')

    atoms = read_xsd2(struct_path)
    #atoms = read(struct_path)
    #atoms.set_constraint(FixAtoms(indices=range(len(atoms))))

    # sort atoms by symbols and z-positions especially for supercells 
    if args.sort: 
        numbers = atoms.numbers 
        zposes = atoms.positions[:,2].tolist()
        sorted_indices = np.lexsort((zposes,numbers))
        atoms = atoms[sorted_indices]
        
        map_indices = dict(zip(sorted_indices, range(len(atoms))))
        copt = atoms.info.get('copt', None)
        if copt: 
            new_copt = [map_indices[key] for key in atoms.info['copt']]
            atoms.info['copt'] = new_copt 
    
    create_vasp_inputs(atoms, incar=args.incar, directory=directory)

    # submit job automatically 
    if args.sub: 
        command = 'sbatch vasp.slurm'
        proc = subprocess.Popen(
            command, shell=True, cwd=directory,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding = 'utf-8'
        )
        errorcode = proc.wait(timeout=120) # 10 seconds
        if errorcode:
            raise ValueError('Error in generating random cells.')

        print(''.join(proc.stdout.readlines()))

    pass 

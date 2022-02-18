#!/usr/bin/env python3 
import os
import time
import json
import logging
import argparse
import subprocess 

import shutil 

from collections import Counter 

import xml.etree.ElementTree as ET
from xml.dom import minidom 

from pathlib import Path 

import numpy as np 

from ase import Atoms 
from ase.io import read, write
from ase.io.vasp import write_vasp 
from ase.calculators.vasp import Vasp
#from ase.calculators.vasp import Vasp2
from ase.constraints import FixAtoms

from GDPy.utils.vasp.main import read_xsd2


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




def create_copt(atoms):
    """"""

    return content

def create_slurm(directory, partition='k2-hipri', time='3:00:00', ncpus='32'):
    """create job script"""
    content = "#!/bin/bash -l \n"
    content += "#SBATCH --partition=%s        # queue\n" %partition 
    content += "#SBATCH --job-name=%s         # Job name\n" %Path(directory).name 
    content += "#SBATCH --time=%s              # Time limit hrs:min:sec\n" %time 
    content += "#SBATCH --nodes=1                    # Number of nodes\n"
    content += "#SBATCH --ntasks=%s                  # Number of cores\n" %ncpus
    content += "#SBATCH --cpus-per-task=1            # Number of cores per MPI task \n"
    content += "#SBATCH --mem=10G                    # Number of cores per MPI task \n"
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
    content += "mpirun -n 32 /mnt/scratch/chemistry-apps/dkb01416/vasp/installed/intel-2016/5.4.1-TS/vasp_std 2>&1 > vasp.out\n"
    content += 'echo `date "+%Y-%m-%d %H:%M:%S"` `pwd` >> $HOME/finished\n'

    with open(os.path.join(directory,'vasp.slurm'), 'w') as fopen:
        fopen.write(content)

    return 


def create_vasp_inputs(atoms, incar=None, directory='vasp-test'): 
    """only for molecules"""
    # pseudo 
    #pp_path = "/mnt/scratch/chemistry-apps/dkb01416/vasp/PseudoPotential"
    pp_path = "/mnt/scratch2/users/40247882/bak.dkb01416/vasp/PseudoPotential"
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
    calc = Vasp(
        command = "vasp_std", 
        directory = 'dummy',  
        **vasp_params, 
    )

    # convert to pathlib usage 
    if directory != Path.cwd() and not directory.exists():
        directory.mkdir()

    calc.initialize(atoms)

    calc.string_params['system'] = Path(directory).name

    content = '\n>>>>> Modified ASE for VASP <<<<<\n'
    content += '    directory -> %s\n' %directory 
    print(content)

    # ===== POSCAR ===== 
    poscar = os.path.join(directory, 'POSCAR')
    write_vasp(poscar, atoms, direct=True, vasp5=True)

    symbols = atoms.get_chemical_symbols() 
    all_atoms = Counter(symbols) 
    cons = atoms.constraints
    print(cons)
    if len(cons) == 1:
        cons_indices = cons[0].get_indices() 
        fixed_symbols = [symbols[i] for i in cons_indices]
        fixed_atoms = Counter(fixed_symbols)
    else:
        fixed_atoms = all_atoms.copy()
        for key in fixed_atoms.keys():
            fixed_atoms[key] = 0

    natoms = len(atoms) 
    nfixed = np.sum(list(fixed_atoms.values()))

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

    print(content)

    calc.write_incar(atoms, directory=directory)
    
    # ===== POTCAR =====
    calc.write_potcar(directory=directory)
    content = "POTCAR -->\n"
    content += "    Using POTCAR from %s\n" %pp_path
    print(content)

    # ===== VDW =====
    calc.copy_vdw_kernel(directory=directory)

    content = "VDW -->\n"
    content += "    Using vdW Functional as %s\n" %calc.string_params['gga']
    print(content)

    # ===== SLURM =====
    create_slurm(directory)

    content = "JOB -->\n"
    content += "    Write to %s\n" %str('vasp.slurm')
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-f", "--file", default=None,
        help="structure file in any format"
    )
    group.add_argument(
        "-sd", "--stru_dir", default=None,
        help="structure file in any format"
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

    stru_files = []
    if args.stru_dir is not None:
        stru_dir = Path(args.stru_dir)
        for p in stru_dir.glob("*.xsd"):
            stru_files.append(p)
        stru_files.sort()
    if args.file is not None:
        # Add all possible arguments in INCAR file.
        struct_path = Path(args.file)
        stru_files.append(struct_path)
        #if struct_path.suffix != '.xsd': 
        #    raise ValueError('only support xsd format now...')

    cwd = Path(args.cwd) 
    for struct_path in stru_files:
        if cwd != Path.cwd(): 
            directory = cwd.resolve() / struct_path.stem
        else:
            directory = Path.cwd().resolve() / struct_path.stem
        
        if directory.exists(): 
            raise ValueError('targeted directory exists.')

        atoms = read_xsd2(struct_path)
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
                raise ValueError('Error in submitting jobs...')

            print(''.join(proc.stdout.readlines()))


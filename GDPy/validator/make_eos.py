#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

import argparse
from pathlib import Path 

import numpy as np 

from ase.io import read, write 
from ase.io.vasp import write_vasp
from ase.calculators.vasp import Vasp2
from ase.constraints import FixAtoms

def check_convergence(atoms, quantity='force'):
    """ase ignores forces of fixed atoms"""
    if quantity == 'force':
        forces = atoms.get_forces() 
        #print(forces)
        max_force = np.max(np.fabs(forces)) 
        print('max_force', max_force)
        if max_force < 0.05:
            return True 
        else:
            return False
    elif quantity == 'stress':
        stress = atoms.get_stress()
        max_stress = np.max(np.fabs(stress))
        print('max_stress', max_stress)
        if max_stress < 1e-3:
            return True 
        else:
            return False
    else:
        raise ValueError('wrong quantity')

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

def submit_slurm(directory):
    # submit job automatically 
    command = 'sbatch vasp.slurm'
    proc = subprocess.Popen(
        command, shell=True, cwd=directory,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding = 'utf-8'
    )
    print(''.join(proc.stdout.readlines()))

    errorcode = proc.wait(timeout=120) # 10 seconds
    if errorcode:
        raise ValueError('Error in submitting the job.')

    pass 

def create_slurm(directory, partition='k2-hipri', time='3:00:00', ncpus='32'):
    """create job script"""
    content = "#!/bin/bash -l \n"
    content += "#SBATCH --partition=%s        # queue\n" %partition 
    content += "#SBATCH --job-name=%s         # Job name\n" %Path(directory).name 
    content += "#SBATCH --time=%s              # Time limit hrs:min:sec\n" %time 
    content += "#SBATCH --nodes=1-2                  # Number of nodes\n"
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


def find_bulks(directory, incar_template):
    # find vasp directories 
    vasp_dirs = []
    cwd = Path(directory) 
    for p in cwd.glob('*'): 
        incar = p / 'INCAR'
        vasp_xml = p / 'vasprun.xml'
        if p.is_dir() and incar.exists() and vasp_xml.exists():
            vasp_dirs.append(p)

    # 
    scalings = [0.80, 0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20, 1.25]

    # vasp working directory 
    for vwd in vasp_dirs:
        result_dir = cwd / (str(vwd) + '_eos')
        if not result_dir.exists():
            result_dir.mkdir()
        else:
            continue
        print('cur dir: %s' %vwd)
        vasp_xml = vwd / 'vasprun.xml'
        atoms = read(vasp_xml, format='vasp-xml')
        if check_convergence(atoms, 'stress'): 
            print('energy: %f' %atoms.get_potential_energy())
            prim_cell = atoms.cell.copy() 
            cons = FixAtoms(indices=range(len(atoms)))
            for scale in scalings:
                scale_dir = result_dir / (str(int(scale*1000)).zfill(4))
                if not scale_dir.exists():
                    scale_dir.mkdir()
                print(scale_dir)
                scaled_atoms = atoms.copy()
                scaled_atoms.set_cell(prim_cell*scale, scale_atoms=True)
                scaled_atoms.set_constraint(cons)
                create_vasp_inputs(scaled_atoms, incar=incar_template, directory=scale_dir)
                submit_slurm(scale_dir)
        else:
            print('not converged!!!')
    pass 


if __name__ == '__main__': 
    # Set argument parser.
    parser = argparse.ArgumentParser()

    # Add optional argument.
    parser.add_argument("-t", "--task", nargs='?', const='SC', help="set task type")
    parser.add_argument("-k", "--kpoints", help="set k-points")
    parser.add_argument("-q", "--queue", choices=('Gold', 'Test'), help="pbs queue type")
    parser.add_argument("-n", "--ncpu", help="cpu number in total")

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

    args = parser.parse_args()

    find_bulks(args.cwd, args.incar)

    pass 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
structure file + vasp script + job script
"""

import argparse
import warnings
import shutil
from pathlib import Path

from ase.io import read, write

def current_slurm(name, stru_path, indices=':'):
    """"""
    content = "#!/bin/bash -l\n"
    content += "#SBATCH --partition=k2-medpri        \n"
    content += "#SBATCH --job-name=%s       \n" %name
    content += "#SBATCH --nodes=1           \n"
    content += "#SBATCH --ntasks=32         \n"
    content += "#SBATCH --time=24:00:00     \n"
    content += "#SBATCH --output=slurm.o%j  \n"
    content += "#SBATCH --error=slurm.e%j   \n"
    content += "\n"
    content += "source ~/envs/source_intel-2016.sh\n"
    content += "conda activate dpdev\n"
    content += "\n"
    content += "export VASP_PP_PATH=\"/mnt/scratch/chemistry-apps/dkb01416/vasp/PseudoPotential\"\n"
    content += "export ASE_VASP_VDW=\"/mnt/scratch/chemistry-apps/dkb01416/vasp/pot\"\n"
    content += "export VASP_COMMAND=\"mpirun -n 32 /mnt/scratch/chemistry-apps/dkb01416/vasp/installed/intel-2016/5.4.1-TS/vasp_std 2>&1 > vasp.out\"\n"
    content += "\n"
    content += "echo `date \"+%Y-%m-%d %H:%M:%S\"` `pwd` >> $HOME/submitted\n"
    content += "./compute_by_ase.py -p ./INCAR_OXIDE -s %s -i %s\n" %(stru_path, indices)
    content += "echo `date \"+%Y-%m-%d %H:%M:%S\"` `pwd` >> $HOME/finished\n"

    return content

def create_chunks(tot, n):
    size = int(tot/n)
    chunks = []
    for i in range(n-1):
        chunks.append("\"%d:%d\"" %(i*size,(i+1)*size))
    chunks.append("\"%d:\""%((n-1)*size))

    return chunks

def create_files(vasp_script, incar_template, stru_file, nchunk=1):
    #vasp_script = Path('../calc_test/compute_by_ase.py')
    #incar_template = Path('../calc_temp/INCAR_OXIDE')
    #stru_file = Path('/users/40247882/projects/oxides/gdp-main/it-0009/dpmd-nvt-oxides/PtO_100-nvt/sorted/PtO_100-nvt-sel.xyz')
    vasp_script = Path(vasp_script)
    incar_template = Path(incar_template)
    stru_file = Path(stru_file)

    frames = read(stru_file, ':')
    print('numebr of frames in %s is %d' %(stru_file.name, len(frames)))
    chunks = create_chunks(len(frames), nchunk)

    for i, chunk in enumerate(chunks):
        wd = Path(stru_file.stem + '-' + str(i))
        print('try to create %s' %wd)
        if wd.exists():
            warnings.warn('%s exists and skip it' %wd, UserWarning)
            continue
        else:
            wd.mkdir()
        # copy files
        shutil.copy(stru_file, wd / stru_file.name)
        shutil.copy(incar_template, wd / incar_template.name)
        shutil.copy(vasp_script, wd / vasp_script.name)
        # write job script
        content = current_slurm(wd.name, stru_file.name, chunk)
        with open(wd / 'vasp.slurm', 'w') as fopen:
            fopen.write(content)

    return


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--structure', 
        default='example.xyz', 
        help='input structures stored in xyz format file'
    )
    parser.add_argument(
        '-v', '--vasp', 
        default='compute_by_ase.py', 
        help='input structures stored in xyz format file'
    )
    parser.add_argument(
        '-p', '--parameter', 
        default='INCAR',
        help='incar template in vasp incar format'
    )
    parser.add_argument(
        '-n', '--nchunks', 
        default=1, type=int,
        help='unsupported frame selection'
    )

    args = parser.parse_args()

    create_files(args.vasp, args.parameter, args.structure, args.nchunks)

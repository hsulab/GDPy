#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pathlib
import shutil
import time
import subprocess
import warnings

import numpy as np

from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from numpy.core.numeric import outer

from GDPy.ga.make_all_vasp import create_by_ase


def check_convergence(atoms, fmax=0.05):
    """Check the convergence of the trajectory"""

    forces = atoms.get_forces()

    max_force = np.max(np.fabs(forces))

    converged = False
    if max_force < fmax:
        converged = True 

    return converged

# vasp utils
def read_sort(directory):
    """Create the sorting and resorting list from ase-sort.dat.
    If the ase-sort.dat file does not exist, the sorting is redone.
    """
    sortfile = directory / 'ase-sort.dat'
    if os.path.isfile(sortfile):
        sort = []
        resort = []
        with open(sortfile, 'r') as fd:
            for line in fd:
                s, rs = line.split()
                sort.append(int(s))
                resort.append(int(rs))
    else:
        # warnings.warn(UserWarning, 'no ase-sort.dat')
        raise ValueError('no ase-sort.dat')

    return sort, resort

class SlurmQueueRun:

    prefix = 'cand'

    def __init__(
            self, data_connection, tmp_folder, 
            n_simul, incar, prefix,
            qsub_command='sbatch', qstat_command='squeue',
            find_neighbors=None, perform_parametrization=None
        ):
        self.dc = data_connection
        self.n_simul = n_simul
        self.incar = incar
        self.prefix = prefix
        self.machine_prefix = prefix[:6] # only use the first six chars
        self.qsub_command = qsub_command
        self.qstat_command = qstat_command
        self.tmp_folder = pathlib.Path(tmp_folder)
        self.find_neighbors = find_neighbors
        self.perform_parametrization = perform_parametrization
        self.__cleanup__()

    def relax(self, a):
        """ Add a structure to the queue. This method does not fail
            if sufficient jobs are already running, but simply
            submits the job. """
        self.__cleanup__()
        self.dc.mark_as_queued(a) # this marks relaxation is in the queue
        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)
        
        # create atom structure
        dname = pathlib.Path('{0}/{1}{2}'.format(self.tmp_folder, self.prefix, a.info['confid']))
        create_by_ase(a, self.incar, dname)

        # write job script
        #job_name = '{0}_{1}'.format(self.job_prefix, a.info['confid'])
        #with open('tmp_job_file.job', 'w') as fopen:
        #    fopen.write(self.job_template_generator(job_name, fname))
        #    fopen.close()
        
        # submit job
        print(self.__submit_job(dname))

        return
    
    def __submit_job(self, dname, job_script='vasp.slurm'):
        # submit job
        command = '{0} {1}'.format(self.qsub_command, job_script)
        proc = subprocess.Popen(
            command, shell=True, cwd=dname,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding = 'utf-8'
        )
        errorcode = proc.wait(timeout=10) # 10 seconds
        if errorcode:
            raise ValueError('Error in generating random cells.')

        output = ''.join(proc.stdout.readlines())

        return output

    def enough_jobs_running(self):
        """ Determines if sufficient jobs are running. """
        return self.number_of_jobs_running() >= self.n_simul

    def number_of_jobs_running(self):
        return len(self.__get_running_job_ids())

    def __get_running_job_ids(self):
        """ Determines how many jobs are running. The user
            should use this or the enough_jobs_running method
            to verify that a job needs to be started before
            calling the relax method."""
        # self.__cleanup__()
        # TODO: reform
        p = subprocess.Popen(
            ['`which {0}` -u `whoami` --format=\"%.12i %.12P %.24j %.4t %.12M %.12L %.5D %.4C\"'.format(self.qstat_command)],
            shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            close_fds=True, universal_newlines=True
        )
        fout = p.stdout
        lines = fout.readlines()
        # print(''.join(lines)) # output of squeue
        confids = []
        for line in lines[1:]: # skipe first info line
            data = line.strip().split()
            jobid, name, status = data[0], data[2], data[3]
            if name.startswith(self.prefix) and status in ['R','Q','PD']:
                #print(jobid)
                #print(name)
                confid = name.strip(self.prefix)
                #print(confid)
                confids.append(int(confid))
        #print(len(confids))

        return confids
    
    def check_status(self):
        """
        check job status
        """
        print('\n===== check job status =====')
        running_ids = self.__get_running_job_ids()
        #exit()
        confs = self.dc.get_all_candidates_in_queue() # conf ids
        for confid in confs:
            if confid in running_ids:
                continue
            cand_dir = self.tmp_folder / (self.prefix+str(confid))
            print(cand_dir)
            # TODO: this only works for vasp, should be more general
            vasprun = cand_dir / 'vasprun.xml'
            if vasprun.exists() and vasprun.stat().st_size > 0:
                frames = read(vasprun, ':')
                print('nframes: ', len(frames))
                atoms_sorted = frames[-1]
                # resort
                sort, resort = read_sort(cand_dir)
                atoms = atoms_sorted.copy()[resort]
                calc = SinglePointCalculator(
                    atoms,
                    energy=atoms_sorted.get_potential_energy(),
                    forces=atoms_sorted.get_forces(apply_constraint=False)[resort]
                )
                calc.name = 'vasp'
                atoms.calc = calc

                # check forces
                if check_convergence(atoms):
                    print('save this configuration to the database')
                    atoms.info['confid'] = confid
                    # add few information
                    atoms.info['data'] = {}
                    atoms.info['key_value_pairs'] = {'extinct': 0}
                    atoms.info['key_value_pairs']['raw_score'] = -atoms.get_potential_energy()
                    self.dc.add_relaxed_step(
                        atoms,
                        find_neighbors=self.find_neighbors,
                        perform_parametrization=self.perform_parametrization
                    )
                else:
                    # TODO: save relaxation trajectory for MLP?
                    # still leave queued=1 in the db, but resubmit
                    resubmit = True
                    if resubmit:
                        print('copy data...')
                        saved_cards = ['POSCAR', 'CONTCAR', 'OUTCAR']
                        for card in saved_cards:
                            card_path = cand_dir / card
                            saved_card_path = cand_dir / (card+'_old')
                            shutil.copy(card_path, saved_card_path)
                        shutil.copy(cand_dir / 'CONTCAR', cand_dir / 'POSCAR')
                        print('resubmit job...')
                        print(self.__submit_job(cand_dir))
                    else:
                        pass
            else:
                print('the job isnot running...')

        return
    
    def __cleanup__(self):
        """
        """
        # self.check_status()

        return

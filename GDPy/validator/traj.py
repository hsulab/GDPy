#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np

from ase.io import read, write

from GDPy.validator.validator import AbstractValidator
from GDPy.potential.register import create_potter, PotentialRegister


"""Benchmark minimisation trajectories...

TODO: MD trajectories as well? Use same random-seed for velocities and see when 
the trajectories diverge...

"""


class TrajValidator(AbstractValidator):

    def run(self):
        """"""
        params = self.task_params

        # - create a target driver
        driver_params = params.get("driver")
        pot_name = driver_params.get("backend", None)
        assert pot_name is not None, "Need a specific backend for the driver."

        manager = PotentialRegister()
        potter = manager.create_potential(pot_name=pot_name)
        potter.register_calculator({})
        potter.version = "unknown"
        target_driver = potter.create_driver(driver_params)

        # - run over wdirs
        wdir = pathlib.Path(params.get("wdir"))
        print(wdir)

        init_stru_fpath = params.get("init")
        init_frames = read(init_stru_fpath, ":")

        info_data = []
        wdir_names = []
        ref_energies, init_energies, end_energies = [], [], []
        for atoms in init_frames:
            cur_wdir = wdir/atoms.info["wdir"]
            wdir_names.append(atoms.info["wdir"])
            # -- input info
            ref_ene = atoms.get_potential_energy() # energy by input structure
            ref_maxfrc = np.max(np.fabs(atoms.get_forces(apply_constraint=True))) # NOTE: may not have constraint
            ref_energies.append(ref_ene)
            # -- read trajectory
            target_driver.directory = cur_wdir
            traj_frames = target_driver.read_trajectory()
            nframes_traj = len(traj_frames)
            # -- 
            init_ene = traj_frames[0].get_potential_energy()
            init_maxfrc = np.max(np.fabs(traj_frames[0].get_forces(apply_constraint=True)))
            init_energies.append(init_ene)
            end_ene = traj_frames[-1].get_potential_energy()
            end_maxfrc = np.max(np.fabs(traj_frames[-1].get_forces(apply_constraint=True)))
            end_energies.append(end_ene)
            pos_rmse = np.sqrt(np.mean(np.square(traj_frames[0].get_positions() - traj_frames[-1].get_positions())))
            info_data.append(
                [
                    atoms.info["wdir"], nframes_traj, ref_ene, ref_maxfrc, init_ene, init_maxfrc, end_ene, end_maxfrc, 
                    init_ene-ref_ene, end_ene-init_ene, pos_rmse
                ]
            )

        # - compare sorted energies
        numbers = list(range(len(wdir_names)))
        ref_numbers = sorted(numbers, key=lambda i: ref_energies[i])
        init_numbers = sorted(numbers, key=lambda i: init_energies[i])
        end_numbers = sorted(numbers, key=lambda i: end_energies[i])
        
        # - info
        #   all data
        #content = ("{:<12s}  "*11+"{:<8s}  "*3+"\n").format(
        #    "wdir", "nframes", "ref [eV]", "ref [eV/Ang]", "ene [eV]", "frc [eV/Ang]", 
        #    "ene [eV]", "frc [eV/Ang]", "init-ref", "end-init", "pos", "ref", "init", "end"
        #)
        #for i, data in enumerate(info_data):
        #    content += ("{:<12s}  "+"{:<12d}  "+"{:<12.4f}  "*9+"{:<8d}  "*3+"\n").format(
        #        *data, ref_numbers.index(i), init_numbers.index(i), end_numbers.index(i)
        #    )
        # -- ene data
        ene_content = ("{:<12s}  "*7+"\n").format(
            "#wdir", "nframes", "ref [eV]", "init [eV]", "end [eV]", 'init-ref', "end-init"
        )
        frc_content = ("{:<12s}  "*6+"\n").format(
            "#wdir", "nframes", "ref [eV/Ang]", "init [eV/Ang]", "end [eV/Ang]", "pos"
        )
        sort_content = ("{:<12s}  "*2+"{:<8s}  "*3+"\n").format(
            "#wdir", "nframes", "ref", "init", "end",
        )
        for i, data in enumerate(info_data):
            ene_content += ("{:<12s}  "+"{:<12d}  "+"{:<12.4f}  "*5+"\n").format(
                data[0], data[1], data[2], data[4], data[6], data[8], data[9]
            )
            frc_content += ("{:<12s}  "+"{:<12d}  "+"{:<12.4f}  "*4+"\n").format(
                data[0], data[1], data[3], data[5], data[7], data[10]
            )
            sort_content += ("{:<12s}  "+"{:<12d}  "+"{:<8d}  "*3+"\n").format(
                data[0], data[1], ref_numbers.index(i), init_numbers.index(i), end_numbers.index(i) 
            )

        self.logger.info(ene_content)
        with open(self.directory/"ene.dat", "w") as fopen:
            fopen.write(ene_content)

        self.logger.info(frc_content)
        with open(self.directory/"frc.dat", "w") as fopen:
            fopen.write(frc_content)

        self.logger.info(sort_content)
        with open(self.directory/"sort.dat", "w") as fopen:
            fopen.write(sort_content)
        
        return


if __name__ == "__main__":
    ...
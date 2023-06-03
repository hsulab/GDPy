#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.neb import NEB
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.validator.validator import AbstractValidator

"""Validate reaction pathway using NEB.
"""

def make_clean_atoms(atoms_):
    """Create a clean atoms from the input."""
    atoms = Atoms(
        symbols=atoms_.get_chemical_symbols(),
        positions=atoms_.get_positions().copy(),
        cell=atoms_.get_cell().copy(),
        pbc=copy.deepcopy(atoms_.get_pbc())
    )

    return atoms

class ReactionValidator(AbstractValidator):


    def run(self):
        """"""
        # - parse optimiser, and create workers
        #self.drivers = {}
        #for dyn_method, dyn_dict in self.task_params["driver"].items():
        #    param_dict = dyn_dict.copy()
        #    param_dict.update(dict(method=dyn_method))
        #    cur_driver = self.pm.create_driver(
        #        dyn_params = param_dict
        #    )
        #    self.drivers.update({dyn_method: cur_driver})
        driver = self.pm.create_driver(self.task_params["driver"])

        # --- NEB calculation ---
        pathways = self.task_params.get("pathways", None)
        if pathways is not None:
            pass
        for name, params in pathways.items():
            self.logger.info(f"===== pathway {name} =====")
            self._irun(name, params, driver)

        return
    
    def _irun(self, name: str, params: dict, driver):
        """"""
        # - create cur out
        ipath = self.directory/name
        if not ipath.exists():
            ipath.mkdir()

        # - read reference structures (mep or IS+TS+FS)
        mep_fpath = params.get("mep", None)
        if mep_fpath is not None:
            frames = read(mep_fpath, ":") # NOTE: need at least three structures
            #print(frames)
            nframes = len(frames)
            names = [a.info.get("name", None) for a in frames]
        else:
            # -- read IS+TS+FS
            frames = []
    
        # - check IS and FS
        ref_energies = [a.get_potential_energy() for a in frames]
        ref_en_ini, ref_en_fin = ref_energies[0], ref_energies[-1]
        displacements = self._compute_mep(frames)

        data = np.array(
            [range(len(frames)), displacements, ref_energies, ref_energies-ref_energies[0]]
        ).T
        np.savetxt(
            ipath/"ref-mep.dat", data,
            fmt="%4d  %12.4f  %12.4f  %12.4f", 
            header="{:<3s}  {:<12s}  {:<12s}  {:<12s}".format("N", "Coord", "E", "dE")
        )

        # - re-calc IS and FS
        opt_worker = driver
        opt_worker.directory = ipath / "IS"
        ini_atoms = opt_worker.run(frames[0].copy()) 
        opt_worker.directory = ipath / "FS"
        fin_atoms = opt_worker.run(frames[-1].copy()) 
        #print(ini_atoms.positions)

        mlp_en_ini, mlp_en_fin = ini_atoms.get_potential_energy(), fin_atoms.get_potential_energy()
        self.logger.info(f"ref mlp\nini: {ref_en_ini}  {mlp_en_ini}\nfin: {ref_en_fin} {mlp_en_fin}")

        # - TODO: refine TS?
        #print("check ts")
        #for i, n in enumerate(names):
        #    # NOTE: have multiple TSs along one pathway?
        #    if n == "TS":
        #        transition = frames[i]
        #        en_ts = transition.get_potential_energy()
        #        print(en_ts)
        #        ts_worker = self.drivers["ts"]
        #        ts_worker.directory = ipath / "TS"
        #        new_transition = ts_worker.run(transition.copy()) 
        #        en_ts = new_transition.get_potential_energy()
        #        print(en_ts)

        # - mep
        rxn_params = self.task_params["rxn_params"]
        rxn_method = rxn_params.get("method", "neb") # find mep
        if rxn_method == "neb":
            nimages = rxn_params["nimages"]
            self.logger.info(f"{rxn_method} with {nimages} images.")
            #if nframes == 2: # NOTE: create new structures
            images = [ini_atoms]
            images += [ini_atoms.copy() for i in range(nimages-2)]
            images.append(fin_atoms)
            #else: # NOTE: use reference mep structures
            #    #print("nimages -> nframes: ", nframes)
            #    images = frames.copy()

            # set calculator
            self.calc.reset()
            for atoms in images:
                atoms.calc = self.calc

            self.logger.info("start NEB calculation...")
            neb = NEB(
                images, 
                allow_shared_calculator=True,
                climb = rxn_params.get("climb", False),
                k = rxn_params.get("k", 0.1)
                # dynamic_relaxation = False
            )
            neb.interpolate(apply_constraint=True) # interpolate configurations

            traj_path = str((ipath / f"rxn-{rxn_method}.traj").absolute())
            # TODO: only ASE backend has driver_cls
            qn = driver.driver_cls(neb, logfile="-", trajectory=traj_path)

            steps, fmax = driver.run_params["steps"], driver.run_params["fmax"]
            qn.run(steps=steps, fmax=fmax)

            # recheck energy
            mep_frames = read(traj_path, "-%s:" %nimages)
        else:
            raise NotImplementedError(f"{rxn_method} is not implemented.")

        # - save mlp results
        mep_frames_ = []
        energies, en_stdvars = [], []
        for a_ in mep_frames:
            a = make_clean_atoms(a_)
            self.calc.reset()
            a.calc = self.calc
            energies.append(a.get_potential_energy())
            en_stdvars.append(a.info.get("en_stdvar", 0.0))
            # -- make clean atoms
            spc = SinglePointCalculator(
                a, energy=a.get_potential_energy(), 
                forces=a.get_forces().copy()
            )
            a.calc = spc
            mep_frames_.append(a)
        mep_frames = mep_frames_
        mlp_energies = np.array(energies)
        #print(mlp_energies)
        #print(mep_frames[0].positions)

        write(ipath/"rxn-mep.xyz", mep_frames)

        displacements = self._compute_mep(mep_frames)
        data = np.array(
            [range(len(mep_frames)), displacements, mlp_energies, mlp_energies-mlp_energies[0]]
        ).T
        np.savetxt(
            ipath/"rxn-mep.dat", data,
            fmt="%4d  %12.4f  %12.4f  %12.4f", 
            header="{:<3s}  {:<12s}  {:<12s}  {:<12s}".format("N", "Coord", "E", "dE")
        )

        return

    def _compute_mep(self, frames):
        """"""        
        nframes = len(frames)
        # - calc reaction coordinate and energies
        from ase.geometry import find_mic
        differences = np.zeros(nframes)
        init_pos = frames[0].get_positions()
        for i in range(1,nframes):
            a = frames[i]
            vector = a.get_positions() - init_pos
            vmin, vlen = find_mic(vector, a.get_cell())
            differences[i] = np.linalg.norm(vlen)

        return differences
    
    def analyse(self):

        return


if __name__ == "__main__":
    pass

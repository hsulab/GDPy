#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import copy
import dataclasses
import itertools
import pathlib
import re
import shutil
from typing import Optional, Union, List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic
from ase.constraints import FixAtoms
from ase.neb import interpolate, idpp_interpolate

from .. import parse_constraint_info
from ..reactor import AbstractReactor
from ..utils import plot_bands, plot_mep, compute_rxn_coords


@dataclasses.dataclass
class StringReactorSetting:

    #: Reactor setting.
    backend: str = "external"

    #: Period to save the trajectory.
    dump_period: int = 1

    #: Period to save the restart file.
    ckpt_period: int = 100

    #: Number of images along the pathway.
    nimages: int = 7

    #: Align IS and FS based on the mic.
    mic: bool = True

    #: Optimiser.
    optimiser: str = "bfgs"

    #: Spring constant, eV/Ang2.
    k: float = 5.0

    #: Whether use CI-NEB.
    climb: bool = False

    #: Convergence coordinate RMS.
    rrms: float = 0.5

    #: Convergence coordinate MAX.
    rmax: float = 0.5

    #: Convergence force RMS.
    frms: float = 0.25

    #: Convergence force tolerance.
    fmax: float = 0.05

    #: Maximum number of steps.
    steps: int = 100

    #: FixAtoms.
    constraint: Optional[str] = None

    #: Parameters that are used to update.
    _internals: dict = dataclasses.field(default_factory=dict)

    def get_init_params(self):
        """"""

        return copy.deepcopy(self._internals)

    def get_run_params(self):
        """"""

        raise NotImplementedError(
            f"{self.__class__.__name__} has no function for run params."
        )


class AbstractStringReactor(AbstractReactor):

    name: str = "string"

    traj_name: str = "nebtraj.xyz"

    @AbstractReactor.directory.setter
    def directory(self, directory_: Union[str, pathlib.Path]):
        self._directory = pathlib.Path(directory_)
        # avoid inconsistent in ASE
        # the actual calc directory will be set in _irun
        self.calc.directory = str(self.directory)

        return

    def run(self, structures: List[Atoms], read_ckpt=True, *args, **kwargs):
        """"""
        super().run(structures=structures, *args, **kwargs)

        # - compatibility
        read_cache = kwargs.get("read_cache", None)
        if read_cache is not None:
            read_ckpt = read_cache

        # - Double-Ended Methods...
        ini_atoms, fin_atoms = structures
        try:
            self._print(f"ini_atoms: {ini_atoms.get_potential_energy()}")
            self._print(f"fin_atoms: {fin_atoms.get_potential_energy()}")
        except RuntimeError:
            # RuntimeError: Atoms object has no calculator.
            self._print("Not energies attached to IS and FS.")

        # - backup old parameters
        prev_params = copy.deepcopy(self.calc.parameters)

        # -
        if not self._verify_checkpoint():
            self._debug(f"... start from the scratch @ {self.directory.name} ...")
            self.directory.mkdir(parents=True, exist_ok=True)
            self._irun([ini_atoms, fin_atoms], *args, **kwargs)
        else:
            self._debug(f"... restart @ {self.directory.name} ...")
            # - check if converged
            converged = self.read_convergence()
            if not converged:
                self._debug(f"... unconverged @ {self.directory.name} ...")
                ckpt_wdir = self._save_checkpoint() if read_ckpt else None
                self._debug(f"... checkpoint @ {str(ckpt_wdir)} ...")
                self._irun(structures, ckpt_wdir=ckpt_wdir, *args, **kwargs)
            else:
                self._debug(f"... converged @ {self.directory.name} ...")

        self.calc.parameters = prev_params
        self.calc.reset()

        # - check again
        curr_band, converged = None, self.read_convergence()
        if converged:
            self._debug(f"... 2. converged @ {self.directory.name} ...")
            band_frames = self.read_trajectory()  # (nbands, nimages)
            if band_frames:
                # FIXME: make below a function
                plot_mep(self.directory, band_frames[-1])
                write(self.directory / "temptraj.xyz", itertools.chain(*band_frames))

                curr_band = band_frames[-1]

                rxn_coords = compute_rxn_coords(curr_band)

                energies = [a.get_potential_energy() for a in curr_band]
                imax = 1 + np.argsort(energies[1:-1])[-1]
                # NOTE: maxforce in cp2k is norm(atomic_forces)
                maxfrc = np.max(curr_band[imax].get_forces(apply_constraint=True))

                self._print(
                    f"rxncoords: {rxn_coords[0]:.2f} -> {rxn_coords[imax]:.2f} "
                    + f"-> {rxn_coords[-1]:.2f}"
                )
                self._print(
                    f"maxfrc: {maxfrc} Ea_f: {energies[imax]-energies[0]:<8.4f} "
                    + f"dE: {energies[-1]-energies[0]:<8.4f}"
                )
            else:
                self._debug(f"... CANNOT read bands @ {self.directory.name} ...")
                ...
        else:
            self._debug(f"... 2. unconverged @ {self.directory.name} ...")

        return curr_band

    @abc.abstractmethod
    def _irun(self, structures: List[Atoms], *args, **kwargs):
        """"""

        return

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """Check if the current directory has any valid outputs or it just created
        the input files.

        """

        return self.directory.exists()

    def _save_checkpoint(self, *args, **kwargs):
        """"""
        # - find previous runs...
        prev_wdirs = sorted(self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"))
        self._debug(f"prev_wdirs: {prev_wdirs}")
        curr_index = len(prev_wdirs)

        curr_wdir = self.directory / f"{str(curr_index).zfill(4)}.run"
        self._debug(f"curr_wdir: {curr_wdir}")

        # - backup files
        curr_wdir.mkdir()
        for x in self.directory.iterdir():
            if not re.match(r"[0-9]{4}\.run", x.name):
                # NOTE: default is to move everything to the new folder
                # if x.name in self.saved_fnames:
                #    shutil.move(x, curr_wdir)
                # else:
                #    x.unlink()
                shutil.move(x, curr_wdir)
            else:
                ...

        return curr_wdir

    def _preprocess_constraints(self, atoms, cons_text: str) -> None:
        """"""
        atoms._del_constraints()
        mobile_indices, frozen_indices = parse_constraint_info(
            atoms, cons_text, ignore_ase_constraints=True, ret_text=False
        )
        if frozen_indices:
            atoms.set_constraint(FixAtoms(indices=frozen_indices))

        return

    def _align_structures(
        self, structures: List[Atoms], run_params: dict, *args, **kwargs
    ) -> List[Atoms]:
        """Create a reaction pathway based on two structures.

        Args:
            structures: Two structures - Initial state and Final State.
            run_params: We need thet latest `constraint` information.

        Returns:
            A List of Atoms structures.

        """
        nstructures = len(structures)
        if nstructures == 2:
            # - check lattice consistency
            ini_atoms, fin_atoms = structures
            c1, c2 = ini_atoms.get_cell(complete=True), fin_atoms.get_cell(
                complete=True
            )
            assert np.allclose(c1, c2), "Inconsistent unit cell..."

            # - align structures
            shifts = fin_atoms.get_positions() - ini_atoms.get_positions()
            if self.setting.mic:
                self._print("Align IS and FS based on MIC.")
                curr_vectors, curr_distances = find_mic(shifts, c1, pbc=True)
                self._debug(f"curr_vectors: {curr_vectors}")
                self._print(f"disp: {np.linalg.norm(curr_vectors)}")
                fin_atoms.positions = ini_atoms.get_positions() + curr_vectors
            else:
                self._print(f"disp: {np.linalg.norm(shifts)}")

            cons_text = run_params.pop("constraint", None)
            self._preprocess_constraints(ini_atoms, cons_text)
            self._preprocess_constraints(fin_atoms, cons_text)

            # - find mep
            nimages = self.setting.nimages
            images = [ini_atoms]
            images += [ini_atoms.copy() for i in range(nimages - 2)]
            images.append(fin_atoms)

            interpolate(
                images=images,
                mic=False,
                interpolate_cell=False,
                use_scaled_coord=False,
                apply_constraint=False,
            )
        else:
            self._print("Use a pre-defined pathway.")
            images = [a.copy() for a in structures]

        return images

    def read_trajectory(self, *args, **kwargs):
        """"""
        # - find previous runs...
        prev_wdirs = sorted(self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"))
        self._debug(f"prev_wdirs: {prev_wdirs}")

        traj_list = []
        for w in prev_wdirs:
            curr_frames = self._read_a_single_trajectory(wdir=w)
            traj_list.append(curr_frames)

        cache_nebtraj = self.directory / self.traj_name
        if cache_nebtraj.exists() and cache_nebtraj.stat().st_size != 0:
            traj_list.append(self._read_a_single_trajectory(wdir=self.directory))

        # - concatenate
        traj_frames, ntrajs = [], len(traj_list)
        if ntrajs > 0:
            traj_frames.extend(traj_list[0])
            for i in range(1, ntrajs):
                prev_end_band, curr_beg_band = traj_list[i - 1][-1], traj_list[i][0]
                for j, (a, b) in enumerate(zip(prev_end_band, curr_beg_band)):
                    assert np.allclose(
                        a.positions, b.positions
                    ), f"Traj {i-1} and traj {i} are not consecutive in positions."
                traj_frames.extend(traj_list[i][1:])
        else:
            ...

        if traj_frames:
            # FIXME: make below a function
            plot_mep(self.directory, traj_frames[-1])

            curr_band = traj_frames[-1]

            rxn_coords = compute_rxn_coords(curr_band)

            energies = [a.get_potential_energy() for a in curr_band]
            imax = 1 + np.argsort(energies[1:-1])[-1]
            # NOTE: maxforce in cp2k is norm(atomic_forces)
            maxfrc = np.max(curr_band[imax].get_forces(apply_constraint=True))

            self._print(f"imax: {imax}")
            self._print(
                f"rxncoords: {rxn_coords[0]:.2f} -> {rxn_coords[imax]:.2f} "
                + f"-> {rxn_coords[-1]:.2f}"
            )
            self._print(
                f"maxfrc: {maxfrc} Ea_f: {energies[imax]-energies[0]:<8.4f} "
                + f"dE: {energies[-1]-energies[0]:<8.4f}"
            )

        return traj_frames

    def as_dict(self) -> dict:
        """"""
        params = {}

        # self._print(f"{self.setting.backend = }")

        for k, v in dataclasses.asdict(self.setting).items():
            if not k.startswith("_"):
                params[k] = v

        return params


if __name__ == "__main__":
    ...

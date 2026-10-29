"""Finite-difference vibrational analysis with an ASE calculator."""

import dataclasses
import json
import shutil

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.vibrations import Vibrations

from gdpx.execution.driver import DriverSetting
from gdpx.structures.groups import evaluate_group_expression

from .driver import AseDriver, save_trajectory


@dataclasses.dataclass
class AseFrequencySetting(DriverSetting):
    task: str = "freq"
    group: str | None = None
    indices: list[int] | None = None
    maxstep: float = 0.01
    nfree: int = 2

    def __post_init__(self):
        if self.group is not None and self.indices is not None:
            raise ValueError("Specify either group or indices for freq, not both.")
        if not np.isfinite(self.maxstep) or self.maxstep <= 0:
            raise ValueError("freq maxstep must be finite and positive (angstrom).")
        if self.nfree not in (2, 4):
            raise ValueError("freq nfree must be 2 or 4.")

    def get_run_params(self, **kwargs):
        return {"constraint": kwargs.get("constraint", self.constraint)}


class AseFrequencyDriver(AseDriver):
    """Displace a selected group, retaining its full chemical environment."""

    setting_cls = AseFrequencySetting

    def _verify_checkpoint(self, *args, **kwargs):
        # Dynamics checkpoints are not vibration caches. Explicit reruns start
        # fresh so neither changed structures nor calculators reuse old forces.
        return False

    def read_convergence_from_trajectory(self, frames):
        return bool(frames and frames[-1].info.get("freq_completed", False))

    def _irun(self, atoms, *args, **kwargs):
        trajectory = self.directory / self.xyz_fname
        trajectory.unlink(missing_ok=True)
        result_path = self.directory / "vibrations.json"
        result_path.unlink(missing_ok=True)
        (self.directory / "vibrations.txt").unlink(missing_ok=True)
        setting = self.setting
        self._preprocess_constraints(atoms, setting.get_run_params(**kwargs))
        frozen = set()
        for constraint in atoms.constraints:
            if not isinstance(constraint, FixAtoms):
                raise ValueError("ASE freq supports only FixAtoms constraints.")
            frozen.update(constraint.get_indices())
        explicit = setting.indices is not None or setting.group is not None
        indices = (
            list(setting.indices)
            if setting.indices is not None
            else sorted(evaluate_group_expression(atoms, setting.group))
        )
        if not explicit:
            indices = [i for i in indices if i not in frozen]
        if not indices:
            raise ValueError("freq requires a nonempty group of atoms.")
        if any(
            isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= len(atoms)
            for i in indices
        ):
            raise ValueError("freq indices must be valid zero-based atom indices.")
        if len(set(indices)) != len(indices):
            raise ValueError("freq indices must be unique.")
        if frozen.intersection(indices):
            raise ValueError("freq selected atoms overlap FixAtoms constraints.")

        # A trajectory is written only after every displacement and output has
        # succeeded; workers must not collect a partial spectrum as completed.
        cache = self.directory / "vibrations"
        if cache.exists():
            shutil.rmtree(cache)
        atoms.calc = self.calc
        vib = Vibrations(atoms, indices=indices, name=str(cache), delta=setting.maxstep, nfree=setting.nfree)
        try:
            vib.run()
            data = vib.get_vibrations()
            frequencies = data.get_frequencies()
            energies = data.get_energies()
            result = {
                "indices": indices,
                "symbols": [atoms[i].symbol for i in indices],
                "maxstep_angstrom": setting.maxstep,
                "nfree": setting.nfree,
                "frequencies_cm1": frequencies.real.tolist(),
                "imaginary_frequencies_cm1": frequencies.imag.tolist(),
                "energies_eV": energies.real.tolist(),
                "imaginary_energies_eV": energies.imag.tolist(),
                "zero_point_energy_eV": float(data.get_zero_point_energy()),
                "hessian_eV_angstrom2": data.get_hessian_2d().tolist(),
                "modes": data.get_modes(all_atoms=True).tolist(),
            }
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            vib.summary(log=str(self.directory / "vibrations.txt"))
            atoms.info["step"] = 0
            save_trajectory(atoms, trajectory)
            # Preserve the completion marker in collected and archived frames.
            completed = read(trajectory)
            completed.info["freq_completed"] = True
            write(trajectory, completed)
        except BaseException:
            try:
                self._finalize_calculators()
            except Exception as cleanup_error:
                self._debug(f"Failed to finalize ASE frequency driver: {cleanup_error!r}")
            raise
        else:
            self._finalize_calculators()

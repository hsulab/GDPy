import io
import os
import pathlib
import pickle
import tarfile
from typing import Optional

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, all_changes
from ase.calculators.lammps import Prism, unitconvert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses, atomic_numbers
from ase.io import read
from ase.io.lammpsdata import write_lammps_data

from gdpx import config
from gdpx.backend.lammps import add_model_deviation_to_atoms_info, parse_thermo_data_by_pattern
from gdpx.group import evaluate_constraint_expression, evaluate_group_expression
from gdpx.utils.strconv import integers_to_string

from .constants import ASELMPCONFIG, parse_type_list


def _read_a_single_trajectory(
    wdir: pathlib.Path,
    mdir,
    units: str,
    archive_path: Optional[pathlib.Path] = None,
    print_func=config._print,
    debug_func=config._debug,
) -> list[Atoms]:
    traj_io, log_io = None, None
    if archive_path is None:
        traj_io = open(wdir / ASELMPCONFIG.trajectory_filename)
        log_io = open(wdir / ASELMPCONFIG.log_filename)
        prism_file = wdir / ASELMPCONFIG.prism_filename
        prism_io = open(prism_file, "rb") if prism_file.exists() else None
        devi_path = wdir / ASELMPCONFIG.deviation_filename
        devi_io = open(devi_path) if devi_path.exists() else None
    else:
        rpath = wdir.relative_to(mdir.parent)
        traj_tarname = str(rpath / ASELMPCONFIG.trajectory_filename)
        prism_tarname = str(rpath / ASELMPCONFIG.prism_filename)
        log_tarname = str(rpath / ASELMPCONFIG.log_filename)
        devi_tarname = str(rpath / ASELMPCONFIG.deviation_filename)
        prism_io, devi_io = None, None
        with tarfile.open(archive_path, "r:gz") as tar:
            for tarinfo in tar:
                if tarinfo.name.startswith(wdir.name):
                    if tarinfo.name == traj_tarname:
                        traj_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                    elif tarinfo.name == prism_tarname:
                        prism_io = io.BytesIO(tar.extractfile(tarinfo.name).read())
                    elif tarinfo.name == log_tarname:
                        log_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                    elif tarinfo.name == devi_tarname:
                        devi_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())

    assert traj_io is not None
    assert log_io is not None

    timesteps = []
    while True:
        line = traj_io.readline()
        if "TIMESTEP" in line:
            timesteps.append(int(traj_io.readline().strip()))
        if not line:
            break
    traj_io.seek(0)

    prismobj = pickle.load(prism_io) if prism_io is not None else None

    curr_traj_frames_ = read(
        traj_io,
        index=":",
        format="lammps-dump-text",
        prismobj=prismobj,
        units=units,
    )
    nframes_traj = len(curr_traj_frames_)
    timesteps = timesteps[:nframes_traj]

    thermo_dict = parse_thermo_data_by_pattern(log_io.readlines(), print_func=print_func, debug_func=debug_func)
    pot_energies = [unitconvert.convert(p, "energy", units, "ASE") for p in thermo_dict["PotEng"]]
    nframes_thermo = len(pot_energies)
    nframes = min(nframes_traj, nframes_thermo)
    debug_func(f"nframes in lammps: {nframes} traj {nframes_traj} thermo {nframes_thermo}")

    curr_traj_frames, curr_energies = [], []
    for i, t in enumerate(timesteps):
        if t in thermo_dict["Step"]:
            curr_atoms = curr_traj_frames_[i]
            curr_atoms.info["step"] = t
            curr_traj_frames.append(curr_atoms)
            curr_energies.append(pot_energies[thermo_dict["Step"].tolist().index(t)])

    for pot_eng, atoms in zip(curr_energies, curr_traj_frames):
        forces = atoms.get_forces()
        sp_calc = SinglePointCalculator(atoms, energy=pot_eng, forces=forces)
        atoms.calc = sp_calc

    if devi_io is not None:
        add_model_deviation_to_atoms_info(devi_io, curr_traj_frames, units=units)

    traj_io.close()
    log_io.close()
    if prism_io is not None:
        prism_io.close()
    if devi_io is not None:
        devi_io.close()

    return curr_traj_frames


class Lammps(FileIOCalculator):
    name: str = "Lammps"
    implemented_properties: list[str] = ["energy", "forces", "stress"]

    default_parameters: dict = dict(
        task="min",
        dump_period=1,
        ckpt_period=100,
        dynamics="",
        steps=0,
        constraint=None,
        etol=0.0,
        ftol=0.05,
        is_classic=False,
        read_restart=None,
        units="metal",
        atom_style="atomic",
        atom_modify=None,
        processors=None,
        newton=None,
        pair_style=None,
        pair_coeff=None,
        pair_modify=None,
        kspace_style=None,
        neighbor="2.0 bin",
        neigh_modify="every 10 check yes",
        mass="* 1.0",
        halt="",
        extra_fix=[],
        plumed=None,
    )

    type_list: Optional[list[str]] = None
    cached_traj_frames: Optional[list[Atoms]] = None

    def __init__(self, command="lmp", label=name, **kwargs):
        FileIOCalculator.__init__(self, command=command, label=label, **kwargs)
        command_ = self.profile.command
        if "-in" not in command_:
            command_ += " -in in.lammps 2>&1 > lmp.out"
        self.profile.command = command_
        assert self.pair_style is not None, "pair_style is not set."

    def __getattr__(self, key):
        if key != "parameters" and key in self.parameters:
            return self.parameters[key]
        return object.__getattribute__(self, key)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        assert atoms is not None
        self.type_list = parse_type_list(atoms)
        FileIOCalculator.calculate(self, atoms, properties, system_changes)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        write_velocities = atoms.get_kinetic_energy() > 0.0
        prismobj = Prism(atoms.get_cell())
        prism_file = os.path.join(self.directory, ASELMPCONFIG.prism_filename)
        with open(prism_file, "wb") as fopen:
            pickle.dump(prismobj, fopen)
        stru_data = os.path.join(self.directory, ASELMPCONFIG.inputstructure_filename)
        assert self.type_list is not None
        write_lammps_data(
            stru_data,
            atoms,
            specorder=self.type_list,
            force_skew=False,
            prismobj=prismobj,
            velocities=write_velocities,
            units=self.units,
            atom_style=self.atom_style,
        )
        self._write_input(atoms, prismobj)

    def _is_finished(self):
        is_finished, end_info = False, "not finished"
        log_filepath = pathlib.Path(os.path.join(self.directory, ASELMPCONFIG.log_filename))
        if log_filepath.exists():
            ERR_FLAG = "ERROR: "
            END_FLAG = "Total wall time:"
            with open(log_filepath) as fopen:
                lines = fopen.readlines()
            for line in lines:
                if line.strip().startswith(ERR_FLAG):
                    is_finished = True
                    end_info = " ".join(line.strip().split()[1:])
                    break
                if line.strip().startswith(END_FLAG):
                    is_finished = True
                    end_info = " ".join(line.strip().split()[1:])
                    break
        return is_finished, end_info

    def read_results(self):
        self.results = {}
        curr_wdir = pathlib.Path(self.directory)
        self.cached_traj_frames = _read_a_single_trajectory(
            mdir=curr_wdir,
            wdir=curr_wdir,
            units=self.units,
            print_func=lambda _: "",
            debug_func=lambda _: "",
        )
        converged_frame = self.cached_traj_frames[-1]
        self.results["forces"] = converged_frame.get_forces().copy()
        self.results["energy"] = converged_frame.get_potential_energy()
        for k, v in converged_frame.info.items():
            if "devi" in k:
                self.results[k] = v

    def _write_input(self, atoms, prismobj):
        parts = []

        parts.append(self._write_header(atoms, prismobj))
        parts.append(self._write_masses_and_charges())
        parts.append(self._write_pair_styles())
        parts.append(self._write_neighbor_and_constraints(atoms))
        parts.append(self._write_output_section())
        parts.append(self._write_fixes_section(atoms))
        parts.append(self._write_simulation_tasks())

        content = "\n".join(parts)

        in_file = os.path.join(self.directory, ASELMPCONFIG.input_fname)
        with open(in_file, "w") as fopen:
            fopen.write(content)

    def _write_header(self, atoms, prismobj):
        lines = []
        lines.append(f"restart         {self.ckpt_period}  restart.*.data")
        lines.append("")
        lines.append(f"units           {self.units}")
        lines.append(f"atom_style      {self.atom_style}")
        if self.atom_modify is not None:
            lines.append(f"atom_modify {self.atom_modify}")
        if self.processors is not None:
            lines.append(f"processors  {self.processors}")
        pbc = atoms.get_pbc()
        if "boundary" in self.parameters:
            lines.append("boundary {0}".format(self.parameters["boundary"]))
        else:
            lines.append(
                "boundary {0} {1} {2}".format(
                    *tuple("fp"[int(x)] for x in pbc)
                )
            )
        if self.newton:
            lines.append(f"newton  {self.newton}")
        if self.read_restart is None:
            lines.append(f"read_data	    {ASELMPCONFIG.inputstructure_filename}")
        else:
            lines.append(f"read_restart    {self.read_restart}")
        if prismobj.is_skewed():
            lines.append("box             tilt large")
            lines.append("change_box      all triclinic")
        return "\n".join(lines) + "\n"

    def _write_masses_and_charges(self):
        lines = []
        assert self.type_list is not None
        for idx, elem in enumerate(self.type_list):
            mass_val = atomic_masses[atomic_numbers[elem]]
            lines.append("mass %d %f" % (idx + 1, mass_val))
        lines.append("")
        if self.atom_style == "charge" and self.type_charges:
            for itype, charge in enumerate(self.type_charges):
                lines.append(f"set type {itype + 1} charge {charge}")
            lines.append("")
        return "\n".join(lines)

    def _write_pair_styles(self):
        lines = []
        assert self.type_list is not None
        type_list_str = " ".join(self.type_list)
        pair_dicts = dict(
            pair_style=dict(
                type_list=type_list_str,
                out_freq=self.dump_period,
            ),
            pair_coeff=dict(
                type_list=type_list_str,
            ),
        )
        if self.is_classic:
            lines.append(f"pair_style  {self.pair_style}")
            if isinstance(self.pair_coeff, str):
                pair_coeff = [self.pair_coeff]
            else:
                pair_coeff = self.pair_coeff
            for coeff in pair_coeff:
                lines.append(f"pair_coeff  {coeff}")
        else:
            potential = self.pair_style.strip().split()[0]
            if potential == "reax/c":
                assert self.atom_style == "charge", "reax/c should have charge atom_style"
                lines.append(f"pair_style  {self.pair_style}")
                lines.append(f"pair_coeff  {self.pair_coeff} {type_list_str}")
                lines.append("fix             reaxqeq all qeq/reax 1 0.0 10.0 1e-6 reax/c")
            elif potential == "eann":
                pot_data = self.pair_style.strip().split()[1:]
                endp = len(pot_data)
                for ip, p in enumerate(pot_data):
                    if p == "out_freq":
                        endp = ip
                        break
                pot_data = pot_data[:endp]
                if len(pot_data) > 1:
                    pair_style = "eann {} out_freq {}".format(" ".join(pot_data), self.dump_period)
                else:
                    pair_style = "eann {}".format(" ".join(pot_data))
                lines.append("pair_style  {}".format(pair_style))
                if self.pair_coeff is None:
                    pair_coeff = "double * *"
                else:
                    pair_coeff = self.pair_coeff
                lines.append(f"pair_coeff	{pair_coeff} {type_list_str}")
            else:
                lines.append("pair_style  " + self.pair_style.format(**pair_dicts["pair_style"]))
                lines.append("pair_coeff  " + self.pair_coeff.format(**pair_dicts["pair_coeff"]))
        if self.pair_modify is not None:
            lines.append(f"pair_modify {self.pair_modify}")
        if self.kspace_style is not None:
            lines.append(f"kspace_style  {self.kspace_style}")
        return "\n".join(lines) + "\n\n"

    def _write_neighbor_and_constraints(self, atoms):
        lines = []
        lines.append(f"neighbor        {self.neighbor}")
        if self.neigh_modify:
            lines.append(f"neigh_modify    {self.neigh_modify}")
        lines.append("")
        mobile_indices, frozen_indices = evaluate_constraint_expression(atoms, self.constraint)
        if mobile_indices:
            mobile_text = integers_to_string(mobile_indices, inp_convention="ase")
            lines.append(f"group mobile id {mobile_text}")
            lines.append("")
        if frozen_indices:
            frozen_text = integers_to_string(frozen_indices, inp_convention="ase")
            lines.append(f"group frozen id {frozen_text}")
            lines.append("fix cons frozen setforce 0.0 0.0 0.0")
        lines.append("")
        return "\n".join(lines)

    def _write_output_section(self):
        lines = []
        if self.task == "min":
            lines.append("thermo_style    custom step pe ke etotal temp press vol fmax fnorm")
        elif self.task == "md":
            lines.append("compute mobileTemp mobile temp")
            lines.append("thermo_style    custom step c_mobileTemp pe ke etotal press vol lx ly lz xy xz yz")
        lines.append(f"thermo         {self.dump_period}")
        lines.append("thermo_modify   flush yes")
        if self.atom_style == "atomic":
            lines.append(
                f"dump		1 all custom {self.dump_period} {ASELMPCONFIG.trajectory_filename} id type element x y z fx fy fz vx vy vz"
            )
        elif self.atom_style == "charge":
            lines.append(
                f"dump		1 all custom {self.dump_period} {ASELMPCONFIG.trajectory_filename} id type element q x y z fx fy fz vx vy vz"
            )
        assert self.type_list is not None
        lines.append(f"dump_modify 1 element {' '.join(self.type_list)} flush yes")
        lines.append("")
        return "\n".join(lines)

    def _write_fixes_section(self, atoms):
        lines = []
        for i, fix_info in enumerate(self.extra_fix):
            if isinstance(fix_info, str):
                lines.append("{:<24s}  {:<24s}  {:<s}".format("fix", f"extra{i}", fix_info))
            else:
                group_indices = evaluate_group_expression(atoms, fix_info[0])
                group_text = integers_to_string(group_indices, inp_convention="ase")
                lines.append("{:<24s}  {:<24s}  id  {:<s}".format("group", f"extra_group_{i}", group_text))
                lines.append(
                    "{:<24s}  {:<24s}  {:<s}  {:<s}".format("fix", f"extra{i}", f"extra_group_{i}", fix_info[1])
                )
        lines.append("")
        if self.halt:
            lines.append(self.halt)
            lines.append("")
        return "\n".join(lines)

    def _write_simulation_tasks(self):
        lines = []
        if self.task == "min":
            if isinstance(self.dynamics, list):
                lines.append("\n".join(self.dynamics))
            else:
                lines.append(str(self.dynamics))
            lines.append(
                "minimize        {:f} {:f} {:d} {:d}".format(
                    unitconvert.convert(self.etol, "energy", "ASE", self.units),
                    unitconvert.convert(self.ftol, "force", "ASE", self.units),
                    self.steps,
                    2 * self.steps,
                )
            )
        elif self.task == "md":
            dynamics_lines = list(self.dynamics)
            if self.read_restart is not None:
                dynamics_lines[0] = "#  use velocities in restart"
            lines.append("\n".join(dynamics_lines))
            if self.plumed is not None:
                lines.append("fix             metad all plumed plumedfile plumed.inp outfile plumed.out")
            lines.append(f"run             {self.steps}")
        return "\n".join(lines) + "\n"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
import warnings

from ase import Atoms, units
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.cp2k import InputSection, parse_input

from .parser import read_cp2k_energy_force


class Cp2kFileIO(FileIOCalculator):

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    default_parameters = dict(
        auto_write=False,
        basis_set="DZVP-MOLOPT-SR-GTH",
        basis_set_file="BASIS_MOLOPT",
        pseudo_potential="GTH-PBE",
        potential_file="POTENTIAL",
        inp="",
        force_eval_method="Quickstep",
        charge=0,
        uks=False,
        stress_tensor=False,
        poisson_solver="auto",
        xc="PBE",
        max_scf=50,
        cutoff=400 * units.Rydberg,
        print_level="MEDIUM",
    )

    """This calculator is consistent with v9.1 and v2022.1.
    """

    def __init__(
        self, restart=None, label="cp2k", atoms=None, command="cp2k.psmp", **kwargs
    ):
        """Construct CP2K-calculator object"""
        super().__init__(
            restart=restart, label=label, atoms=atoms, command=command, **kwargs
        )

        # complete command
        command_ = self.profile.command
        if "-i" in command_:
            ...
        else:
            label_name = pathlib.Path(self.label).name
            command_ += f" -i {label_name}.inp -o {label_name}.out"
        self.profile.command = command_

        return

    def read_results(self):
        """"""
        super().read_results()

        label_name = pathlib.Path(self.label).name

        # TODO: stress
        run_type = self.get_run_type().upper()
        if run_type in ["GEO_OPT", "CELL_OPT", "MD"]:
            trajectory = read_cp2k_outputs(self.directory, prefix=label_name)
            atoms = trajectory[-1]
            self.results["energy"] = atoms.get_potential_energy()
            self.results["free_energy"] = atoms.get_potential_energy(
                force_consistent=True
            )
            self.results["forces"] = atoms.get_forces()
        elif run_type in ["ENERGY_FORCE"]:
            atoms = self.atoms
            self.results = read_cp2k_energy_force(self.directory, prefix=label_name)
            assert self.results["energy"] is not None, f"{self.results['energy'] =}"
            assert self.results["forces"].shape[0] == len(
                atoms
            ), f"{self.results['forces'] =}"
        else:
            raise RuntimeError()

        scf_convergence = read_cp2k_convergence(
            pathlib.Path(self.directory) / "cp2k.out"
        )
        atoms.info["scf_convergence"] = scf_convergence
        if not scf_convergence:
            atoms.info["error"] = f"Unconverged SCF at {self.directory}."

        return

    def write_input(self, atoms, properties=None, system_changes=None):
        """"""
        super().write_input(atoms, properties, system_changes)

        # Support mixed basis_set
        prev_basis_set = self.parameters.basis_set
        if isinstance(prev_basis_set, str):
            curr_basis_set = {
                k: prev_basis_set for k in list(set(atoms.get_chemical_symbols()))
            }
        elif isinstance(prev_basis_set, dict):
            for k in list(set(atoms.get_chemical_symbols())):
                if k not in prev_basis_set:
                    raise RuntimeError(f"No basis_set for {k}.")
            curr_basis_set = prev_basis_set
        else:
            raise RuntimeError(f"Unknown basis_set {prev_basis_set}.")
        self.parameters.basis_set = curr_basis_set

        label_name = pathlib.Path(self.label).name
        wdir = pathlib.Path(self.directory)
        with open(wdir / f"{label_name}.inp", "w") as fopen:
            fopen.write(self._generate_input())

        self.parameters.basis_set = prev_basis_set

        return

    def get_run_type(self) -> str:
        """"""
        root = parse_input(self.parameters.inp)

        run_type = ""
        for subsection in root.subsections:
            if subsection.name == "GLOBAL":
                for keyword in subsection.keywords:
                    kw = keyword.strip().split()
                    if kw[0] == "RUN_TYPE":
                        run_type = kw[1]
                        break
                if run_type:
                    break
        else:
            ...

        assert run_type in [
            "ENERGY_FORCE",
            "GEO_OPT",
            "CELL_OPT",
            "MD",
        ], f"Unknown run_type `{run_type}`."

        return run_type

    def _generate_input(self):
        """Generates a CP2K input file"""
        p = self.parameters
        root = parse_input(p.inp)
        label_name = pathlib.Path(self.label).name
        root.add_keyword("GLOBAL", "PROJECT " + label_name)
        if p.print_level:
            root.add_keyword("GLOBAL", "PRINT_LEVEL " + p.print_level)
        # root.add_keyword("GLOBAL", "RUN_TYPE " + "CELL_OPT")
        if p.force_eval_method:
            root.add_keyword("FORCE_EVAL", "METHOD " + p.force_eval_method)
        if p.stress_tensor:
            root.add_keyword("FORCE_EVAL", "STRESS_TENSOR ANALYTICAL")
            root.add_keyword(
                "FORCE_EVAL/PRINT/STRESS_TENSOR", "_SECTION_PARAMETERS_ ON"
            )
        if p.basis_set_file:
            root.add_keyword(
                "FORCE_EVAL/DFT", "BASIS_SET_FILE_NAME " + p.basis_set_file
            )
        if p.potential_file:
            root.add_keyword(
                "FORCE_EVAL/DFT", "POTENTIAL_FILE_NAME " + p.potential_file
            )
        if p.cutoff:
            root.add_keyword("FORCE_EVAL/DFT/MGRID", "CUTOFF [eV] %.18e" % p.cutoff)
        if p.max_scf:
            root.add_keyword("FORCE_EVAL/DFT/SCF", "MAX_SCF %d" % p.max_scf)
            root.add_keyword("FORCE_EVAL/DFT/LS_SCF", "MAX_SCF %d" % p.max_scf)

        if p.xc:
            legacy_libxc = ""
            for functional in p.xc.split():
                functional = functional.replace("LDA", "PADE")  # resolve alias
                xc_sec = root.get_subsection("FORCE_EVAL/DFT/XC/XC_FUNCTIONAL")
                # libxc input section changed over time
                if functional.startswith("XC_") and self._shell.version < 3.0:
                    legacy_libxc += " " + functional  # handled later
                elif functional.startswith("XC_") and self._shell.version < 5.0:
                    s = InputSection(name="LIBXC")
                    s.keywords.append("FUNCTIONAL " + functional)
                    xc_sec.subsections.append(s)
                elif functional.startswith("XC_"):
                    s = InputSection(name=functional[3:])
                    xc_sec.subsections.append(s)
                else:
                    s = InputSection(name=functional.upper())
                    xc_sec.subsections.append(s)
            if legacy_libxc:
                root.add_keyword(
                    "FORCE_EVAL/DFT/XC/XC_FUNCTIONAL/LIBXC",
                    "FUNCTIONAL " + legacy_libxc,
                )

        if p.uks:
            root.add_keyword("FORCE_EVAL/DFT", "UNRESTRICTED_KOHN_SHAM ON")

        if p.charge and p.charge != 0:
            root.add_keyword("FORCE_EVAL/DFT", "CHARGE %d" % p.charge)

        # add Poisson solver if needed
        if p.poisson_solver == "auto" and not any(self.atoms.get_pbc()):
            root.add_keyword("FORCE_EVAL/DFT/POISSON", "PERIODIC NONE")
            root.add_keyword("FORCE_EVAL/DFT/POISSON", "PSOLVER  MT")

        # write coords
        syms = self.atoms.get_chemical_symbols()
        atoms = self.atoms.get_positions()
        for elm, pos in zip(syms, atoms):
            line = "%s %.18e %.18e %.18e" % (elm, pos[0], pos[1], pos[2])
            root.add_keyword("FORCE_EVAL/SUBSYS/COORD", line, unique=False)

        # write cell
        pbc = "".join([a for a, b in zip("XYZ", self.atoms.get_pbc()) if b])
        if len(pbc) == 0:
            pbc = "NONE"
        root.add_keyword("FORCE_EVAL/SUBSYS/CELL", "PERIODIC " + pbc)
        c = self.atoms.get_cell()
        for i, a in enumerate("ABC"):
            line = "%s %.18e %.18e %.18e" % (a, c[i, 0], c[i, 1], c[i, 2])
            root.add_keyword("FORCE_EVAL/SUBSYS/CELL", line)

        # determine pseudo-potential
        potential = p.pseudo_potential
        if p.pseudo_potential == "auto":
            if p.xc and p.xc.upper() in (
                "LDA",
                "PADE",
                "BP",
                "BLYP",
                "PBE",
            ):
                potential = "GTH-" + p.xc.upper()
            else:
                msg = "No matching pseudo potential found, using GTH-PBE"
                warnings.warn(msg, RuntimeWarning)
                potential = "GTH-PBE"  # fall back

        # write atomic kinds
        subsys = root.get_subsection("FORCE_EVAL/SUBSYS").subsections
        kinds = dict([(s.params, s) for s in subsys if s.name == "KIND"])
        for elem in set(self.atoms.get_chemical_symbols()):
            if elem not in kinds.keys():
                s = InputSection(name="KIND", params=elem)
                subsys.append(s)
                kinds[elem] = s
            if p.basis_set:
                kinds[elem].keywords.append("BASIS_SET " + p.basis_set[elem])
            if potential:
                kinds[elem].keywords.append("POTENTIAL " + potential)

        output_lines = ["!!! Generated by ASE !!!"] + root.write()
        return "\n".join(output_lines)


if __name__ == "__main__":
    ...

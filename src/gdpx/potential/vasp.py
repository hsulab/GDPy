import os
import pathlib

from gdpx.backend.ase import DummyCalculator

from .manager import BasePotentialManager


def set_environs(pp_path: str, vdw_path: str) -> None:
    """Set files need for calculation.

    Note:
        pp_path and vdw_path may not exist since we would like to create a
        dummy calculator.

    """
    # Pseudo potential path
    if "VASP_PP_PATH" in os.environ.keys():
        os.environ.pop("VASP_PP_PATH", "")
    os.environ["VASP_PP_PATH"] = pp_path

    # Vdw kernel path
    vdw_envname = "ASE_VASP_VDW"
    if vdw_envname in os.environ.keys():
        _ = os.environ.pop(vdw_envname, "")
    os.environ[vdw_envname] = vdw_path

    return


def instantiate_vasp_interactive_calculator(
    calc_cls, command: str, calc_params: dict, is_remote: bool, inp_fdict: dict, use_socket: bool
):
    """"""
    # Command must contain the machine prefix, otherwise,
    # the vasp process will fail.
    calc = calc_cls(command=command, use_socket=use_socket)

    # Set some default electronic parameters
    calc.set_xc_params("PBE")  # incar may not set GGA
    calc.set(lorbit=10)
    calc.set(gamma=True)
    if not is_remote and inp_fdict["incar"] is not None:
        calc.read_incar(inp_fdict["incar"])

    # Set some vasp_interactive parameters
    calc.set(potim=0.0)
    calc.set(ibrion=-1)
    calc.set(ediffg=0)
    # calc.set(isif=3) # Does not support stress for now...

    set_environs(inp_fdict["pp_path"], inp_fdict["vdw_path"])

    # Update residual params
    calc.set(**calc_params)

    return calc


class VaspManager(BasePotentialManager):
    name = "vasp"

    implemented_backends = (
        "vasp",
        "vasp_interactive",
        "vasp_interactive_disp",
    )
    valid_combinations = (
        ("vasp", "vasp"),
        ("vasp_interactive", "ase"),
        ("vasp_interactive_disp", "ase"),
    )

    def _set_environs(self, pp_path, vdw_path) -> None:
        """Set files need for calculation.

        Note:
            pp_path and vdw_path may not exist since we would like to create a
            dummy calculator.

        """
        # Pseudo potential path
        if "VASP_PP_PATH" in os.environ.keys():
            os.environ.pop("VASP_PP_PATH", "")
        os.environ["VASP_PP_PATH"] = pp_path

        # Vdw kernel path
        vdw_envname = "ASE_VASP_VDW"
        if vdw_envname in os.environ.keys():
            _ = os.environ.pop(vdw_envname, "")
        os.environ[vdw_envname] = vdw_path

        return

    def register_calculator(self, calc_params: dict) -> None:
        """"""
        super().register_calculator(calc_params)

        # check whether the calculation will be sent to remote
        # If so, incar will be not read until actually simulation is performed.
        # The stored potter parameters should not contain remote.
        # Maybe it is better to move incar-related thing to VaspDriver...
        self.calc_params.pop("remote", None)
        is_remote = calc_params.pop("remote", False)

        # Some extra parameters
        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", pathlib.Path.cwd())

        use_socket = calc_params.pop("use_socket", False)

        # Some system-specific settings
        magmom_init = calc_params.pop("magmom_init", None)

        # Check whether check pp and vdw existence
        # since sometimes we'd like a dummy calculator

        inp_fdict = dict(
            incar=calc_params.pop("incar", None),
            pp_path=calc_params.pop("pp_path", ""),
            vdw_path=calc_params.pop("vdw_path", ""),
        )

        if not is_remote:
            for fname in inp_fdict.keys():
                inp_fdict[fname] = str(pathlib.Path(inp_fdict[fname]).resolve())
            self.calc_params.update(**inp_fdict)
        else:
            for fname, fpath in inp_fdict.items():
                if not pathlib.Path(fpath).is_absolute():
                    raise RuntimeError(f"{fname} for remote must be an absolute path.")

        calc = DummyCalculator()
        if self.calc_backend == "vasp":
            from ase.calculators.vasp import Vasp

            calc = Vasp(directory=directory, command=command)

            # Set some default electronic parameters
            calc.set_xc_params("PBE")  # incar may not set GGA
            calc.set(lorbit=10)
            calc.set(gamma=True)
            if not is_remote and inp_fdict["incar"] is not None:
                calc.read_incar(inp_fdict["incar"])
            self._set_environs(inp_fdict["pp_path"], inp_fdict["vdw_path"])

            # Update residual params
            calc.set(**calc_params)
        elif self.calc_backend == "vasp_interactive":
            from vasp_interactive import VaspInteractive

            # Command must contain the machine prefix, otherwise,
            # the vasp process will fail.
            calc = VaspInteractive(directory=directory, command=command, use_socket=use_socket)

            # Set some default electronic parameters
            calc.set_xc_params("PBE")  # incar may not set GGA
            calc.set(lorbit=10)
            calc.set(gamma=True)
            if not is_remote and inp_fdict["incar"] is not None:
                calc.read_incar(inp_fdict["incar"])

            # Set some vasp_interactive parameters
            calc.set(potim=0.0)
            calc.set(ibrion=-1)
            calc.set(ediffg=0)
            # calc.set(isif=3) # Does not support stress for now...

            self._set_environs(inp_fdict["pp_path"], inp_fdict["vdw_path"])

            # Update residual params
            calc.set(**calc_params)
        elif self.calc_backend == "vasp_interactive_disp":
            from dftd3.ase import DFTD3
            from vasp_interactive import VaspInteractive

            from gdpx.backend.vasp.calculators import VaspInteractiveWithDispersion

            disp_calc_params = calc_params.pop("dispersion", None)
            if disp_calc_params is None:
                raise Exception("vasp_interactive_disp must have `dispersion` section in `params`.")
            dispersion_type = disp_calc_params.pop("type", None)
            if dispersion_type != "dftd3":
                raise Exception("vasp_interactive_disp only supports `type` of `dftd3`.")
            disp_calc = DFTD3(**disp_calc_params)

            vasp_calc = instantiate_vasp_interactive_calculator(
                VaspInteractive,
                command=command,
                calc_params=calc_params,
                is_remote=is_remote,
                inp_fdict=inp_fdict,
                use_socket=use_socket,
            )

            calc = VaspInteractiveWithDispersion(
                calcs=[vasp_calc, disp_calc],
                save_host=True,
                directory=directory,
            )

        else:
            ...  # The backend has already been checked.

        # HACK: Some system-specific electronic structure settings
        calc.magmom_settings = magmom_init

        self.calc = calc

        return

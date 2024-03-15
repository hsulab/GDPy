#!/usr/bin/env python3
# -*- coding: utf-8 -*


import itertools
import pathlib
import re
import shutil

from typing import Union, List

from ase import Atoms

from .. import AbstractTrainer, AbstractPotentialManager, DummyCalculator

from gdpx.computation.mixer import CommitteeCalculator


def parse_reann_input_config(para: Union[str, pathlib.Path]) -> List[str]:
    """Parse `para/input_density` and return the `atomtype`.

    Args:
        para: The para directory path.

    Return:
        A List of Strings, which is the type_list.

    """
    type_list = []

    para = pathlib.Path(para)
    p_density = para / "input_density"
    if p_density.exists():
        with open(p_density, "r") as fopen:
            lines = fopen.readlines()
        for line in lines:
            if line.strip().startswith("atomtype"):
                # type_list = line.strip().split("#")[0].split("=")[1]
                m = re.findall("\[.*\]", line)
                assert len(m) == 1
                type_list = [str(x.strip(" '\"")) for x in m[0][1:-1].split(",")]
                break
        else:
            raise RuntimeError(f"No atomtype found in {str(para)}.")
    else:
        raise FileNotFoundError(f"{str(para)} does not exist.")

    return type_list


def _atoms2reannconfig(atoms: Atoms, point: int) -> str:
    """"""
    lattice_constants = atoms.get_cell(complete=True).flatten().tolist()

    content = f"{point = }\n"
    content += (("{:>12.8f}  " * 3 + "\n") * 3).format(*lattice_constants)
    content += ("pbc " + "{:<d} " * 3 + "\n").format(*atoms.get_pbc())

    # TODO: If input atoms have no forces?
    for s, m, pos, frc in zip(
        atoms.get_chemical_symbols(),
        atoms.get_masses(),
        atoms.get_positions(),
        atoms.get_forces(),
    ):
        content += (
            f"{s:<3s}  {m:<8.4f}  "
            + f"{pos[0]:>24.8f} {pos[1]:>24.8f} {pos[2]:>24.8f}"
            + f"{frc[0]:>24.8f} {frc[1]:>24.8f} {frc[2]:>24.8f}"
            + "\n"
        )
    
    # TODO: use electronic free energy?
    content += f"abprop: {atoms.get_potential_energy()}\n"

    return content


def convert_structures_to_reannconfig(
    structures: List[Atoms], fpath: Union[str, pathlib.Path]
) -> None:
    """"""
    content = ""
    for i, atoms in enumerate(structures):
        content += _atoms2reannconfig(atoms, i + 1)

    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as fopen:
        fopen.write(content)

    return content


class ReannDataloader:

    name: str = "reann"

    def __init__(self, directory: Union[str, pathlib.Path], *args, **kwargs) -> None:
        """"""
        self.directory = pathlib.Path(directory).resolve()

        return
    
    def as_dict(self, ) -> dict:
        """"""
        params = {}
        params["name"] = self.name
        #params["batchsizes"] = 4
        params["directory"] = str(self.directory.resolve())

        return params


class ReannTrainer(AbstractTrainer):

    name = "reann"
    command = ""
    freeze_command = ""
    prefix = "config"

    def __init__(
        self,
        config: Union[str, pathlib.Path],
        type_list: List[str] = None,
        train_epochs: int = 200,
        directory=".",
        command="train",
        freeze_command="freeze",
        random_seed: Union[int, dict] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            config,
            type_list,
            train_epochs,
            directory,
            command,
            freeze_command,
            random_seed,
            *args,
            **kwargs,
        )

        # TODO: convert `para` to a dict?
        self.config = pathlib.Path(self.config).resolve()
        self._type_list = parse_reann_input_config(self.config)

        return

    @property
    def frozen_name(self):
        """"""
        return f"PES.pt"

    def _resolve_train_command(self, *args, **kwargs):
        """"""

        return self.command

    def _resolve_freeze_command(self, *args, **kwargs):
        """"""

        return self.freeze_command

    def _prepare_dataset(self, dataset, *args, **kwargs):
        """Prepare a reann dataset for training.

        Currently, it only supports converting xyz dataset.

        """
        self._print(f"{dataset = }")
        # NOTE: make sure the dataset path exists, sometimes it will be access
        #       before training to create a shared dataset
        self.directory.mkdir(parents=True, exist_ok=True)

        if not isinstance(dataset, ReannDataloader):
            set_names, train_frames, test_frames, adjusted_batchsizes = (
                dataset.split_train_and_test()
            )

            # NOTE: reann does not support split-system training,
            #       so we need merge all structures into one List
            train_frames = itertools.chain(*train_frames)
            convert_structures_to_reannconfig(
                train_frames, self.directory / "train" / "configuration"
            )

            test_frames = itertools.chain(*test_frames)
            convert_structures_to_reannconfig(
                test_frames, self.directory / "val" / "configuration"
            )

            dataset = ReannDataloader(directory=self.directory)
        else:
            ...

        return dataset

    def write_input(self, dataset, *args, **kwargs):
        """"""
        self._print(f"write {self.name} inputs...")

        # - convert dataset
        dataset = self._prepare_dataset(dataset)
        
        # - copy input file
        _ = shutil.copytree(self.config, self.directory / "para", dirs_exist_ok=True)

        # TODO: change input configuration
        with open(self.directory/"para"/"input_nn", "r") as fopen:
            lines = fopen.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip().startswith("Epoch"):
                line = f"  Epoch={self.train_epochs}\n"
            elif line.strip().startswith("folder"):
                line = "  folder={}\n".format("\""+str(dataset.directory.resolve())+"/"+"\"")
            else:
                ...
            new_lines.append(line)

        with open(self.directory/"para"/"input_nn", "w") as fopen:
            fopen.write("".join(new_lines))

        return

    def read_convergence(self) -> bool:
        """"""
        self._print(f"check {self.name} training convergence...")
        converged = False

        log_path = self.directory / "nn.err"
        if log_path.exists():
            with open(log_path, "r") as fopen:
                lines = fopen.readlines()
            try:
                end_info = lines[-1]
                if end_info.strip() == "terminated normal":
                    converged = True
                self._debug(f"{end_info = }")
            except:
                self._print(f"The endline of `nn.err` is strange.")
        else:
            ...

        return converged
    
    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["config"] = str(self.config) # For YAML...

        return params


class ReannManager(AbstractPotentialManager):

    name = "reann"
    implemented_backends = [
        "ase",
    ]

    valid_combinations = (("ase", "ase"),)

    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        # - some shared params
        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", pathlib.Path.cwd())

        # -
        type_list = calc_params.pop("type_list", [])
        
        # --- model files
        model_ = calc_params.get("model", [])
        if not isinstance(model_, list):
            model_ = [model_]

        models = []
        for m in model_:
            m = pathlib.Path(m).resolve()
            if not m.exists():
                raise FileNotFoundError(f"Cant find model file {str(m)}")
            models.append(str(m))

        precision = calc_params.pop("precision", "float32")
        assert precision in ["float32", "float64"]

        # TODO: make this a dataclass??
        #       currently, default disable uncertainty estimation
        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        # - misc
        max_nneigh = calc_params.get("max_nneigh", 25000)

        # - parse calc_params
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import torch
                from reann.ASE.calculators.reann import REANN
                from reann.ASE import getneigh
                device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
                if precision == "float32":
                    precision = torch.float32
                elif precision == "float64":
                    precision = torch.float64
                else:
                    ...
            except:
                raise ModuleNotFoundError("Please install reann and torch to use the ase interface.")

            calcs = []
            for m in models:
                calc = REANN(
                    atomtype=type_list, maxneigh=max_nneigh, getneigh=getneigh, 
                    nn=m, device=device, dtype=precision
                )
                calcs.append(calc)
            if len(calcs) == 1:
                calc = calcs[0]
            elif len(calcs) > 1:
                if estimate_uncertainty:
                    calc = CommitteeCalculator(calcs=calcs)
                else:
                    calc = calcs[0]
            else:
                ...
        else:
            ...
        
        self.calc = calc

        return


if __name__ == "__main__":
    ...

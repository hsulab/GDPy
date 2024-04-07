#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import itertools
import pathlib
import re
import shutil

from typing import Union, List

import omegaconf

from ase import Atoms

from .. import AbstractTrainer, AbstractPotentialManager, DummyCalculator
from .. import CommitteeCalculator


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


def _convert_a_single_line(line: str):
    """"""
    # - clean line
    data = line.strip().split("#")[0]
    k, v = data.split("=")
    k = k.strip()
    v = v.strip()

    def is_number(s) -> bool:
        """"""
        try:
            float(s)
            return True
        except ValueError:
            ...

        return False

    def is_list(s) -> bool:
        """"""
        m = re.findall("\[.*\]", s)
        if len(m) == 0:
            return False
        elif len(m) == 1:
            return True
        else:
            raise RuntimeError(f"Fail to parse {s}.")

    # - convert value
    if is_number(v):
        v = v.strip()
        if v.isdigit():
            v = int(v)
        else:
            v = float(v)
    elif is_list(v):
        v_ = []
        entries = v.strip()[1:-1].split(",")
        for x in entries:
            x = x.strip()
            if is_number(x):
                if x.isdigit():
                    x = int(x)
                else:
                    x = float(x)
            else:  # assume it is a simple string
                x = x.strip(" '\"")
            v_.append(x)
        v = v_
    else:  # assume it is a simple string
        v = v.strip(" '\"")

    return (k, v)


def load_reann_input_para(para: Union[str, pathlib.Path]) -> dict:
    """"""
    params = {}

    para = pathlib.Path(para)

    # - input_density
    with open(para / "input_density", "r") as fopen:
        lines = fopen.readlines()

    params["density"] = {}
    for line in lines:
        if not line.strip().startswith("#"):
            k, v = _convert_a_single_line(line)
            params["density"][k] = v

    # - input_dnn
    with open(para / "input_nn", "r") as fopen:
        lines = fopen.readlines()

    params["nn"] = {}
    for line in lines:
        if not line.strip().startswith("#"):
            k, v = _convert_a_single_line(line)
            params["nn"][k] = v

    return params

def dump_reann_input_para(config_params: dict, para: Union[str, pathlib.Path]):
    """"""
    with open(para/"input_density", "w") as fopen:
        content = ""
        for k, v in config_params["density"].items():
            if isinstance(v, str) and (v != "True" or v != "False"):
                content += "{}=\"{}\"\n".format(k, str(v))
            else:
                content += f"{k}={str(v)}\n"
        fopen.write(content)

    with open(para/"input_nn", "w") as fopen:
        content = ""
        for k, v in config_params["nn"].items():
            if isinstance(v, str) and (v != "True" and v != "False"):
                content += "{}=\"{}\"\n".format(k, str(v))
            else:
                content += f"{k}={str(v)}\n"
        fopen.write(content)

    return


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

    def __init__(self, batchsize: int, directory: Union[str, pathlib.Path]="./", *args, **kwargs) -> None:
        """"""
        self.batchsize = batchsize
        self.directory = pathlib.Path(directory).resolve()

        return

    def as_dict(
        self,
    ) -> dict:
        """"""
        params = {}
        params["name"] = self.name
        params["batchsize"] = self.batchsize
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
        print_epochs: int = 5,
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
            print_epochs,
            directory,
            command,
            freeze_command,
            random_seed,
            *args,
            **kwargs,
        )

        # self.config = pathlib.Path(self.config).resolve()
        if isinstance(config, dict) or isinstance(config, omegaconf.dictconfig.DictConfig):
            self.config = config
        elif isinstance(config, str) or isinstance(config, pathlib.Path):
            self.config = load_reann_input_para(config)
        else:
            raise RuntimeError(f"Fail to parse {config = }.")

        self._type_list = self.config["density"]["atomtype"]

        return

    @property
    def frozen_name(self):
        """"""
        return f"PES.pt"
    
    @property
    def ckpt_name(self) -> str:
        """"""

        return "REANN.pth"

    def _resolve_train_command(self, *args, **kwargs):
        """"""

        return self.command

    def _resolve_freeze_command(self, *args, **kwargs):
        """"""

        return self.freeze_command
    
    def get_checkpoint(self):
        """"""

        return pathlib.Path(self.directory/self.ckpt_name).resolve()
    
    def _train_from_the_restart(self, dataset, init_model) -> str:
        """Train from the restart"""
        def _train_from_the_scratch(dataset, init_model) -> str:
            # NOTE: `para/input*` will be overwritten by current self.config
            command = self._train_from_the_scratch(dataset, init_model)
            if init_model is not None:
                self._print(f"{self.name} init training from model {init_model}.")
                (self.directory/self.ckpt_name).unlink(missing_ok=True)
                shutil.copyfile(init_model, self.directory/self.ckpt_name)
                prev_config = load_reann_input_para(self.directory/"para")
                prev_config["nn"]["table_init"] = 1
                _ = dump_reann_input_para(prev_config, self.directory/"para")
            else:
                ...
            
            return command

        if not self.directory.exists():
            command = _train_from_the_scratch(dataset, init_model)
        else:
            ckpt_info = self.directory/self.ckpt_name
            if ckpt_info.exists() and ckpt_info.stat().st_size != 0:
                # -
                log_path = self.directory/"nn.err"
                if log_path.exists():
                    with open(self.directory/"nn.err", "r") as fopen:
                        lines = fopen.readlines()
                    epoch_lines = [l for l in lines if l.strip().startswith("Epoch")]
                    try:
                        end_epoch = int(epoch_lines[-1].split()[1])
                        self._debug(f"{end_epoch =}")
                    except:
                        end_epoch = 0
                        self._print(f"The endline of `nn.err` is strange.")
                    # - 
                    prev_config = load_reann_input_para(self.directory/"para")
                    prev_config["nn"]["table_init"] = 1
                    prev_config["nn"]["Epoch"] = prev_config["nn"]["Epoch"] - end_epoch
                    prev_config["nn"]["patience_epoch"] = 0
                    assert prev_config["nn"]["Epoch"] >= 0
                    _ = dump_reann_input_para(prev_config, self.directory/"para")
                    self._print(f"{self.name} restarts training from epoch {end_epoch}.")
                    command = self._resolve_train_command()
                else:
                    # NOTE: This is a new training but init from a previous model.
                    command = _train_from_the_scratch(dataset, init_model)
            else:
                # restart from the scratch and overwrite exists.
                command = _train_from_the_scratch(dataset, init_model)

        return command

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

            dataset = ReannDataloader(
                directory=self.directory, batchsize=dataset.batchsize
            )
        else:
            ...

        return dataset

    def write_input(self, dataset, *args, **kwargs):
        """"""
        self._print(f"write {self.name} inputs...")

        # - convert dataset
        dataset = self._prepare_dataset(dataset)

        # - copy input file
        (self.directory / "para").mkdir(parents=True, exist_ok=True)

        config_params = copy.deepcopy(self.config)
        config_params["nn"]["Epoch"] = self.train_epochs
        config_params["nn"]["print_epoch"] = self.print_epochs
        config_params["nn"]["batchsize_train"] = dataset.batchsize
        config_params["nn"]["batchsize_val"] = dataset.batchsize
        config_params["nn"]["folder"] = str(dataset.directory.resolve()) + "/"

        dump_reann_input_para(config_params, self.directory/"para")

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
        self.calc_params.update(model=models)

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
                from reann.ASE import getneigh
                from .calculators.reann import REANN

                device = torch.device(
                    "cuda" if torch.cuda.is_available() else torch.device("cpu")
                )
                if precision == "float32":
                    precision = torch.float32
                elif precision == "float64":
                    precision = torch.float64
                else:
                    ...
            except:
                raise ModuleNotFoundError(
                    "Please install reann and torch to use the ase interface."
                )

            calcs = []
            for m in models:
                calc = REANN(
                    atomtype=type_list,
                    maxneigh=max_nneigh,
                    getneigh=getneigh,
                    nn=m,
                    device=device,
                    dtype=precision,
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

    def switch_uncertainty_estimation(self, status: bool = True):
        """Switch on/off the uncertainty estimation."""
        # NOTE: Sometimes the manager loads several models and supports uncertainty
        #       by committee but the user disables it. We need change the calc to
        #       the correct one as the loaded one is just a single calculator.
        if not hasattr(self, "calc"):
            raise RuntimeError(
                "Fail to switch uncertainty status as it does not have a calc."
            )
        # print(f"{self.calc}")

        # NOTE: make sure manager.as_dict() can have correct param
        self.calc_params["estimate_uncertainty"] = status

        # - convert calculator
        if self.calc_backend == "ase":
            if status:
                if isinstance(self.calc, CommitteeCalculator):
                    ...  # nothing to do
                else:  # reload models
                    self.register_calculator(self.calc_params)
            else:
                if isinstance(self.calc, CommitteeCalculator):
                    # TODO: save previous calc?
                    self.calc = self.calc.calcs[0]
                else:
                    ...
        elif self.calc_backend == "lammps":
            ...
        else:
            # TODO:
            # Other backends cannot have uncertainty estimation,
            # give a warning?
            ...

        return

    def remove_loaded_models(self, *args, **kwargs):
        """Loaded TF models should be removed before any copy.deepcopy operations."""
        self.calc.reset()
        if self.calc_backend == "ase":
            if isinstance(self.calc, CommitteeCalculator):
                for c in self.calc.calcs:
                    c.pes = None
                    c.getneigh = None
            else:
                self.calc.pes = None
                self.calc.getneigh = None
        else:
            ...

        return


if __name__ == "__main__":
    ...

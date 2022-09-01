#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
Potential Manager
deals with various machine learning potentials
"""

import os
import json
import yaml
import abc
import copy
import pathlib
from pathlib import Path
from typing import Union, List, NoReturn


from GDPy.trainer.train_potential import find_systems, generate_random_seed

from GDPy.utils.command import run_command

from GDPy.scheduler.factory import create_scheduler


class AbstractPotential(abc.ABC):
    """
    Create various potential instances
    """

    name = "potential"
    version = "m00" # calculator name

    external_engines = ["lammps", "lasp"]

    implemented_backends = []
    backends = dict(
        single = [], # single pointe energy
        dynamics = [] # dynamics (opt, ts, md)
    )
    valid_combinations = []

    _calc = None
    _scheduler = None

    def __init__(self):
        """
        """

        self.uncertainty = None # uncertainty estimation method

        return
    
    @property
    def calc(self):
        return self._calc
    
    @calc.setter
    def calc(self, calc_):
        self._calc = calc_
        return 
    
    @property
    def scheduler(self):
        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, scheduler_):
        self._scheduler = scheduler_
        return
    
    @abc.abstractmethod
    def register_calculator(self, calc_params, *agrs, **kwargs):
        """ register calculator
        """
        self.calc_backend = calc_params.pop("backend", self.name)
        if self.calc_backend not in self.implemented_backends:
            raise RuntimeError(f"Unknown backend for potential {self.name}")

        self.calc_params = copy.deepcopy(calc_params)

        return

    def create_driver(
        self, 
        dyn_params: dict = {},
        *args, **kwargs
    ):
        """ create a worker for dynamics
            default the dynamics backend will be the same as calc
            however, ase-based dynamics can be used for all calculators
        """
        if not hasattr(self, "calc") or self.calc is None:
            raise AttributeError("Cant create driver before a calculator has been properly registered.")
            
        # parse backends
        self.dyn_params = dyn_params
        dynamics = dyn_params.get("backend", self.calc_backend)
        if dynamics == "external":
            dynamics = self.calc_backend
            #if dynamics.lower() in self.external_engines:
            #    dynamics = self.calc_backend
            #else:
            #    dynamics == "ase"

        if [self.calc_backend, dynamics] not in self.valid_combinations:
            raise RuntimeError(f"Invalid dynamics backend {dynamics} based on {self.calc_backend} calculator")
        
        # create dynamics
        calc = self.calc

        if dynamics == "ase":
            from GDPy.computation.ase import AseDriver as driver_cls
            # use ase no need to recaclc constraint since atoms has one
            # cons_indices = None # this is used in minimise
        elif dynamics == "lammps":
            from GDPy.computation.lammps import LmpDriver as driver_cls
            # use lammps optimisation
            #else:
            #    raise NotImplementedError("no other eann lammps dynamics")
        elif dynamics == "lasp":
            from GDPy.computation.lasp import LaspDriver as driver_cls
        elif dynamics == "vasp":
            from GDPy.computation.vasp import VaspDriver as driver_cls

        driver = driver_cls(calc, dyn_params, directory=calc.directory)
        driver.pot_params = self.as_dict()
        
        return driver
    
    def register_scheduler(self, params_, *args, **kwargs) -> NoReturn:
        """ register machine used to submit jobs
        """
        params = copy.deepcopy(params_)
        scheduler = create_scheduler(params)

        self.scheduler = scheduler

        return

    def create_worker(self, driver, scheduler, *args, **kwargs):
        """ a machine operates calculations submitted to queue
        """
        # TODO: check driver is consistent with this potential

        from GDPy.computation.worker.worker import DriverBasedWorker
        worker = DriverBasedWorker(driver, scheduler)

        # TODO: check if scheduler resource satisfies driver's command

        return worker
    
    def as_dict(self):
        """"""
        params = {}
        params["name"] = self.name

        pot_params = {"backend": self.calc_backend}
        pot_params.update(copy.deepcopy(self.calc_params))
        params["params"] = pot_params

        # TODO: add train params?

        return params

    
class VaspManager(AbstractPotential):

    name = "vasp"

    implemented_backends = ["vasp"]

    valid_combinations = [
        ["vasp", "vasp"] # calculator, dynamics
    ]

    def __init__(self):

        return

    def _set_environs(self, pp_path, vdw_path):
        # ===== environs TODO: from inputs 
        # - ASE_VASP_COMMAND
        # pseudo 
        if "VASP_PP_PATH" in os.environ.keys():
            os.environ.pop("VASP_PP_PATH")
        os.environ["VASP_PP_PATH"] = pp_path

        # vdw 
        vdw_envname = "ASE_VASP_VDW"
        if vdw_envname in os.environ.keys():
            _ = os.environ.pop(vdw_envname)
        os.environ[vdw_envname] = vdw_path

        return

    def register_calculator(self, calc_params):
        """"""
        # - check backends
        super().register_calculator(calc_params)

        # - some extra params
        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())

        # TODO: check pp and vdw
        incar = calc_params.pop("incar", None)
        pp_path = calc_params.pop("pp_path", None)
        vdw_path = calc_params.pop("vdw_path", None)

        if self.calc_backend == "vasp":
            # return ase calculator
            from ase.calculators.vasp import Vasp
            calc = Vasp(directory=directory, command=command)

            # - set some default params
            calc.set_xc_params("PBE") # NOTE: since incar may not set GGA
            calc.set(lorbit=10)
            calc.set(gamma=True)
            calc.set(lreal="Auto")
            if incar is not None:
                calc.read_incar(incar)
            self._set_environs(pp_path, vdw_path)
            #print("fmax: ", calc.asdict()["inputs"])
            # - update residual params
            calc.set(**calc_params)
        else:
            pass
        
        self.calc = calc

        return


class RXManager(AbstractPotential):

    name = 'RX'
    implemented_backends = ['lammps']

    def __init__(self, backend: str, models: str, type_map: dict):
        """"""
        self.model = models
        self.type_map = type_map

        return

    def generate_calculator(self):
        """ generate calculator with various backends
        """
        if self.backend == 'lammps':
            # return reax pair related content
            content = "units           real\n"
            content += "atom_style      charge\n"
            content += "\n"
            content += "neighbor            1.0 bin\n" # for npt, ghost atom issue
            content += "neigh_modify     every 10 delay 0 check no\n"
            content += 'pair_style  reax/c NULL\n'
            content += 'pair_coeff  * * %s %s\n' %(self.model, ' '.join(list(self.type_map.keys())))
            content += "fix             2 all qeq/reax 1 0.0 10.0 1e-6 reax/c\n"
            calc = content
        else:
            pass

        return calc

class DpManager(AbstractPotential):

    name = 'DP'
    implemented_backends = ['ase', 'lammps']

    def __init__(self, backend: str, models: Union[str, list], type_map: dict):
        """ create a dp manager
        """
        self.backend = backend
        if self.backend not in self.implemented_backends:
            raise NotImplementedError('Backend %s is not implemented.' %self.backend)

        # check models
        self.models = models
        self._parse_models()
        self._check_uncertainty_support()

        self.type_map = type_map

        return
    
    def _parse_models(self):
        """"""
        if isinstance(self.models, str):
            pot_path = pathlib.Path(self.models)
            pot_dir, pot_pattern = pot_path.parent, pot_path.name
            models = []
            for pot in pot_dir.glob(pot_pattern):
                models.append(str(pot/'graph.pb'))
            self.models = models
        else:
            for m in self.models:
                if not pathlib.Path(m).exists():
                    raise ValueError('Model %s does not exist.' %m)

        return
    
    def _check_uncertainty_support(self):
        """"""
        self.uncertainty = False
        if len(self.models) > 1:
            self.uncertainty = True

        return
    
    def generate_calculator(self):
        """ generate calculator with various backends
        """
        if self.backend == 'ase':
            # return ase calculator
            from GDPy.computation.dp import DP
            calc = DP(model=self.models, type_dict=self.type_map)
        elif self.backend == 'lammps':
            # return deepmd pair related content
            #content = "units           metal\n"
            #content += "atom_style      atomic\n"
            content = "neighbor        1.0 bin\n"
            content += "pair_style      deepmd %s out_freq ${THERMO_FREQ} out_file model_devi.out\n" \
                %(' '.join([m for m in self.models]))
            content += 'pair_coeff    \n'
            calc = content

        return calc
    
    def create_thermostat(self):

        return
    
    def create_ensemble(self):
        """
        """
        # TODO: make these paramters
        main_dict = []
        iter_directory = ""
        find_systems = ""
        generate_random_seed = ""

        # parse params
        machine_json = main_dict['machines']['trainer']
        num_models = main_dict['training']['num_models']
        train_json = main_dict['training']['json']

        # prepare dataset
        """
        xyz_files = []
        for p in main_database.glob('*.xyz'):
            xyz_files.append(p)
    
        frames = []
        for xyz_file in xyz_files:
            frames.extend(read(xyz_file, ':'))
    
        print(len(frames))
        """

        # find systems
        # data_path = Path('/users/40247882/projects/oxides/gdp-main/merged-dataset/raw_data')
        data_path = iter_directory / 'raw_data'
        systems = find_systems(data_path)

        # machine file
        from ..machine.machine import SlurmMachine
        slurm_machine = SlurmMachine(machine_json)

        # read json
        with open(train_json, 'r') as fopen:
            params_dict = json.load(fopen)
        # change seed and system dirs
        ensemble_dir = iter_directory / 'ensemble'
        ensemble_dir.mkdir()
        for idx in range(num_models):
            #print(params_dict)
            model_dir = ensemble_dir / ('model-'+str(idx))
            model_dir.mkdir()
            seed = generate_random_seed()
            params_dict['model']['descriptor']['seed'] = int(seed)
            params_dict['model']['fitting_net']['seed'] = int(seed)

            params_dict['training']['systems'] = [str(s) for s in systems]

            with open(model_dir/'dp.json', 'w') as fopen:
                json.dump(params_dict, fopen, indent=4)

            # write machine 
            #restart = True
            restart = False
            slurm_machine.machine_dict['job-name'] = 'model-'+str(idx)
            if restart:
                parent_model = '/users/40247882/projects/oxides/gdp-main/it-0008/ensemble/model-%d/model.ckpt' %idx
                command = "dp train ./dp.json --init-model %s 2>&1 > dp.out" %parent_model
                slurm_machine.machine_dict['command'] = command
            else:
                pass
            slurm_machine.write(model_dir/'dptrain.slurm')

        return

    def freeze_ensemble(self):
        """ freeze trained potentials
        """
        # find models
        model_dirs = []
        for p in self.ensemble_path.glob('model*'):
            model_dirs.append(p)
        model_dirs.sort()

        # freeze models
        for model_dir in model_dirs:
            if self.check_finished(model_dir):
                self.__freeze(model_dir)
            self.__freeze(model_dir)

        return

    def check_finished(self, model_path):
        """check if the training is finished"""
        converged = False
        model_path = pathlib.Path(model_path)
        dpout_path = model_path / 'dp.out'
        if dpout_path.exists():
            content = dpout_path.read_text()
            line = content.split('\n')[-3]
            print(line)
            #if 'finished' in line:
            #    converged = True

        return converged

    def __freeze(self, model_path):
        command = "dp freeze -o graph.pb"
        output = run_command(model_path, command)
        print(output)
        return 

class EannManager(AbstractPotential):

    name = "eann"
    implemented_backends = ["ase", "lammps"]

    backends = dict(
        single = ["ase", "lammps"], # single pointe energy
        dynamics = ["ase", "lammps"] # dynamics (opt, ts, md)
    )

    valid_combinations = [
        ["ase", "ase"], # calculator, dynamics
        ["lammps", "ase"],
        ["lammps", "lammps"],
    ]

    TRAIN_INPUT_NAME = "input_nn.json"
    
    def __init__(self):

        return
    
    def __parse_models(self):
        """"""
        if isinstance(self.models, str):
            pot_path = pathlib.Path(self.models)
            pot_dir, pot_pattern = pot_path.parent, pot_path.name
            models = []
            for pot in pot_dir.glob(pot_pattern):
                models.append(str(pot))
            self.models = models
        else:
            for m in self.models:
                if not pathlib.Path(m).exists():
                    raise ValueError('Model %s does not exist.' %m)

        return
    
    def __check_uncertainty_support(self):
        """"""
        self.uncertainty = False
        if len(self.models) > 1:
            self.uncertainty = True

        return
    
    def register_calculator(self, calc_params, *args, **kwargs):
        """"""
        super().register_calculator(calc_params)

        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())
        atypes = calc_params.pop("type_list", [])

        models = calc_params.pop("file", None)
        pair_style = calc_params.get("pair_style", None)

        type_map = {}
        for i, a in enumerate(atypes):
            type_map[a] = i

        if self.calc_backend == "ase":
            # return ase calculator
            from eann.interface.ase.calculator import Eann
            if models:
                calc = Eann(
                    model=models, type_map=type_map,
                    command = command, directory=directory
                )
            else:
                calc = None
        elif self.calc_backend == "lammps":
            # eann has different backends (ase, lammps)
            from GDPy.computation.lammps import Lammps
            if pair_style:
                calc = Lammps(command=command, directory=directory, **calc_params)

                # TODO: assert unit and atom_style
                #content = "units           metal\n"
                #content += "atom_style      atomic\n"
                #content = "neighbor        0.0 bin\n"
                #content += "pair_style      eann %s \n" \
                #    %(' '.join([m for m in models]))
                #content += "pair_coeff * * double %s" %(" ".join(atypes))
                #calc = content
            else:
                calc = None
        
        self.calc = calc

        return
    
    def register_trainer(self, train_params_: dict):
        """"""
        train_params = copy.deepcopy(train_params_)
        self.train_config = train_params.get("config", None)

        self.train_size = train_params.get("size", 1)
        self.train_dataset = train_params.get("dataset", None)

        scheduelr_params = train_params.get("scheduler", {}) 
        self.train_scheduler = create_scheduler(scheduelr_params)

        train_command = train_params.get("train", None)
        self.train_command = train_command

        freeze_command = train_params.get("freeze", None)
        self.freeze_command = freeze_command

        return

    def _make_train_files(self, dataset=None, train_dir=Path.cwd()):
        """ make files for training
        """
        # - add dataset to config
        if not dataset:
            dataset = self.train_dataset
        assert dataset, f"No dataset has been set for the potential {self.name}."

        # TODO: for now, only List[Atoms]
        from GDPy.computation.utils import get_composition_from_atoms
        groups = {}
        for atoms in dataset:
            composition = get_composition_from_atoms(atoms)
            key = "".join([k+str(v) for k,v in composition])
            if key in groups:
                groups[key].append(atoms)
            else:
                groups[key] = [atoms]
        from ase.io import read, write
        systems = []
        dataset_dir = train_dir/"dataset"
        dataset_dir.mkdir()
        for key, frames in groups.items():
            k_dir = dataset_dir/key
            k_dir.mkdir()
            write(k_dir/"frames.xyz", frames)
            systems.append(str(k_dir.resolve()))

        dataset_config = dict(
            style = "auto",
            systems = systems,
            batchsizes = [[self.train_config["training"]["batchsize"],len(systems)]]
        )

        train_config = copy.deepcopy(self.train_config)
        train_config.update(dataset=dataset_config)
        with open(train_dir/"config.yaml", "w") as fopen:
            yaml.safe_dump(train_config, fopen, indent=2)

        return
    
    def train(self, dataset=None, train_dir=Path.cwd()):
        """"""
        self._make_train_files(dataset, train_dir)
        # run command

        return
    
    def freeze(self, train_dir=Path.cwd()):
        """ freeze model and return a new calculator
            that may have a committee for uncertainty
        """
        # - find subdirs
        train_dir = Path(train_dir)
        mdirs = []
        for p in train_dir.iterdir():
            if p.is_dir() and p.name.startswith("m"):
                mdirs.append(p.resolve())
        assert len(mdirs) == self.train_size, "Number of models does not equal model size..."

        # - find models
        calc_params = copy.deepcopy(self.calc_params)
        calc_params.update(backend=self.calc_backend)
        if self.calc_backend == "ase":
            models = []
            for p in mdirs:
                models.append(str(p/"eann_latest_py_DOUBLE.pt"))
            calc_params["file"] = models
        elif self.calc_backend == "lammps":
            models = []
            for p in mdirs:
                models.append(str(p/"eann_latest_lmp_DOUBLE.pt"))
            calc_params["pair_style"] = "eann {}".format(" ".join(models))
        #print("models: ", models)
        self.register_calculator(calc_params)

        return


class LaspManager(AbstractPotential):

    name = "lasp"
    implemented_backends = ["lasp"]
    valid_combinations = [
        ["lasp", "lasp"], # calculator, dynamics
        ["lasp", "ase"]
    ]

    def __init__(self):

        return

    def register_calculator(self, calc_params):
        """ params
            command
            directory
            pot
        """
        super().register_calculator(calc_params)

        self.calc_params["pot_name"] = self.name

        # NOTE: need resolved pot path
        pot = {}
        pot_ = calc_params.get("pot")
        for k, v in pot_.items():
            pot[k] = str(Path(v).resolve())
        self.calc_params["pot"] = pot

        self.calc = None
        if self.calc_backend == "lasp":
            from GDPy.computation.lasp import LaspNN
            self.calc = LaspNN(**self.calc_params)
        elif self.calc_backend == "lammps":
            # TODO: add lammps calculator
            pass
        else:
            raise NotImplementedError(f"{self.name} does not have {self.calc_backend}.")

        return
    

class NequipManager(AbstractPotential):

    implemented_backends = ["ase"]
    valid_combinations = [
        ["ase", "ase"], # calculator, dynamics
        ["lammps", "lammps"]
    ]
    
    def __init__(self):
        pass

    def register_calculator(self, calc_params):
        """"""
        self.calc_params = calc_params

        backend = calc_params["backend"]
        if backend not in self.implemented_backends:
            raise RuntimeError()

        command = calc_params["command"]
        directory = calc_params["directory"]
        models = calc_params["file"]
        atypes = calc_params["type_list"]

        type_map = {}
        for i, a in enumerate(atypes):
            type_map[a] = i

        if backend == "ase":
            # return ase calculator
            from nequip.ase import NequIPCalculator
            calc = NequIPCalculator.from_deployed_model(
                model_path=models
            )
        elif backend == "lammps":
            pass
        
        self.calc = calc

        return


if __name__ == "__main__":
    pass
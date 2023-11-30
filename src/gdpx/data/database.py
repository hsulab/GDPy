#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import time
import logging
import pathlib

from typing import Union, List, NoReturn
from collections import Counter


from ase import Atoms
from ase.io import read, write
from ase.formula import Formula


class StructureSystem():

    """A system contains structures from either a folder or a database.
    """

    name = None

    desc = ""
    chemical_formula:Formula = None
    substrate = "" # molecule, bulk, surface, cluster
    composition:dict = {}

    frames = None

    pfunc = print
    
    #: System-dependent parameters, e.g., kpts.
    specific_params:dict = {}

    def __init__(self, origin, name, composition:dict =None, *args, **kwargs) -> None:
        """"""
        self.origin = pathlib.Path(origin) # TODO: either a folder or a database
        self.name = name

        self.desc, self.chemical_formula, self.substrate = self._parse_name_components()
        composition_ = dict(Counter(self.chemical_formula))
        if composition is not None:
            assert composition == composition_, "Input composition is inconsistent with name."
        self.composition = composition_

        self.specific_params = kwargs

        return

    def _parse_name_components(self):
        """System name should be one of three formats.

        1. simple chemical formula. 2. custom name plus chemical formula.
        3. custom name, chemical formula and system type.

        """
        desc, substrate = "", ""
        name_parts = self.name.split("-")
        if len(name_parts) == 1:
            chemical_formula = name_parts[0]
        elif len(name_parts) == 2:
            desc, chemical_formula = name_parts
        elif len(name_parts) == 3:
            desc, chemical_formula, substrate = name_parts
        else:
            raise ValueError("directory name must be as xxx, xxx-xxx, or xxx-xxx-xxx")
        chemical_formula = Formula(chemical_formula)

        return desc, chemical_formula, substrate
    
    def _read_structures(self) -> List[Atoms]:
        """"""
        sys_path = self.origin/self.name

        total_frames = []
        stru_files = list(sys_path.glob("*.xyz")) # TODO: support more formats
        stru_files.sort()
        for p in stru_files:
            frames = read(p, ":")
            total_frames.extend(frames)
            self.pfunc(f"  structure path: {str(p)}")
            self.pfunc(f"    number of frames: {len(frames)}")
        self.pfunc(f"  nframes: {len(total_frames)}")

        self.frames = total_frames

        return total_frames
    
    def _mask_structures(self):
        """"""

        return


class StructureDatabase:

    """
    """

    name = "StructureDatabase"

    #: Working directory.
    _directory: pathlib.Path = None

    restart = False

    def __init__(self, config_params: dict, pot_worker=None, ref_worker=None) -> None:
        """Initialise a structure database.

        Args:
            config_params: Configuration parameters.

        """
        self.directory = pathlib.Path.cwd()
        self._init_logger(self.directory)

        # params should have
        #   - database
        #   - type_list
        #   - systems
        self.database = config_params.get("database", None) # path
        assert self.database, "Database is undefined."
        self.database = pathlib.Path(self.database)

        system_dict = config_params.get("systems", {})

        # - create systems
        self.systems = []
        for name, sys_params in system_dict.items():
            sys_params = copy.deepcopy(sys_params)
            composition = sys_params.pop("composition", None)
            sys = StructureSystem(
                origin=self.database, name=name, composition=composition,
                **sys_params
            )
            sys.pfunc = self.logger.info
            self.systems.append(sys)
        assert len(self.systems) > 0, "No systems are found."

        self.pot_worker = pot_worker
        self.ref_worker = ref_worker

        return

    @property
    def directory(self) -> pathlib.Path:
        """Working directory.

        Note:
            When setting directory, info_fpath would be set as well.

        """
        return self._directory
    
    @directory.setter
    def directory(self, directory_) -> NoReturn:
        self._directory = pathlib.Path(directory_)

        return 

    def _init_logger(self, working_directory):
        """"""
        self.logger = logging.getLogger(__name__)

        log_level = logging.INFO

        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        working_directory = pathlib.Path(working_directory)
        log_fpath = working_directory / (self.name+".out")

        if self.restart:
            fh = logging.FileHandler(filename=log_fpath, mode="a")
        else:
            fh = logging.FileHandler(filename=log_fpath, mode="w")

        fh.setLevel(log_level)
        #fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        #ch.setFormatter(formatter)

        #self.logger.addHandler(ch)
        #self.logger.addHandler(fh)

        # begin!
        self.logger.info(
            "\nStart at %s\n", 
            time.asctime( time.localtime(time.time()) )
        )

        return
    
    def run(self, params: dict):
        """"""
        # - summarise systems
        nframes = 0
        for sys in self.systems:
            self.logger.info(f"SystemName: {sys.name}")
            _ = sys._read_structures()
            nframes += len(sys.frames)
        self.logger.info(f"Total Number of Frames: {nframes} by {len(self.systems)} systems.")

        # - operations...
        assert len(params), "Currently, only one operation is supported."
        for op_name, op_params in params.items():
            if op_name == "select":
                self._select(op_params)
            elif op_name == "label":
                self._label(op_params)
            break

        return
    
    def _select(self, params):
        """"""
        from gdpx.selector import create_selector
        selector = create_selector(params)
        selector.logger = self.logger

        out_path = self.directory / "selected"
        if not out_path.exists():
            out_path.mkdir()

        for sys in self.systems:
            sys_sel_path = out_path/sys.name
            if not sys_sel_path.exists():
                sys_sel_path.mkdir()
            selector.directory = sys_sel_path
            selected_frames = selector.select(sys.frames)
            write(sys_sel_path/"sel.xyz", selected_frames)

        return
    
    def _label(self, params):
        """Label structures with given potential.

        This is useful when many structures need benchmark.

        """
        # - update some specific params of worker
        worker = self.pot_worker
        worker.logger = self.logger
        #worker._submit = False
        
        out_path = self.directory / "label"
        if not out_path.exists():
            out_path.mkdir()
        #else: # TODO: warning?
        #    raise FileNotFoundError("Directory label exists.")
        
        res_path = out_path / "results"
        if not res_path.exists():
            res_path.mkdir()

        # TODO: add an outlier detector? selection...
        #       sometimes expedition gives unphysical structures
        for sys in self.systems:
            # -- update basic info for the worker
            nframes_in = len(sys.frames)
            worker.directory = out_path/sys.name
            worker.batchsize = nframes_in

            # -- check outputs
            frames_out_path = res_path/(sys.name+".xyz")
            nframes_saved = 0
            if frames_out_path.exists():
                frames_saved = read(frames_out_path, ":")
                nframes_saved = len(frames_saved)
            if nframes_saved == nframes_in: # TODO: check if frames are consistent?
                self.logger.info(f"!! {sys.name} has been saved.")
                continue

            # --- update some specific params, e.g., kpts.
            for k in worker.driver.syswise_keys:
                v = sys.specific_params.get(k, None)
                if v:
                    worker.driver.init_params.update(**{k: v})
            worker.run(sys.frames)

            # --- retrieve results
            worker.inspect(resubmit=True)
            if worker.get_number_of_running_jobs() == 0:
                ret_frames = worker.retrieve() # TODO: option for re-collect results?
                nframes_ret = len(ret_frames)
                if nframes_in == nframes_ret:
                    self.logger.info(f"{sys.name} finished and writes structures.")
                    write(res_path/(sys.name+".xyz"), ret_frames)
                else:
                    self.logger.info(f"{sys.name} finishes {nframes_ret} out of {nframes_in}.")

        return
    
    def _compress(self):
        """Compress dataset based on Boltzmann distribution."""

        return



if __name__ == "__main__":
    pass
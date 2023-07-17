#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import warnings

class Register:

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            warnings.warn("Key %s already in registry %s." % (key, self._name), UserWarning)
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""
        
        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            return add(None, target)

        return lambda x: add(target, x)

    def __getitem__(self, key):
        if key not in self._dict:
            raise Exception(f"No {key} in {self._dict.keys()}")
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()
    
    def __repr__(self) -> str:
        """"""
        content = f"{self._name.upper()}:\n"

        keys = sorted(list(self._dict.keys()))
        nkeys = len(keys)
        ncols = 5
        nrows = int(nkeys/ncols)
        for i in range(nrows):
            content += ("  "+"{:<24s}"*ncols+"\n").format(*keys[i*ncols:i*ncols+ncols])

        nrest = nkeys - nrows*ncols
        if nrest > 0:
            content += ("  "+"{:<24s}"*nrest+"\n").format(*keys[nrows*ncols:])

        return content

class registers:

    #: Session operations.
    operation: Register = Register("operation")

    #: Session variables.
    variable: Register = Register("variable")

    #: Session placeholder
    placeholder: Register = Register("placeholder")

    #: Schedulers.
    scheduler: Register = Register("scheduler")

    #: Managers (Potentials).
    manager: Register = Register("manager")

    #: Trainers (Potential Trainers).
    trainer: Register = Register("trainer")

    #: Dataloaders (Datasets).
    dataloader: Register = Register("dataloader")

    #: Regions.
    region: Register = Register("region")

    #: Builders.
    builder: Register = Register("builder")

    #: Modifiers.
    modifier: Register = Register("modifier")

    #: Reactors.
    reactor: Register = Register("reactor")

    #: Expeditions.
    expedition: Register = Register("expedition")

    #: Selectors.
    selector: Register = Register("selector")

    #: Describers.
    describer: Register = Register("describer")

    #: Comparators.
    comparator: Register = Register("comparator")

    #: Validators.
    validator: Register = Register("validator")

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")
    
    @staticmethod
    def get(mod_name: str, cls_name: str, convert_name: bool=True, *args, **kwargs):
        """Acquire the target class from modules."""
        # - convert the cls_name by the internal convention
        if convert_name:
            #cls_name = cls_name.capitalize() + mod_name.capitalize()
            cls_name = "".join([x.capitalize() for x in cls_name.strip().split("_")]) + mod_name.capitalize()

        # - get the class
        curr_register = getattr(registers, mod_name)
        target_cls = curr_register[cls_name]

        return target_cls
    
    @staticmethod
    def create(mode_name: str, cls_name: str, convert_name: bool=True, *args, **kwargs):
        """"""
        target_cls = registers.get(mode_name, cls_name, convert_name, *args, **kwargs)
        instance = target_cls(*args, **kwargs)

        return instance

SCHEDULER_MODULES = ["local", "lsf", "pbs", "slurm"]

ALL_MODULES = [
    # - working components.
    # -- schedulers
    ("GDPy.scheduler", SCHEDULER_MODULES),
    # -- managers (potentials)
    ("GDPy.potential", ["managers"]),
    ("GDPy.potential.managers", [
        "vasp", "espresso", "cp2k", 
        "xtb",
        "eam", "emt", "reax", 
        "eann", "deepmd", "lasp", "nequip", "schnet"
    ]),
    # -- trainers (potentials)
    ("GDPy.potential.managers", ["deepmd"]),
    # -- dataloaders (datasets)
    ("GDPy.data", ["dataset"]),
    # -- region
    ("GDPy.builder", ["region"]),
    # -- builders
    ("GDPy", ["builder"]),
    # -- genetic-algorithm-related
    ("GDPy.builder", ["crossover", "mutation"]),
    # -- selectors
    ("GDPy.selector", ["invariant", "interval", "property", "descriptor"]),
    # -- describer
    ("GDPy.describer", ["soap"]),
    # -- comparators
    ("GDPy", ["comparator"]),
    # -- expeditions
    ("GDPy", ["expedition"]),
    # -- reactors
    ("GDPy.reactor", ["afir"]),
    # -- validators
    ("GDPy", ["validator"]),
    # - session operations.
    ("GDPy.data", ["operations"]), 
    ("GDPy.builder", ["interface"]),
    ("GDPy.computation", ["operations"]),
    ("GDPy.reactor", ["interface"]),
    ("GDPy.validator", ["interface"]),
    ("GDPy.selector", ["interface"]),
    # - session variables.
    ("GDPy.data", ["interface"]),
    ("GDPy.validator", ["interface"]),
    ("GDPy.scheduler", ["interface"]),
    ("GDPy.computation", ["interface"]),
    ("GDPy.potential", ["interface"]),
    ("GDPy.describer", ["interface"]),
]


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        warnings.warn("Module {} import failed: {}".format(name, err), UserWarning)
        ...
    
    return


def import_all_modules_for_register(custom_module_paths=None) -> str:
    """Import all modules for register."""
    modules = []
    for base_dir, submodules in ALL_MODULES:
        for name in submodules:
            full_name = base_dir + "." + name
            modules.append(full_name)
    if isinstance(custom_module_paths, list):
        modules += custom_module_paths
    #print("ALL MODULES: ", modules)
    errors = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            errors.append((module, error))
    _handle_errors(errors)

    return

if __name__ == "__main__":
    ...